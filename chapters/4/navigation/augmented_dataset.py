"""
Augmented Training Dataset for Robust Navigation Model Training

This module implements the on-the-fly data augmentation strategy outlined in the
AUGMENTATION_PLAN.md. It provides enhanced training datasets that apply realistic
transformations to create robust, rotation, scale, and noise-invariant models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from typing import Tuple, List, Dict, Optional, Union
import math
from pathlib import Path

from .flight_config import FlightPathConfig


class RobustNavigationDataset(Dataset):
    """
    Enhanced dataset implementing the complete augmentation strategy for robust navigation.

    This dataset applies on-the-fly augmentations to training samples including:
    - Dynamic orientation (rotation based on aircraft heading)
    - Variable scale (altitude simulation)
    - Environmental and sensor noise effects

    This forces the model to learn underlying terrain features rather than
    memorizing superficial patterns from idealized imagery.
    """

    def __init__(self,
                 tiles: List[np.ndarray],
                 coordinates: List[Tuple[float, float]],
                 flight_name: str = "main_evaluation",
                 enable_rotation: bool = True,
                 enable_scale: bool = True,
                 enable_environmental_noise: bool = True,
                 rotation_range: Tuple[float, float] = (0.0, 360.0),
                 scale_range: Tuple[float, float] = (0.7, 1.4),
                 noise_probability: float = 0.7,
                 use_dry_extraction: bool = True):
        """
        Initialize robust navigation dataset.

        Args:
            tiles: List of terrain tile arrays (H, W, C) - not used if use_dry_extraction=True
            coordinates: List of normalized coordinates (x, y)
            flight_name: Name of flight path for heading calculation
            enable_rotation: Enable dynamic orientation augmentation
            enable_scale: Enable variable scale augmentation
            enable_environmental_noise: Enable environmental/sensor noise
            rotation_range: Range of rotation angles in degrees
            scale_range: Range of scale factors (altitude simulation)
            noise_probability: Probability of applying noise effects
            use_dry_extraction: If True, use TerrainWindow for on-the-fly extraction (recommended)
        """
        self.tiles = tiles if not use_dry_extraction else None
        self.coordinates = coordinates
        self.use_dry_extraction = use_dry_extraction
        self.flight_name = flight_name
        self.enable_rotation = enable_rotation
        self.enable_scale = enable_scale
        self.enable_environmental_noise = enable_environmental_noise
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_probability = noise_probability

        # Calculate flight path for heading information
        self._setup_flight_path()

        # Initialize TerrainWindow for DRY extraction
        if self.use_dry_extraction:
            from .terrain_window import TerrainWindow
            self.terrain_window = TerrainWindow()
        else:
            self.terrain_window = None

        # Base normalization transform
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        print(f"✅ RobustNavigationDataset initialized")
        print(f"   Samples: {len(self.tiles) if self.tiles else len(self.coordinates)}")
        print(f"   Flight path: {flight_name}")
        print(f"   DRY extraction: {use_dry_extraction}")
        print(f"   Rotation augmentation: {enable_rotation}")
        print(f"   Scale augmentation: {enable_scale}")
        print(f"   Environmental noise: {enable_environmental_noise}")

    def _setup_flight_path(self):
        """Setup flight path coordinates for heading calculations."""
        flight_path = FlightPathConfig.get_flight_path(self.flight_name)
        self.flight_coordinates = FlightPathConfig.create_flight_coordinates(flight_path)

        # Calculate headings along the flight path
        self.flight_headings = self._calculate_headings()

    def _calculate_headings(self) -> np.ndarray:
        """
        Calculate aircraft headings along the flight path.

        Returns:
            Array of heading angles in degrees
        """
        coords = self.flight_coordinates

        # Calculate direction vectors between consecutive points
        direction_vectors = np.diff(coords, axis=0)

        # Calculate heading angles (in degrees from north)
        # Note: atan2(dy, dx) gives angle from east, we want from north
        headings_rad = np.arctan2(direction_vectors[:, 0], -direction_vectors[:, 1])
        headings_deg = np.degrees(headings_rad)

        # Normalize to [0, 360] range
        headings_deg = (headings_deg + 360) % 360

        # Add final heading (same as previous)
        headings_deg = np.append(headings_deg, headings_deg[-1])

        return headings_deg

    def _get_aircraft_heading(self, coordinate: Tuple[float, float]) -> float:
        """
        Get aircraft heading for a given coordinate based on flight path.

        Args:
            coordinate: Normalized (x, y) coordinate

        Returns:
            Aircraft heading in degrees
        """
        # Find closest point on flight path
        coord_array = np.array(coordinate)
        distances = np.linalg.norm(self.flight_coordinates - coord_array, axis=1)
        closest_idx = np.argmin(distances)

        return self.flight_headings[closest_idx]

    def _apply_rotation_augmentation(self, image: Image.Image, coordinate: Tuple[float, float]) -> Image.Image:
        """
        Apply rotation variation (small turbulence effects only).

        NOTE: Primary rotation for aircraft perspective is now handled at terrain extraction
        level in TerrainWindow. This method only adds small random variations to simulate
        aircraft turbulence and attitude variations.

        Args:
            image: PIL Image (already rotated to aircraft perspective)
            coordinate: Normalized coordinate (not used for heading calculation anymore)

        Returns:
            PIL Image with small turbulence rotation applied
        """
        if not self.enable_rotation:
            return image

        # Apply only small random rotation variations to simulate turbulence
        # The main aircraft heading rotation is now handled during terrain extraction
        turbulence_variation = random.uniform(-5, 5)  # ±5 degrees for turbulence

        return image.rotate(turbulence_variation, expand=False, fillcolor=(0, 0, 0))

    def _apply_scale_augmentation(self, image: Image.Image) -> Image.Image:
        """
        Apply variable scale augmentation to simulate altitude changes.

        Args:
            image: PIL Image to scale

        Returns:
            Scaled PIL Image
        """
        if not self.enable_scale:
            return image

        # Random scale factor within specified range
        scale_factor = random.uniform(*self.scale_range)

        # Get original size
        width, height = image.size

        # Calculate new size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize image
        scaled_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Crop or pad to original size
        if scale_factor > 1.0:
            # Crop to original size (zoomed in - lower altitude)
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            scaled_image = scaled_image.crop((left, top, left + width, top + height))
        else:
            # Pad to original size (zoomed out - higher altitude)
            pad_x = (width - new_width) // 2
            pad_y = (height - new_height) // 2

            # Create new image with black padding
            padded_image = Image.new('RGB', (width, height), (0, 0, 0))
            padded_image.paste(scaled_image, (pad_x, pad_y))
            scaled_image = padded_image

        return scaled_image

    def _apply_atmospheric_effects(self, image: Image.Image) -> Image.Image:
        """
        Apply atmospheric effects like fog and haze.

        Args:
            image: PIL Image

        Returns:
            Image with atmospheric effects
        """
        # Create semi-transparent fog overlay
        fog_intensity = random.uniform(0.1, 0.4)  # Variable fog intensity
        fog_color = random.choice([(200, 200, 200), (180, 180, 180), (220, 220, 220)])

        # Create fog overlay
        fog_overlay = Image.new('RGB', image.size, fog_color)

        # Blend with original image
        return Image.blend(image, fog_overlay, fog_intensity)

    def _apply_lighting_effects(self, image: Image.Image) -> Image.Image:
        """
        Apply lighting and glare effects to simulate different times of day.

        Args:
            image: PIL Image

        Returns:
            Image with lighting effects
        """
        # Random brightness adjustment (different times of day)
        brightness_factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # Random contrast adjustment
        contrast_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        # Random saturation adjustment
        saturation_factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)

        # Simulate lens flare (occasionally)
        if random.random() < 0.1:  # 10% chance of lens flare
            image = self._add_lens_flare(image)

        return image

    def _add_lens_flare(self, image: Image.Image) -> Image.Image:
        """
        Add simulated lens flare effect.

        Args:
            image: PIL Image

        Returns:
            Image with lens flare
        """
        # Simple lens flare simulation - add bright spot
        width, height = image.size
        flare_x = random.randint(width // 4, 3 * width // 4)
        flare_y = random.randint(height // 4, 3 * height // 4)

        # Create circular bright spot
        flare_overlay = Image.new('RGB', image.size, (0, 0, 0))
        pixels = flare_overlay.load()

        flare_radius = random.randint(20, 40)
        flare_intensity = random.randint(100, 200)

        for x in range(max(0, flare_x - flare_radius), min(width, flare_x + flare_radius)):
            for y in range(max(0, flare_y - flare_radius), min(height, flare_y + flare_radius)):
                distance = math.sqrt((x - flare_x) ** 2 + (y - flare_y) ** 2)
                if distance <= flare_radius:
                    intensity = int(flare_intensity * (1 - distance / flare_radius))
                    pixels[x, y] = (intensity, intensity, intensity)

        return Image.blend(image, flare_overlay, 0.3)

    def _apply_motion_blur(self, image: Image.Image, coordinate: Tuple[float, float]) -> Image.Image:
        """
        Apply directional motion blur consistent with aircraft heading.

        Args:
            image: PIL Image
            coordinate: Normalized coordinate for heading lookup

        Returns:
            Image with motion blur
        """
        # Get aircraft heading for directional blur
        heading = self._get_aircraft_heading(coordinate)

        # Convert heading to blur direction
        blur_strength = random.uniform(0.5, 1.5)

        # Create motion blur kernel based on direction
        # This is a simplified implementation - in practice you might use more sophisticated kernels
        if random.random() < 0.3:  # 30% chance of motion blur
            blur_radius = random.uniform(0.5, 1.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return image

    def _apply_sensor_noise(self, image: Image.Image) -> Image.Image:
        """
        Apply sensor noise to simulate camera imperfections.

        Args:
            image: PIL Image

        Returns:
            Image with sensor noise
        """
        # Convert to numpy for noise addition
        img_array = np.array(image)

        # Add Gaussian noise
        noise_std = random.uniform(2, 8)  # Noise standard deviation
        noise = np.random.normal(0, noise_std, img_array.shape).astype(np.float32)

        # Add noise and clip to valid range
        noisy_array = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_array)

    def _apply_environmental_noise(self, image: Image.Image, coordinate: Tuple[float, float]) -> Image.Image:
        """
        Apply complete environmental and sensor noise pipeline.

        Args:
            image: PIL Image
            coordinate: Normalized coordinate

        Returns:
            Image with environmental effects
        """
        if not self.enable_environmental_noise or random.random() > self.noise_probability:
            return image

        # Randomly apply different effects
        effects = []

        if random.random() < 0.6:  # 60% chance of atmospheric effects
            effects.append('atmospheric')

        if random.random() < 0.8:  # 80% chance of lighting effects
            effects.append('lighting')

        if random.random() < 0.4:  # 40% chance of motion blur
            effects.append('motion_blur')

        if random.random() < 0.5:  # 50% chance of sensor noise
            effects.append('sensor_noise')

        # Apply selected effects
        for effect in effects:
            if effect == 'atmospheric':
                image = self._apply_atmospheric_effects(image)
            elif effect == 'lighting':
                image = self._apply_lighting_effects(image)
            elif effect == 'motion_blur':
                image = self._apply_motion_blur(image, coordinate)
            elif effect == 'sensor_noise':
                image = self._apply_sensor_noise(image)

        return image

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get augmented sample at given index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (augmented_image_tensor, coordinate_tensor)
        """
        coordinate = self.coordinates[idx]

        if self.use_dry_extraction:
            # Use DRY TerrainWindow extraction with all augmentations handled properly
            image_tensor = self._extract_with_dry_augmentation(coordinate)
        else:
            # Legacy path: use pre-extracted tiles with post-processing augmentations
            tile = self.tiles[idx]
            image_tensor = self._extract_with_legacy_augmentation(tile, coordinate)

        # Convert coordinate to tensor
        coordinate_tensor = torch.tensor(coordinate, dtype=torch.float32)

        return image_tensor, coordinate_tensor

    def _extract_with_dry_augmentation(self, coordinate: Tuple[float, float]) -> torch.Tensor:
        """
        Extract terrain using DRY TerrainWindow with proper augmentation.

        This eliminates letterboxing by handling rotation at the coordinate level.
        """
        # Convert normalized coordinates to pixel coordinates
        pixel_x = coordinate[0] * 7500
        pixel_y = coordinate[1] * 7500

        # Calculate aircraft heading for this location
        aircraft_heading = self._get_aircraft_heading(coordinate) if self.enable_rotation else None

        # Add small random variation to heading for turbulence simulation
        if aircraft_heading is not None:
            turbulence = random.uniform(-5, 5)  # ±5 degrees for realistic variation
            aircraft_heading += turbulence

        # Calculate zoom factor for scale augmentation
        zoom = random.uniform(*self.scale_range) if self.enable_scale else 1.0

        # Prepare environmental effects
        environmental_effects = {}

        # Check if custom segment effects are specified
        if hasattr(self.terrain_window, '_segment_effects'):
            environmental_effects = self.terrain_window._segment_effects.copy()
        elif self.enable_environmental_noise and random.random() < self.noise_probability:
            # Randomly apply various environmental effects
            if random.random() < 0.6:  # 60% chance of fog
                environmental_effects['fog_intensity'] = random.uniform(0.1, 0.4)

            if random.random() < 0.8:  # 80% chance of lighting changes
                environmental_effects['brightness'] = random.uniform(0.7, 1.3)
                environmental_effects['contrast'] = random.uniform(0.8, 1.2)

            if random.random() < 0.4:  # 40% chance of motion blur
                environmental_effects['motion_blur'] = random.uniform(0.5, 1.5)

            if random.random() < 0.5:  # 50% chance of sensor noise
                environmental_effects['noise_std'] = random.uniform(2, 8)

        # Extract terrain using enhanced DRY TerrainWindow
        terrain_image = self.terrain_window.extract_model_input(
            pixel_x, pixel_y,
            model_input_size=224,
            aircraft_heading=aircraft_heading,
            environmental_effects=environmental_effects if environmental_effects else None
        )

        # Convert to tensor and normalize
        image_tensor = TF.to_tensor(terrain_image)
        image_tensor = self.normalize(image_tensor)

        return image_tensor

    def _extract_with_legacy_augmentation(self, tile: np.ndarray,
                                        coordinate: Tuple[float, float]) -> torch.Tensor:
        """
        Legacy augmentation path using pre-extracted tiles (may cause letterboxing).
        """
        # Convert numpy array to PIL Image
        if isinstance(tile, np.ndarray):
            # Ensure proper format (H, W, C) with values in [0, 255]
            if tile.dtype != np.uint8:
                tile = (tile * 255).astype(np.uint8) if tile.max() <= 1.0 else tile.astype(np.uint8)
            image = Image.fromarray(tile)
        else:
            image = tile

        # Apply augmentation pipeline
        # 1. Dynamic Orientation (Rotation based on aircraft heading)
        image = self._apply_rotation_augmentation(image, coordinate)

        # 2. Variable Scale (Altitude simulation)
        image = self._apply_scale_augmentation(image)

        # 3. Environmental and Sensor Noise
        image = self._apply_environmental_noise(image, coordinate)

        # Convert to tensor and normalize
        image_tensor = TF.to_tensor(image)
        image_tensor = self.normalize(image_tensor)

        return image_tensor


def create_augmented_dataset(tiles: List[np.ndarray],
                           coordinates: List[Tuple[float, float]],
                           flight_name: str = "main_evaluation",
                           **augmentation_kwargs) -> RobustNavigationDataset:
    """
    Convenience function to create augmented dataset.

    Args:
        tiles: List of terrain tile arrays
        coordinates: List of normalized coordinates
        flight_name: Name of flight path for heading calculation
        **augmentation_kwargs: Additional augmentation parameters

    Returns:
        RobustNavigationDataset instance
    """
    return RobustNavigationDataset(
        tiles=tiles,
        coordinates=coordinates,
        flight_name=flight_name,
        **augmentation_kwargs
    )