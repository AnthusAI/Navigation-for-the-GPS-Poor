"""
TerrainWindow: DRY, tested, repeatable utility for extracting terrain windows
from the stitched satellite map at any location and zoom level.

This is the single source of truth for all terrain extraction operations.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union, Dict
import math


class TerrainWindow:
    """
    DRY utility for extracting terrain windows from the stitched satellite map.

    This class provides a single, tested interface for extracting square terrain
    windows at any location and zoom level from the 7500×7500 stitched map.
    All other terrain extraction should use this class to ensure consistency.
    """

    def __init__(self, stitched_map_path: Optional[str] = None):
        """
        Initialize the TerrainWindow extractor.

        Args:
            stitched_map_path: Path to the stitched satellite map
        """
        self.stitched_map = None
        self.map_size = (7500, 7500)  # Standard map size

        # Default paths to try for the stitched map
        default_paths = [
            "../../data/boneyard/davis_monthan_stitched_map.jpg",
            "../data/boneyard/davis_monthan_stitched_map.jpg",
            "data/boneyard/davis_monthan_stitched_map.jpg",
            "davis_monthan_stitched_map.jpg"
        ]

        # Try to load the stitched map
        map_path = stitched_map_path or self._find_stitched_map(default_paths)
        if map_path:
            self.load_stitched_map(map_path)

    def _find_stitched_map(self, paths: list) -> Optional[str]:
        """Find the stitched map from a list of possible paths."""
        for path in paths:
            if Path(path).exists():
                return path
        return None

    def load_stitched_map(self, map_path: str) -> None:
        """
        Load the stitched satellite map.

        Args:
            map_path: Path to the stitched map image

        Raises:
            FileNotFoundError: If the map file doesn't exist
            ValueError: If the map isn't the expected 7500×7500 size
        """
        if not Path(map_path).exists():
            raise FileNotFoundError(f"Stitched map not found: {map_path}")

        # Load and validate the map
        self.stitched_map = np.array(Image.open(map_path))

        if self.stitched_map.shape[:2] != self.map_size:
            print(f"Warning: Map size {self.stitched_map.shape[:2]} differs from expected {self.map_size}")
            self.map_size = self.stitched_map.shape[:2]

        print(f"✅ Stitched map loaded: {self.stitched_map.shape}")

    def extract_window(self, center_x: float, center_y: float,
                      window_size: int, zoom: float = 1.0,
                      aircraft_heading: Optional[float] = None,
                      environmental_effects: Optional[Dict] = None) -> np.ndarray:
        """
        Extract a square terrain window at the specified location and zoom level.

        This is the core DRY method that all other terrain extraction should use.

        Args:
            center_x: X coordinate of window center (in map pixel coordinates)
            center_y: Y coordinate of window center (in map pixel coordinates)
            window_size: Size of the square window to extract (pixels)
            zoom: Zoom factor (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out)
            aircraft_heading: Aircraft heading in degrees (0-360). If provided,
                            rotates terrain to show aircraft perspective where
                            the heading direction is "up" in the extracted image.
            environmental_effects: Dictionary of environmental effects to apply:
                - 'fog_intensity': float 0-1 for atmospheric haze
                - 'brightness': float 0.5-1.5 for lighting variations
                - 'contrast': float 0.5-1.5 for contrast changes
                - 'noise_std': float 0-10 for sensor noise
                - 'motion_blur': float 0-2 for motion blur radius

        Returns:
            Square terrain window with aircraft perspective and environmental effects applied

        Raises:
            RuntimeError: If no stitched map is loaded
            ValueError: If coordinates are outside map bounds
        """
        if self.stitched_map is None:
            raise RuntimeError("No stitched map loaded. Call load_stitched_map() first.")

        # Calculate the extraction area size based on zoom
        # Higher zoom means smaller extraction area that gets resized up
        extraction_size = int(window_size / zoom)

        # Extract terrain window (with rotation if specified)
        if aircraft_heading is not None:
            window = self._extract_rotated_window(
                center_x, center_y, window_size, aircraft_heading, zoom
            )
        else:
            # Standard extraction without rotation
            half_extraction = extraction_size // 2

            # Validate coordinates
            if not self.validate_coordinates(center_x, center_y, half_extraction):
                raise ValueError(f"Coordinates ({center_x}, {center_y}) with extraction size "
                               f"{extraction_size} are outside map bounds {self.map_size}")

            # Calculate extraction bounds
            x_start = max(0, int(center_x - half_extraction))
            x_end = min(self.map_size[1], int(center_x + half_extraction))
            y_start = max(0, int(center_y - half_extraction))
            y_end = min(self.map_size[0], int(center_y + half_extraction))

            # Extract the window
            window = self.stitched_map[y_start:y_end, x_start:x_end]

            # Resize to requested window size if needed
            if window.shape[0] != window_size or window.shape[1] != window_size:
                window = np.array(Image.fromarray(window).resize((window_size, window_size)))

        # Apply environmental effects if specified
        if environmental_effects is not None:
            window = self._apply_environmental_effects(window, environmental_effects)

        return window

    def extract_model_input(self, center_x: float, center_y: float,
                           model_input_size: int = 224,
                           zoom: float = 1.0,
                           aircraft_heading: Optional[float] = None,
                           environmental_effects: Optional[Dict] = None) -> np.ndarray:
        """
        Extract a terrain window formatted for model input.

        This is a convenience method that uses the DRY extract_window method
        but ensures the output is properly formatted for CNN input.

        Args:
            center_x: X coordinate of window center
            center_y: Y coordinate of window center
            model_input_size: Size of model input (default 224 for most CNNs)
            zoom: Zoom factor for altitude simulation
            aircraft_heading: Aircraft heading for realistic perspective (optional)
            environmental_effects: Environmental effects to simulate conditions (optional)

        Returns:
            Terrain window ready for model input (from aircraft perspective with effects applied)
        """
        return self.extract_window(
            center_x, center_y, model_input_size,
            zoom=zoom,
            aircraft_heading=aircraft_heading,
            environmental_effects=environmental_effects
        )

    def extract_context_window(self, center_x: float, center_y: float,
                             context_size: int = 800) -> np.ndarray:
        """
        Extract a larger context window for visualization.

        Args:
            center_x: X coordinate of window center
            center_y: Y coordinate of window center
            context_size: Size of context window (default 800 for error analysis)

        Returns:
            Larger terrain window for visualization context
        """
        return self.extract_window(center_x, center_y, context_size, zoom=1.0)

    def extract_multi_scale_windows(self, center_x: float, center_y: float,
                                   sizes: list = [224, 400, 800]) -> dict:
        """
        Extract multiple terrain windows at different scales.

        Args:
            center_x: X coordinate of window center
            center_y: Y coordinate of window center
            sizes: List of window sizes to extract

        Returns:
            Dictionary mapping size to terrain window
        """
        windows = {}
        for size in sizes:
            windows[size] = self.extract_window(center_x, center_y, size)
        return windows

    def _extract_rotated_window(self, center_x: float, center_y: float,
                               window_size: int, aircraft_heading: float,
                               zoom: float = 1.0) -> np.ndarray:
        """
        Extract terrain window with proper coordinate rotation (no letterboxing).

        Instead of rotating the image after extraction, this method rotates the
        sampling coordinates before extraction, ensuring no black edges.

        Args:
            center_x, center_y: Center coordinates in terrain map
            window_size: Size of output window
            aircraft_heading: Aircraft heading in degrees (0-360)
            zoom: Zoom factor

        Returns:
            Rotated terrain window sampled from original map
        """
        from scipy import ndimage
        import cv2

        # Calculate effective sample size accounting for zoom
        sample_size = int(window_size / zoom)

        # Create coordinate grids for the output window
        # These represent the pixel positions in the final image
        y_coords, x_coords = np.mgrid[0:sample_size, 0:sample_size]

        # Center coordinates around origin
        x_coords = x_coords - sample_size // 2
        y_coords = y_coords - sample_size // 2

        # Convert heading to radians and create rotation matrix
        # We want the aircraft's forward direction to appear as "up" in the rotated image
        angle_rad = np.radians(aircraft_heading)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Apply rotation to coordinates
        x_rotated = cos_a * x_coords - sin_a * y_coords
        y_rotated = sin_a * x_coords + cos_a * y_coords

        # Translate to actual center position
        x_sample = x_rotated + center_x
        y_sample = y_rotated + center_y

        # Validate that we're sampling within map bounds
        margin = sample_size // 2 + 10  # Extra margin for rotation
        if not self.validate_coordinates(center_x, center_y, margin):
            raise ValueError(f"Coordinates ({center_x}, {center_y}) too close to map bounds for rotation")

        # Sample from original terrain map using bilinear interpolation
        # This ensures smooth results without black edges
        rotated_window = np.zeros((sample_size, sample_size, 3), dtype=np.uint8)

        for c in range(3):  # RGB channels
            rotated_window[:, :, c] = ndimage.map_coordinates(
                self.stitched_map[:, :, c],
                [y_sample, x_sample],
                order=1,  # Bilinear interpolation
                mode='nearest',  # Use nearest edge values for out-of-bounds
                prefilter=False
            ).astype(np.uint8)

        # Resize to final window size if needed
        if sample_size != window_size:
            rotated_window = cv2.resize(rotated_window, (window_size, window_size))

        return rotated_window

    def _apply_environmental_effects(self, terrain_window: np.ndarray,
                                   effects: Dict) -> np.ndarray:
        """
        Apply environmental effects to terrain window.

        Args:
            terrain_window: Input terrain window (H, W, 3)
            effects: Dictionary of effects to apply

        Returns:
            Terrain window with environmental effects applied
        """
        import cv2
        from PIL import Image, ImageEnhance

        # Convert to PIL Image for easier manipulation
        window = terrain_window.copy().astype(np.float32)

        # Apply fog/haze effect
        if 'fog_intensity' in effects and effects['fog_intensity'] > 0:
            fog_intensity = np.clip(effects['fog_intensity'], 0, 1)
            # Create fog overlay (uniform gray)
            fog_color = np.array([200, 200, 200], dtype=np.float32)
            window = window * (1 - fog_intensity) + fog_color * fog_intensity

        # Apply brightness adjustment
        if 'brightness' in effects:
            brightness = np.clip(effects['brightness'], 0.1, 3.0)
            window = window * brightness

        # Apply contrast adjustment
        if 'contrast' in effects:
            contrast = np.clip(effects['contrast'], 0.1, 3.0)
            # Contrast adjustment around middle gray
            window = (window - 127.5) * contrast + 127.5

        # Apply motion blur
        if 'motion_blur' in effects and effects['motion_blur'] > 0:
            blur_radius = effects['motion_blur']
            kernel_size = max(1, int(blur_radius * 2))
            # Ensure kernel size is odd and positive
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size >= 3:  # Only apply blur if kernel is meaningful
                window = cv2.GaussianBlur(window, (kernel_size, kernel_size), blur_radius)

        # Apply sensor noise
        if 'noise_std' in effects and effects['noise_std'] > 0:
            noise_std = effects['noise_std']
            noise = np.random.normal(0, noise_std, window.shape).astype(np.float32)
            window = window + noise

        # Clip to valid pixel range and convert back to uint8
        window = np.clip(window, 0, 255).astype(np.uint8)

        return window

    def _rotate_for_aircraft_perspective(self, terrain_patch: np.ndarray,
                                       aircraft_heading: float) -> np.ndarray:
        """
        Rotate terrain patch to show aircraft perspective.

        The aircraft heading direction becomes "up" in the rotated image,
        simulating what the aircraft camera would actually see.

        Args:
            terrain_patch: Input terrain patch to rotate
            aircraft_heading: Aircraft heading in degrees (0-360)
                            0 = North, 90 = East, 180 = South, 270 = West

        Returns:
            Rotated terrain patch with aircraft heading as "up"
        """
        # Convert terrain patch to PIL Image for rotation
        pil_image = Image.fromarray(terrain_patch)

        # Calculate rotation angle
        # We want the aircraft heading direction to be "up" (north) in the final image
        # PIL rotate() rotates counter-clockwise, and we want to rotate the terrain
        # so that the aircraft's heading direction appears as "up"

        # If aircraft is heading 90° (east), we need to rotate terrain -90° (clockwise)
        # so that east becomes "up" in the image
        rotation_angle = -aircraft_heading

        # Rotate with black fill for areas outside the original image
        rotated_image = pil_image.rotate(
            rotation_angle,
            expand=False,  # Don't expand canvas
            fillcolor=(0, 0, 0),  # Black fill for missing areas
            resample=Image.BICUBIC  # High quality interpolation
        )

        # Convert back to numpy array
        rotated_terrain = np.array(rotated_image)

        # Crop to center square to remove any rotation artifacts at edges
        h, w = rotated_terrain.shape[:2]
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2

        cropped_terrain = rotated_terrain[
            start_y:start_y + crop_size,
            start_x:start_x + crop_size
        ]

        return cropped_terrain

    def validate_coordinates(self, x: float, y: float, margin: int = 0) -> bool:
        """
        Validate that coordinates are within map bounds.

        Args:
            x, y: Coordinates to validate
            margin: Additional margin to check (for extraction size)

        Returns:
            True if coordinates are valid
        """
        return (margin <= x < self.map_size[1] - margin and
                margin <= y < self.map_size[0] - margin)

    def get_map_info(self) -> dict:
        """
        Get information about the loaded map.

        Returns:
            Dictionary with map information
        """
        return {
            "loaded": self.stitched_map is not None,
            "size": self.map_size,
            "shape": self.stitched_map.shape if self.stitched_map is not None else None,
            "dtype": self.stitched_map.dtype if self.stitched_map is not None else None
        }

    @staticmethod
    def create_flight_path_windows(flight_name: str = "main_evaluation",
                                  window_size: int = 224) -> Tuple[list, np.ndarray]:
        """
        Create terrain windows along a DRY-configured flight path.

        Args:
            flight_name: Name of the standard flight path to use
            window_size: Size of each window

        Returns:
            Tuple of (terrain_windows_list, flight_coordinates_array)
        """
        # Import here to avoid circular imports
        from .flight_config import FlightPathConfig

        # Use DRY flight configuration
        flight_path = FlightPathConfig.get_flight_path(flight_name)
        flight_coords = FlightPathConfig.create_flight_coordinates(flight_path)

        # Convert normalized coordinates to pixel coordinates for extraction
        pixel_coords = flight_coords * np.array([7500, 7500])

        # Extract windows at each point
        terrain_window = TerrainWindow()
        if terrain_window.stitched_map is None:
            raise RuntimeError("Could not load stitched map for flight path extraction")

        windows = []
        for coord in pixel_coords:
            try:
                window = terrain_window.extract_window(coord[0], coord[1], window_size)
                windows.append(window)
            except ValueError:
                # If coordinates are outside bounds, create a neutral window
                neutral = np.full((window_size, window_size, 3), 128, dtype=np.uint8)
                windows.append(neutral)

        return windows, flight_coords


# Convenience function for backwards compatibility
def extract_terrain_window(center_x: float, center_y: float,
                          window_size: int, zoom: float = 1.0) -> np.ndarray:
    """
    Convenience function for extracting terrain windows.
    Uses the DRY TerrainWindow class internally.
    """
    terrain_window = TerrainWindow()
    return terrain_window.extract_window(center_x, center_y, window_size, zoom)