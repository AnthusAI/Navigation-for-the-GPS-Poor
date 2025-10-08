"""
DRY Flight Path Animator - Generate animated sequences of aircraft flight paths

This module uses the DRY TerrainWindow and FlightPathConfig systems to create
high-quality animated sequences showing the path of simulated aircraft flights.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from .terrain_window import TerrainWindow
from .flight_config import FlightPathConfig


class FlightAnimator:
    """
    Generate animated flight path sequences using DRY terrain projection.

    This class creates high-quality animated GIFs showing aircraft flight paths
    with proper terrain backgrounds, overlays, and timing control.
    """

    def __init__(self):
        """Initialize the flight animator."""
        self.terrain_window = TerrainWindow()

        # Animation parameters (as specified)
        self.target_fps = 120
        self.flight_duration_seconds = 10.0  # Flight before final point
        self.pause_duration_seconds = 5.0    # Pause at final point
        self.final_width = 1200              # Scale to 1200px wide
        self.aspect_ratio = (16, 9)          # 16:9 aspect ratio

        # Calculate frame parameters
        self.flight_frames = int(self.target_fps * self.flight_duration_seconds)  # 1200 frames
        self.pause_frames = int(self.target_fps * self.pause_duration_seconds)    # 600 frames
        self.total_frames = self.flight_frames + self.pause_frames                # 1800 frames

        print(f"Flight Animator initialized:")
        print(f"  Target: {self.target_fps} FPS")
        print(f"  Flight: {self.flight_duration_seconds}s ({self.flight_frames} frames)")
        print(f"  Pause: {self.pause_duration_seconds}s ({self.pause_frames} frames)")
        print(f"  Total: {self.total_frames} frames")
        print(f"  Output: {self.final_width}px wide, {self.aspect_ratio[0]}:{self.aspect_ratio[1]} ratio")

    def _calculate_final_dimensions(self) -> Tuple[int, int]:
        """Calculate final cropped dimensions for 16:9 aspect ratio."""
        width = self.final_width
        height = int(width * self.aspect_ratio[1] / self.aspect_ratio[0])
        return width, height

    def _extract_flight_frames(self, flight_name: str = "main_evaluation",
                             tile_size: int = 800) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Extract terrain frames along the flight path using DRY systems.

        Args:
            flight_name: Name of the flight path to use
            tile_size: Size of terrain tiles to extract

        Returns:
            Tuple of (terrain_frames, flight_coordinates)
        """
        print(f"\n🎬 Extracting terrain frames for flight path...")

        # Get DRY flight configuration
        flight_path = FlightPathConfig.get_flight_path(flight_name)
        flight_info = FlightPathConfig.get_flight_info(flight_name)

        print(f"  Flight: {flight_info['name']}")
        print(f"  Distance: {flight_info['distance_pixels']:.1f} pixels")
        print(f"  Original points: {flight_info['num_points']}")

        # Create high-resolution flight path for smooth animation
        # We need exactly flight_frames points for the animation
        flight_coords_hires = FlightPathConfig.create_flight_coordinates(
            FlightPathConfig.STANDARD_FLIGHT_PATHS[flight_name]._replace(
                num_points=self.flight_frames
            )
        )

        print(f"  Animation points: {len(flight_coords_hires)}")

        # Convert to pixel coordinates for terrain extraction
        pixel_coords = flight_coords_hires * np.array([7500, 7500])

        # Extract terrain frames
        terrain_frames = []
        print(f"  Extracting {len(pixel_coords)} terrain tiles...")

        for i, coord in enumerate(tqdm(pixel_coords, desc="Terrain extraction")):
            try:
                # Use DRY TerrainWindow for consistent extraction
                terrain_tile = self.terrain_window.extract_window(
                    coord[0], coord[1], tile_size
                )
                terrain_frames.append(terrain_tile)

            except ValueError as e:
                # Handle edge cases with neutral terrain
                print(f"    Warning: Frame {i} outside bounds, using neutral tile")
                neutral_tile = np.full((tile_size, tile_size, 3), 128, dtype=np.uint8)
                terrain_frames.append(neutral_tile)

        print(f"  ✅ Extracted {len(terrain_frames)} terrain frames")
        return terrain_frames, flight_coords_hires

    def _scale_and_crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Scale frame to target width and crop to 16:9 aspect ratio.

        Args:
            frame: Input frame as numpy array

        Returns:
            Scaled and cropped frame
        """
        # Convert to PIL for precise scaling
        pil_frame = Image.fromarray(frame.astype(np.uint8))

        # Scale to target width maintaining aspect ratio
        original_width, original_height = pil_frame.size
        scale_factor = self.final_width / original_width
        new_height = int(original_height * scale_factor)

        scaled_frame = pil_frame.resize((self.final_width, new_height), Image.Resampling.LANCZOS)

        # Calculate crop dimensions for 16:9
        target_width, target_height = self._calculate_final_dimensions()

        # Center crop to 16:9
        if new_height > target_height:
            # Crop height
            crop_top = (new_height - target_height) // 2
            crop_box = (0, crop_top, target_width, crop_top + target_height)
        else:
            # Should not happen with our setup, but handle anyway
            crop_box = (0, 0, target_width, new_height)

        cropped_frame = scaled_frame.crop(crop_box)

        # Ensure exact target dimensions
        if cropped_frame.size != (target_width, target_height):
            cropped_frame = cropped_frame.resize((target_width, target_height), Image.Resampling.LANCZOS)

        return np.array(cropped_frame)

    def _add_flight_overlays(self, frame: np.ndarray, frame_index: int,
                           total_flight_frames: int, flight_info: Dict) -> np.ndarray:
        """
        Add flight information overlays to a frame.

        Args:
            frame: Base frame to add overlays to
            frame_index: Current frame index in flight sequence
            total_flight_frames: Total frames in flight sequence
            flight_info: Flight path information

        Returns:
            Frame with overlays added
        """
        # Convert to PIL for overlay operations
        pil_frame = Image.fromarray(frame.astype(np.uint8))
        draw = ImageDraw.Draw(pil_frame)

        # Try to load a font, fallback to default
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
            info_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except (OSError, IOError):
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Frame dimensions
        width, height = pil_frame.size

        # Calculate progress
        progress = frame_index / total_flight_frames
        remaining_frames = total_flight_frames - frame_index
        remaining_seconds = remaining_frames / self.target_fps

        # Title overlay (top)
        title = "Path of Simulated Aircraft Flight"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2

        # Semi-transparent background for title
        title_bg = Image.new('RGBA', (title_width + 40, 50), (0, 0, 0, 180))
        pil_frame.paste(title_bg, (title_x - 20, 10), title_bg)
        draw.text((title_x, 20), title, fill=(255, 255, 255), font=title_font)

        # Progress information (top right)
        progress_text = f"Progress: {progress*100:.1f}%"
        time_text = f"T-{remaining_seconds:.1f}s"

        # Progress background (smaller since no frame number)
        progress_bg = Image.new('RGBA', (200, 60), (0, 0, 0, 150))
        pil_frame.paste(progress_bg, (width - 220, 10), progress_bg)

        draw.text((width - 210, 15), progress_text, fill=(255, 255, 255), font=info_font)
        draw.text((width - 210, 40), time_text, fill=(255, 255, 255), font=info_font)

        # Flight information (bottom left)
        flight_text = f"Flight: {flight_info['name']}"
        route_text = f"Route: ESE Desert → Airbase WNW"

        # Convert pixel distance to approximate real distance
        # Assuming ~1 pixel ≈ 10 meters based on satellite imagery scale
        distance_km = flight_info['distance_pixels'] * 10 / 1000
        distance_text = f"Distance: {distance_km:.1f} km"

        # Flight info background
        flight_bg = Image.new('RGBA', (400, 80), (0, 0, 0, 150))
        pil_frame.paste(flight_bg, (10, height - 90), flight_bg)

        draw.text((20, height - 80), flight_text, fill=(255, 255, 255), font=info_font)
        draw.text((20, height - 55), route_text, fill=(200, 200, 200), font=small_font)
        draw.text((20, height - 30), distance_text, fill=(200, 200, 200), font=small_font)

        # Progress bar (bottom)
        bar_width = width - 40
        bar_height = 8
        bar_x = 20
        bar_y = height - 20

        # Progress bar background
        draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
                      fill=(100, 100, 100))

        # Progress bar fill
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            draw.rectangle([bar_x, bar_y, bar_x + fill_width, bar_y + bar_height],
                          fill=(0, 255, 0))

        return np.array(pil_frame)

    def create_flight_animation(self, flight_name: str = "main_evaluation",
                              tile_size: int = 800, save_path: Optional[str] = None) -> str:
        """
        Create complete flight path animation with specified parameters.

        Args:
            flight_name: Name of the flight path to animate
            tile_size: Size of terrain tiles to extract
            save_path: Path to save the animation GIF

        Returns:
            Path to saved animation file
        """
        print(f"\n🚁 Creating Flight Path Animation")
        print("=" * 50)

        # Get flight information
        flight_info = FlightPathConfig.get_flight_info(flight_name)

        # Extract terrain frames along flight path
        terrain_frames, flight_coords = self._extract_flight_frames(flight_name, tile_size)

        # Process frames for animation
        print(f"\n🎨 Processing frames for animation...")
        processed_frames = []

        # Flight sequence frames
        print(f"  Processing flight frames (1-{self.flight_frames})...")
        for i in tqdm(range(self.flight_frames), desc="Flight frames"):
            # Scale and crop base terrain frame
            scaled_frame = self._scale_and_crop_frame(terrain_frames[i])

            # Add flight overlays
            final_frame = self._add_flight_overlays(
                scaled_frame, i, self.flight_frames, flight_info
            )

            processed_frames.append(Image.fromarray(final_frame.astype(np.uint8)))

        # Pause sequence frames (repeat final frame)
        print(f"  Processing pause frames ({self.flight_frames+1}-{self.total_frames})...")
        final_terrain_frame = self._scale_and_crop_frame(terrain_frames[-1])

        for i in tqdm(range(self.pause_frames), desc="Pause frames"):
            # Add "ARRIVED" overlay for pause frames
            pause_frame = final_terrain_frame.copy()
            pil_frame = Image.fromarray(pause_frame.astype(np.uint8))
            draw = ImageDraw.Draw(pil_frame)

            # Add existing overlays
            overlay_frame = self._add_flight_overlays(
                pause_frame, self.flight_frames - 1, self.flight_frames, flight_info
            )

            # Add "ARRIVED" message
            pil_frame = Image.fromarray(overlay_frame.astype(np.uint8))
            draw = ImageDraw.Draw(pil_frame)

            try:
                arrived_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            except (OSError, IOError):
                arrived_font = ImageFont.load_default()

            arrived_text = "ARRIVED"
            width, height = pil_frame.size
            arrived_bbox = draw.textbbox((0, 0), arrived_text, font=arrived_font)
            arrived_width = arrived_bbox[2] - arrived_bbox[0]
            arrived_x = (width - arrived_width) // 2
            arrived_y = height // 2

            # More transparent red background for arrival message
            arrived_bg = Image.new('RGBA', (arrived_width + 60, 80), (128, 0, 0, 120))
            pil_frame.paste(arrived_bg, (arrived_x - 30, arrived_y - 20), arrived_bg)

            draw.text((arrived_x, arrived_y), arrived_text, fill=(255, 255, 255), font=arrived_font)

            processed_frames.append(pil_frame)

        # Calculate frame duration for target FPS
        frame_duration = int(1000 / self.target_fps)  # milliseconds per frame

        # Save animation
        if save_path is None:
            save_path = f"images/flight_path_animation_{flight_name}.gif"

        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving animation...")
        print(f"  Frames: {len(processed_frames)}")
        print(f"  Duration: {frame_duration}ms per frame ({self.target_fps} FPS)")
        print(f"  Total time: {len(processed_frames) * frame_duration / 1000:.1f} seconds")
        print(f"  Output: {output_path}")

        # Save as animated GIF
        processed_frames[0].save(
            str(output_path),
            save_all=True,
            append_images=processed_frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True
        )

        print(f"✅ Flight animation saved: {output_path}")

        # Summary
        final_width, final_height = self._calculate_final_dimensions()
        print(f"\n📊 Animation Summary:")
        print(f"  Resolution: {final_width}×{final_height} ({self.aspect_ratio[0]}:{self.aspect_ratio[1]})")
        print(f"  Flight: {self.flight_duration_seconds}s @ {self.target_fps} FPS")
        print(f"  Pause: {self.pause_duration_seconds}s @ {self.target_fps} FPS")
        print(f"  Total: {len(processed_frames)} frames, {len(processed_frames) * frame_duration / 1000:.1f}s")
        print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

        return str(output_path)

    def create_preview_frames(self, flight_name: str = "main_evaluation",
                            num_preview_frames: int = 10, save_dir: str = "preview_frames") -> List[str]:
        """
        Create preview frames for testing without generating full animation.

        Args:
            flight_name: Name of the flight path
            num_preview_frames: Number of preview frames to generate
            save_dir: Directory to save preview frames

        Returns:
            List of saved frame paths
        """
        print(f"\n🖼️ Creating {num_preview_frames} preview frames...")

        # Extract subset of terrain frames
        terrain_frames, _ = self._extract_flight_frames(flight_name, 800)
        flight_info = FlightPathConfig.get_flight_info(flight_name)

        # Select evenly spaced frames for preview
        frame_indices = np.linspace(0, len(terrain_frames) - 1, num_preview_frames, dtype=int)

        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, frame_idx in enumerate(frame_indices):
            # Process frame
            scaled_frame = self._scale_and_crop_frame(terrain_frames[frame_idx])
            final_frame = self._add_flight_overlays(
                scaled_frame, frame_idx, len(terrain_frames), flight_info
            )

            # Save preview frame
            preview_path = output_dir / f"preview_frame_{i:03d}.png"
            Image.fromarray(final_frame.astype(np.uint8)).save(preview_path)
            saved_paths.append(str(preview_path))

        print(f"✅ Preview frames saved to: {output_dir}")
        return saved_paths


# Convenience function for easy animation generation
def create_animated_flight_path(flight_name: str = "main_evaluation",
                              save_path: Optional[str] = None) -> str:
    """
    Convenience function to create animated flight path using DRY systems.

    Args:
        flight_name: Name of the flight path to animate
        save_path: Path to save animation

    Returns:
        Path to saved animation
    """
    animator = FlightAnimator()
    return animator.create_flight_animation(flight_name, save_path=save_path)