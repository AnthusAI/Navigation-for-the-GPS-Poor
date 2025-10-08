"""
TerrainWindow: DRY, tested, repeatable utility for extracting terrain windows
from the stitched satellite map at any location and zoom level.

This is the single source of truth for all terrain extraction operations.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union
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
                      window_size: int, zoom: float = 1.0) -> np.ndarray:
        """
        Extract a square terrain window at the specified location and zoom level.

        This is the core DRY method that all other terrain extraction should use.

        Args:
            center_x: X coordinate of window center (in map pixel coordinates)
            center_y: Y coordinate of window center (in map pixel coordinates)
            window_size: Size of the square window to extract (pixels)
            zoom: Zoom factor (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out)

        Returns:
            Square terrain window as numpy array

        Raises:
            RuntimeError: If no stitched map is loaded
            ValueError: If coordinates are outside map bounds
        """
        if self.stitched_map is None:
            raise RuntimeError("No stitched map loaded. Call load_stitched_map() first.")

        # Calculate the extraction area size based on zoom
        # Higher zoom means smaller extraction area that gets resized up
        extraction_size = int(window_size / zoom)
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

        return window

    def extract_model_input(self, center_x: float, center_y: float,
                           model_input_size: int = 224) -> np.ndarray:
        """
        Extract a terrain window formatted for model input.

        This is a convenience method that uses the DRY extract_window method
        but ensures the output is properly formatted for CNN input.

        Args:
            center_x: X coordinate of window center
            center_y: Y coordinate of window center
            model_input_size: Size of model input (default 224 for most CNNs)

        Returns:
            Terrain window ready for model input
        """
        return self.extract_window(center_x, center_y, model_input_size, zoom=1.0)

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