"""
DRY Flight Path Configuration - Single source of truth for all flight paths

This module defines standard flight paths used across all navigation system
components including training, evaluation, and visualization.
"""
import numpy as np
from typing import Tuple, Dict, List, NamedTuple


class FlightPath(NamedTuple):
    """
    Structured flight path definition.

    Attributes:
        name: Descriptive name for this flight path
        start_coords: Starting coordinates (x, y) in normalized space [0,1]
        end_coords: Ending coordinates (x, y) in normalized space [0,1]
        description: Description of the flight route
        num_points: Default number of points along the path
        start_pixel: Starting coordinates in pixel space (7500x7500)
        end_pixel: Ending coordinates in pixel space (7500x7500)
    """
    name: str
    start_coords: Tuple[float, float]
    end_coords: Tuple[float, float]
    description: str
    num_points: int = 150
    start_pixel: Tuple[int, int] = None
    end_pixel: Tuple[int, int] = None


class FlightPathConfig:
    """
    Central configuration for all standard flight paths.

    This is the DRY source of truth that all navigation code should use
    for consistent flight path definitions.
    """

    # Map dimensions for coordinate conversion
    MAP_SIZE = (7500, 7500)

    # Standard flight paths used throughout the system
    STANDARD_FLIGHT_PATHS: Dict[str, FlightPath] = {

        # Main evaluation flight: East South-East to West North-West toward Boneyard
        # This should be the longer, more realistic flight path you requested
        "main_evaluation": FlightPath(
            name="Main Evaluation Flight",
            start_coords=(0.85, 0.70),   # Further East South-East (deeper desert area)
            end_coords=(0.555, 0.555),   # Boneyard (Davis-Monthan AFB)
            description="Primary evaluation flight from deep desert ESE to Boneyard WNW",
            num_points=400,  # Double the length = double the points
            start_pixel=(6375, 5250),    # 0.85*7500, 0.70*7500
            end_pixel=(4167, 4167)       # 0.555*7500, 0.555*7500
        ),

        # Shorter demonstration flight for quick tests
        "demo_flight": FlightPath(
            name="Quick Demo Flight",
            start_coords=(0.65, 0.50),   # Closer starting point
            end_coords=(0.555, 0.555),   # Same Boneyard endpoint
            description="Shorter demo flight for quick testing",
            num_points=150,
            start_pixel=(4875, 3750),    # 0.65*7500, 0.50*7500
            end_pixel=(4167, 4167)       # 0.555*7500, 0.555*7500
        ),

        # Cross-base flight for diversity testing
        "cross_base": FlightPath(
            name="Cross-Base Flight",
            start_coords=(0.70, 0.40),   # North-East approach
            end_coords=(0.50, 0.60),     # South-West exit
            description="Flight crossing the entire base area",
            num_points=180,
            start_pixel=(5250, 3000),    # 0.70*7500, 0.40*7500
            end_pixel=(3750, 4500)       # 0.50*7500, 0.60*7500
        )
    }

    @classmethod
    def get_flight_path(cls, name: str) -> FlightPath:
        """
        Get a standard flight path by name.

        Args:
            name: Name of the flight path

        Returns:
            FlightPath configuration

        Raises:
            KeyError: If flight path name doesn't exist
        """
        if name not in cls.STANDARD_FLIGHT_PATHS:
            available = list(cls.STANDARD_FLIGHT_PATHS.keys())
            raise KeyError(f"Flight path '{name}' not found. Available: {available}")

        return cls.STANDARD_FLIGHT_PATHS[name]

    @classmethod
    def get_default_flight_path(cls) -> FlightPath:
        """Get the default flight path for evaluation."""
        return cls.get_flight_path("main_evaluation")

    @classmethod
    def create_flight_coordinates(cls, flight_path: FlightPath) -> np.ndarray:
        """
        Create coordinate array for a flight path.

        Args:
            flight_path: FlightPath configuration

        Returns:
            Array of (x, y) coordinates in normalized space
        """
        start = np.array(flight_path.start_coords)
        end = np.array(flight_path.end_coords)

        # Create linear interpolation
        t_values = np.linspace(0, 1, flight_path.num_points)
        coordinates = np.array([start + t * (end - start) for t in t_values])

        return coordinates

    @classmethod
    def create_pixel_coordinates(cls, flight_path: FlightPath) -> np.ndarray:
        """
        Create pixel coordinate array for a flight path.

        Args:
            flight_path: FlightPath configuration

        Returns:
            Array of (x, y) coordinates in pixel space
        """
        normalized_coords = cls.create_flight_coordinates(flight_path)
        pixel_coords = normalized_coords * np.array(cls.MAP_SIZE)
        return pixel_coords.astype(int)

    @classmethod
    def get_flight_info(cls, name: str) -> Dict:
        """
        Get comprehensive information about a flight path.

        Args:
            name: Flight path name

        Returns:
            Dictionary with flight path details
        """
        flight_path = cls.get_flight_path(name)
        coordinates = cls.create_flight_coordinates(flight_path)
        pixel_coords = cls.create_pixel_coordinates(flight_path)

        # Calculate flight distance
        total_distance_norm = np.linalg.norm(
            np.array(flight_path.end_coords) - np.array(flight_path.start_coords)
        )
        total_distance_pixels = np.linalg.norm(
            pixel_coords[-1] - pixel_coords[0]
        )

        return {
            "name": flight_path.name,
            "description": flight_path.description,
            "start_normalized": flight_path.start_coords,
            "end_normalized": flight_path.end_coords,
            "start_pixel": tuple(pixel_coords[0]),
            "end_pixel": tuple(pixel_coords[-1]),
            "num_points": flight_path.num_points,
            "distance_normalized": float(total_distance_norm),
            "distance_pixels": float(total_distance_pixels),
            "coordinates_shape": coordinates.shape,
            "direction": f"{flight_path.start_coords} → {flight_path.end_coords}"
        }

    @classmethod
    def list_available_flights(cls) -> List[str]:
        """List all available flight path names."""
        return list(cls.STANDARD_FLIGHT_PATHS.keys())

    @classmethod
    def validate_flight_coordinates(cls, coords: np.ndarray) -> bool:
        """
        Validate that flight coordinates are within valid ranges.

        Args:
            coords: Array of normalized coordinates

        Returns:
            True if all coordinates are valid
        """
        return np.all((coords >= 0) & (coords <= 1))


# Convenience functions for backwards compatibility
def get_default_flight_path() -> Tuple[Tuple[float, float], Tuple[float, float], int]:
    """
    Get default flight path in legacy format.

    Returns:
        Tuple of (start_coords, end_coords, num_points)
    """
    flight = FlightPathConfig.get_default_flight_path()
    return flight.start_coords, flight.end_coords, flight.num_points


def create_standard_flight_coordinates(flight_name: str = "main_evaluation") -> np.ndarray:
    """
    Create flight coordinates using DRY configuration.

    Args:
        flight_name: Name of the standard flight path to use

    Returns:
        Array of normalized flight coordinates
    """
    flight_path = FlightPathConfig.get_flight_path(flight_name)
    return FlightPathConfig.create_flight_coordinates(flight_path)