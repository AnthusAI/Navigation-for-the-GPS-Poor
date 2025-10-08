"""
TerrainExtractor: Extract terrain tiles from satellite maps
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union, List
import pickle

from .utils import CoordinateSystem, validate_coordinates


class TerrainExtractor:
    """
    Extract terrain tiles from satellite imagery or pre-generated datasets.

    This class provides utilities to extract terrain images at specific
    coordinates, either from a full satellite map or from pre-processed datasets.
    """

    def __init__(self, map_path: Optional[str] = None, dataset_path: Optional[str] = None,
                 map_size: Tuple[int, int] = (7500, 7500)):
        """
        Initialize the TerrainExtractor.

        Args:
            map_path: Path to full satellite map image
            dataset_path: Path to pre-processed dataset (.pkl)
            map_size: Size of the satellite map (width, height)
        """
        self.map_size = map_size
        self.coord_system = CoordinateSystem(map_size)

        # Storage for different data sources
        self.satellite_map = None
        self.dataset_tiles = None
        self.dataset_coords = None

        # Load satellite map if provided
        if map_path and Path(map_path).exists():
            self.load_satellite_map(map_path)

        # Load dataset if provided
        if dataset_path and Path(dataset_path).exists():
            self.load_dataset(dataset_path)

    def load_satellite_map(self, map_path: str) -> None:
        """
        Load the full satellite map image.

        Args:
            map_path: Path to satellite map image file
        """
        if not Path(map_path).exists():
            raise FileNotFoundError(f"Satellite map not found: {map_path}")

        self.satellite_map = np.array(Image.open(map_path))
        print(f"✅ Satellite map loaded: {self.satellite_map.shape}")

    def load_dataset(self, dataset_path: str) -> None:
        """
        Load pre-processed terrain dataset.

        Args:
            dataset_path: Path to dataset pickle file
        """
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        self.dataset_tiles = data['tiles']
        self.dataset_coords = data['coordinates']

        if isinstance(self.dataset_tiles, list):
            self.dataset_tiles = np.array(self.dataset_tiles)
            self.dataset_coords = np.array(self.dataset_coords)

        print(f"✅ Dataset loaded: {len(self.dataset_tiles)} tiles")
        print(f"   Tile shape: {self.dataset_tiles[0].shape}")
        print(f"   Coordinate range: X[{self.dataset_coords[:, 0].min():.3f}, {self.dataset_coords[:, 0].max():.3f}], "
              f"Y[{self.dataset_coords[:, 1].min():.3f}, {self.dataset_coords[:, 1].max():.3f}]")

    def extract_tile(self, x: int, y: int, size: int = 224,
                    source: str = "auto") -> np.ndarray:
        """
        Extract a terrain tile at specified coordinates.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels
            size: Size of the square tile to extract
            source: Data source ("satellite", "dataset", or "auto")

        Returns:
            Terrain tile as numpy array
        """
        if source == "auto":
            # Prefer dataset if available, then satellite map
            if self.dataset_tiles is not None:
                source = "dataset"
            elif self.satellite_map is not None:
                source = "satellite"
            else:
                raise RuntimeError("No data source available. Load satellite map or dataset first.")

        if source == "satellite":
            return self._extract_from_satellite(x, y, size)
        elif source == "dataset":
            return self._extract_from_dataset(x, y, size)
        else:
            raise ValueError(f"Unknown source: {source}")

    def _extract_from_satellite(self, x: int, y: int, size: int) -> np.ndarray:
        """Extract tile from full satellite map."""
        if self.satellite_map is None:
            raise RuntimeError("No satellite map loaded")

        # Calculate extraction bounds
        half_size = size // 2
        x_start = max(0, x - half_size)
        x_end = min(self.satellite_map.shape[1], x + half_size)
        y_start = max(0, y - half_size)
        y_end = min(self.satellite_map.shape[0], y + half_size)

        # Extract tile
        tile = self.satellite_map[y_start:y_end, x_start:x_end]

        # Resize if necessary
        if tile.shape[0] != size or tile.shape[1] != size:
            tile = np.array(Image.fromarray(tile).resize((size, size)))

        return tile

    def _extract_from_dataset(self, x: int, y: int, size: int) -> np.ndarray:
        """Extract tile from pre-processed dataset by finding closest match."""
        if self.dataset_tiles is None:
            raise RuntimeError("No dataset loaded")

        # Convert to normalized coordinates
        norm_coord = self.coord_system.normalize(np.array([[x, y]]))[0]

        # Find closest coordinate in dataset
        distances = np.sqrt(np.sum((self.dataset_coords - norm_coord) ** 2, axis=1))
        closest_idx = np.argmin(distances)

        tile = self.dataset_tiles[closest_idx]

        # Resize if necessary
        if tile.shape[0] != size or tile.shape[1] != size:
            tile = np.array(Image.fromarray(tile).resize((size, size)))

        return tile

    def extract_tiles_batch(self, coordinates: np.ndarray, size: int = 224,
                          normalized: bool = True, source: str = "auto") -> List[np.ndarray]:
        """
        Extract multiple terrain tiles at once.

        Args:
            coordinates: Array of (x, y) coordinates
            size: Size of square tiles to extract
            normalized: Whether input coordinates are normalized
            source: Data source to use

        Returns:
            List of terrain tiles
        """
        if normalized:
            pixel_coords = self.coord_system.denormalize(coordinates)
        else:
            pixel_coords = coordinates

        tiles = []
        for coord in pixel_coords:
            tile = self.extract_tile(int(coord[0]), int(coord[1]), size, source)
            tiles.append(tile)

        return tiles

    def get_boneyard_endpoint(self) -> Tuple[int, int]:
        """
        Get the coordinates of the Boneyard endpoint for testing.

        Returns:
            (x, y) pixel coordinates of the Boneyard endpoint
        """
        # Based on the flight path analysis, the Boneyard is around (4167, 4167)
        return (4167, 4167)

    def get_boneyard_tile(self, size: int = 224, source: str = "auto") -> np.ndarray:
        """
        Get a terrain tile from the Boneyard area.

        Args:
            size: Size of the tile to extract
            source: Data source to use

        Returns:
            Terrain tile from Boneyard area
        """
        boneyard_x, boneyard_y = self.get_boneyard_endpoint()
        return self.extract_tile(boneyard_x, boneyard_y, size, source)

    def create_flight_path_tiles(self, start_coords: Tuple[float, float],
                               end_coords: Tuple[float, float],
                               num_points: int = 150, size: int = 224) -> List[np.ndarray]:
        """
        Create terrain tiles along a flight path.

        Args:
            start_coords: Starting coordinates (normalized)
            end_coords: Ending coordinates (normalized)
            num_points: Number of points along the path
            size: Size of tiles to extract

        Returns:
            List of terrain tiles along the flight path
        """
        from .utils import create_flight_path

        # Create flight path coordinates
        flight_coords = create_flight_path(start_coords, end_coords, num_points)

        # Extract tiles along the path
        tiles = self.extract_tiles_batch(flight_coords, size, normalized=True)

        return tiles

    def validate_extraction_area(self, x: int, y: int, size: int) -> bool:
        """
        Validate that extraction area is within bounds.

        Args:
            x, y: Center coordinates
            size: Size of the tile

        Returns:
            True if extraction is valid
        """
        half_size = size // 2

        if self.satellite_map is not None:
            map_height, map_width = self.satellite_map.shape[:2]
            return (half_size <= x < map_width - half_size and
                   half_size <= y < map_height - half_size)
        else:
            # Use default map size if no satellite map loaded
            return (half_size <= x < self.map_size[0] - half_size and
                   half_size <= y < self.map_size[1] - half_size)

    def get_dataset_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a specific sample from the loaded dataset.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Tuple of (tile, coordinates)
        """
        if self.dataset_tiles is None:
            raise RuntimeError("No dataset loaded")

        if not 0 <= index < len(self.dataset_tiles):
            raise IndexError(f"Index {index} out of range [0, {len(self.dataset_tiles)})")

        tile = self.dataset_tiles[index]
        coords = self.dataset_coords[index]

        return tile, coords

    def get_random_samples(self, num_samples: int = 10) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Get random samples from the dataset.

        Args:
            num_samples: Number of samples to retrieve

        Returns:
            Tuple of (tiles_list, coordinates_array)
        """
        if self.dataset_tiles is None:
            raise RuntimeError("No dataset loaded")

        indices = np.random.choice(len(self.dataset_tiles), num_samples, replace=False)

        tiles = [self.dataset_tiles[i] for i in indices]
        coords = self.dataset_coords[indices]

        return tiles, coords

    def get_info(self) -> dict:
        """
        Get information about loaded data sources.

        Returns:
            Information dictionary
        """
        info = {
            "map_size": self.map_size,
            "satellite_map_loaded": self.satellite_map is not None,
            "dataset_loaded": self.dataset_tiles is not None
        }

        if self.satellite_map is not None:
            info["satellite_map_shape"] = self.satellite_map.shape

        if self.dataset_tiles is not None:
            info["dataset_size"] = len(self.dataset_tiles)
            info["dataset_tile_shape"] = self.dataset_tiles[0].shape
            info["coordinate_range"] = {
                "x_min": float(self.dataset_coords[:, 0].min()),
                "x_max": float(self.dataset_coords[:, 0].max()),
                "y_min": float(self.dataset_coords[:, 1].min()),
                "y_max": float(self.dataset_coords[:, 1].max())
            }

        return info

    @staticmethod
    def load_corridor_dataset(artifacts_dir: str = "artifacts") -> 'TerrainExtractor':
        """
        Convenience method to load the corridor dataset.

        Args:
            artifacts_dir: Directory containing dataset files

        Returns:
            TerrainExtractor with corridor dataset loaded
        """
        dataset_path = Path(artifacts_dir) / "corridor_dataset.pkl"

        if not dataset_path.exists():
            raise FileNotFoundError(f"Corridor dataset not found: {dataset_path}")

        extractor = TerrainExtractor(dataset_path=str(dataset_path))
        return extractor