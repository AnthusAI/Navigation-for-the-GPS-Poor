"""
Utility functions for the navigation system
"""
import numpy as np
import torch
from typing import Tuple, List, Optional
from pathlib import Path


def normalize_coordinates(coords: np.ndarray, map_size: Tuple[int, int] = (7500, 7500)) -> np.ndarray:
    """
    Normalize pixel coordinates to [0, 1] range.

    Args:
        coords: Array of (x, y) coordinates in pixels
        map_size: (width, height) of the full map in pixels

    Returns:
        Normalized coordinates in [0, 1] range
    """
    return coords / np.array(map_size)


def denormalize_coordinates(norm_coords: np.ndarray, map_size: Tuple[int, int] = (7500, 7500)) -> np.ndarray:
    """
    Convert normalized coordinates back to pixel coordinates.

    Args:
        norm_coords: Normalized coordinates in [0, 1] range
        map_size: (width, height) of the full map in pixels

    Returns:
        Pixel coordinates
    """
    return norm_coords * np.array(map_size)


def calculate_pixel_error(pred_coords: np.ndarray, true_coords: np.ndarray,
                         map_size: Tuple[int, int] = (7500, 7500)) -> np.ndarray:
    """
    Calculate pixel error between predicted and true coordinates.

    Args:
        pred_coords: Predicted coordinates (normalized or pixel)
        true_coords: True coordinates (normalized or pixel)
        map_size: Map size for conversion if needed

    Returns:
        Euclidean distances in pixels
    """
    # Convert to pixel coordinates if normalized
    if pred_coords.max() <= 1.0:
        pred_coords = denormalize_coordinates(pred_coords, map_size)
    if true_coords.max() <= 1.0:
        true_coords = denormalize_coordinates(true_coords, map_size)

    # Calculate Euclidean distance
    return np.sqrt(np.sum((pred_coords - true_coords) ** 2, axis=1))


def get_device() -> torch.device:
    """
    Get the best available device for computation.

    Returns:
        torch.device: cuda, mps, or cpu
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def validate_coordinates(coords: np.ndarray, normalized: bool = True) -> bool:
    """
    Validate coordinate values are within expected ranges.

    Args:
        coords: Coordinate array to validate
        normalized: Whether coordinates should be in [0, 1] range

    Returns:
        True if coordinates are valid
    """
    if coords.shape[1] != 2:
        return False

    if normalized:
        return np.all((coords >= 0) & (coords <= 1))
    else:
        # Assuming max map size of 10000x10000 for pixel coordinates
        return np.all((coords >= 0) & (coords <= 10000))


def create_flight_path(start_coords: Tuple[float, float],
                      end_coords: Tuple[float, float],
                      num_points: int = 150) -> np.ndarray:
    """
    Create a straight flight path between two points.

    Args:
        start_coords: Starting (x, y) coordinates (normalized)
        end_coords: Ending (x, y) coordinates (normalized)
        num_points: Number of points along the path

    Returns:
        Array of flight path coordinates
    """
    start = np.array(start_coords)
    end = np.array(end_coords)

    # Create linear interpolation
    t_values = np.linspace(0, 1, num_points)
    flight_path = np.array([start + t * (end - start) for t in t_values])

    return flight_path


def load_satellite_map(map_path: Optional[str] = None) -> np.ndarray:
    """
    Load the satellite map image.

    Args:
        map_path: Path to satellite map image

    Returns:
        Satellite map as numpy array
    """
    from PIL import Image

    if map_path is None:
        # Try to find satellite map in common locations
        possible_paths = [
            "chapters/1/images/satellite_map.png",
            "../1/images/satellite_map.png",
            "artifacts/satellite_map.png",
            "images/satellite_map.png"
        ]

        for path in possible_paths:
            if Path(path).exists():
                map_path = path
                break

    if map_path is None or not Path(map_path).exists():
        raise FileNotFoundError("Satellite map not found. Please provide valid map_path.")

    image = Image.open(map_path)
    return np.array(image)


class CoordinateSystem:
    """
    Handle coordinate system conversions between different representations.
    """

    def __init__(self, map_size: Tuple[int, int] = (7500, 7500)):
        self.map_size = map_size
        self.width, self.height = map_size

    def normalize(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to normalized [0, 1] range."""
        return normalize_coordinates(pixel_coords, self.map_size)

    def denormalize(self, norm_coords: np.ndarray) -> np.ndarray:
        """Convert normalized coordinates to pixel coordinates."""
        return denormalize_coordinates(norm_coords, self.map_size)

    def calculate_error(self, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
        """Calculate pixel error between predictions and ground truth."""
        return calculate_pixel_error(pred, true, self.map_size)

    def validate(self, coords: np.ndarray, normalized: bool = True) -> bool:
        """Validate coordinate values."""
        return validate_coordinates(coords, normalized)


def setup_model_device(model: torch.nn.Module) -> Tuple[torch.nn.Module, torch.device]:
    """
    Setup model on the best available device.

    Args:
        model: PyTorch model to setup

    Returns:
        Tuple of (model, device)
    """
    device = get_device()
    model = model.to(device)
    model.eval()  # Set to evaluation mode by default
    return model, device


def batch_predict(model: torch.nn.Module, tiles: np.ndarray,
                 device: torch.device, batch_size: int = 32) -> np.ndarray:
    """
    Perform batch prediction on multiple tiles.

    Args:
        model: Trained PyTorch model
        tiles: Array of input tiles
        device: Device to run predictions on
        batch_size: Batch size for processing

    Returns:
        Array of predictions
    """
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, TensorDataset

    # Setup data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transform tiles
    transformed_tiles = []
    for tile in tiles:
        if tile.shape[0] != 224:  # Need to resize
            tile_pil = transforms.ToPILImage()(tile.astype(np.uint8))
            tile_resized = transforms.Resize((224, 224))(tile_pil)
            tile = np.array(tile_resized)

        tile_tensor = transform(tile)
        transformed_tiles.append(tile_tensor)

    # Create DataLoader
    tile_tensor = torch.stack(transformed_tiles)
    dataset = TensorDataset(tile_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Perform predictions
    predictions = []
    model.eval()

    with torch.no_grad():
        for batch_tiles, in dataloader:
            batch_tiles = batch_tiles.to(device)
            batch_preds = model(batch_tiles)
            predictions.extend(batch_preds.cpu().numpy())

    return np.array(predictions)