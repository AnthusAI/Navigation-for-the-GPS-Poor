"""
Deep learning training and utilities for visual navigation.

This module provides training loops, dataset utilities, and artifact caching.
Model architectures are in models.py to keep things organized.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import os

# Import all models from models.py
from .models import (
    SimpleCNN,
    PoseNet, ImprovedPoseNet, SmallPoseNet, MediumPoseNet, LargePoseNet,
    ResNetPoseNet, CoordConvPoseNet, AttentionPoseNet,
    SimplePoseNet, AccuratePoseNet  # Aliases
)

# Re-export models for convenience
__all__ = [
    # Models
    'SimpleCNN',
    'PoseNet', 'ImprovedPoseNet', 'SmallPoseNet', 'MediumPoseNet', 'LargePoseNet',
    'ResNetPoseNet', 'CoordConvPoseNet', 'AttentionPoseNet',
    'SimplePoseNet', 'AccuratePoseNet',
    # Dataset
    'TerrainDataset',
    'FlightDataset', 'generate_flight_dataset',
    # Training
    'train_model', 'evaluate_model',
    # Caching
    'ArtifactCache', 'create_or_load_dataset', 'train_or_load_model',
    # Utils
    'get_device', 'count_parameters'
]


class TerrainDataset(Dataset):
    """
    A PyTorch Dataset to sample random tiles from a large satellite image.
    Each item is a tuple of (image_tile, coordinates).

    This version is memory-efficient as it loads the large image once and
    crops tiles on-the-fly in __getitem__.
    """
    def __init__(self, image_path: str, num_samples: int, tile_size: int, 
                 seed: int = 42, transform: Optional[nn.Module] = None):
        """
        Args:
            image_path (str): Path to the large satellite image.
            num_samples (int): The total number of random samples to generate.
            tile_size (int): The width and height of the square tiles.
            seed (int): Random seed for reproducibility.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Satellite image not found at {image_path}")
        
        self.image = Image.open(image_path).convert('RGB')
        self.image_width, self.image_height = self.image.size
        self.num_samples = num_samples
        self.tile_size = tile_size
        
        # Pre-generate random coordinates to sample from
        self.coordinates = self._generate_coordinates(seed)

        # Use provided transform or a default one
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _generate_coordinates(self, seed: int) -> List[Tuple[int, int]]:
        """Generates a list of random top-left coordinates for sampling."""
        rng = np.random.RandomState(seed)
        max_x = self.image_width - self.tile_size
        max_y = self.image_height - self.tile_size
        x_coords = rng.randint(0, max_x, self.num_samples)
        y_coords = rng.randint(0, max_y, self.num_samples)
        return list(zip(x_coords, y_coords))

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the sample at the given index.
        Returns:
            tuple: (tile_image, normalized_coordinates) where tile_image is a
                   Tensor and normalized_coordinates is a Tensor of shape (2,).
        """
        x1, y1 = self.coordinates[idx]
        x2, y2 = x1 + self.tile_size, y1 + self.tile_size

        tile_image = self.image.crop((x1, y1, x2, y2))
        
        center_x = x1 + self.tile_size / 2
        center_y = y1 + self.tile_size / 2
        
        norm_x = center_x / self.image_width
        norm_y = center_y / self.image_height
        
        coordinates = torch.tensor([norm_x, norm_y], dtype=torch.float32)

        if self.transform:
            tile_image = self.transform(tile_image)
            
        return tile_image, coordinates


class FlightDataset(Dataset):
    """
    PyTorch Dataset for flight frames and poses.
    
    Args:
        frames: List of image frames (numpy arrays)
        poses: Array of normalized (x, y) positions
        transform: Optional torchvision transform
    """
    
    def __init__(self, frames: List[np.ndarray], poses: np.ndarray, 
                 transform=None):
        self.frames = frames
        self.poses = poses
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame = self.frames[idx]
        pose = self.poses[idx]
        
        if self.transform:
            # Apply transform - it should handle numpy->PIL conversion if needed
            # (via ToPILImage() in the transform itself)
            frame = self.transform(frame)
        else:
            # No transform: convert numpy to tensor manually
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        return frame, torch.from_numpy(pose).float()


def generate_flight_dataset(image: np.ndarray, num_samples: int = 1000,
                           frame_size: Tuple[int, int] = (224, 224),
                           seed: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Generate synthetic flight dataset by extracting frames from aerial image.
    
    Args:
        image: Full aerial image (RGB)
        num_samples: Number of frames to extract
        frame_size: Size of each frame (height, width)
        seed: Random seed for reproducibility
    
    Returns:
        frames: List of image frames
        poses: Normalized (x, y) positions [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = image.shape[:2]
    frame_h, frame_w = frame_size
    
    frames = []
    poses = []
    
    for _ in range(num_samples):
        # Random position
        x = np.random.randint(0, w - frame_w)
        y = np.random.randint(0, h - frame_h)
        
        # Extract frame
        frame = image[y:y+frame_h, x:x+frame_w]
        frames.append(frame)
        
        # Normalize pose to [0, 1]
        pose = np.array([x / (w - frame_w), y / (h - frame_h)], dtype=np.float32)
        poses.append(pose)
    
    return frames, np.array(poses)


def train_model(model: nn.Module, 
               train_loader: DataLoader,
               val_loader: DataLoader,
               device: torch.device,
               num_epochs: int = 20,
               learning_rate: float = 0.001,
               use_scheduler: bool = False,
               early_stopping: bool = False,
               early_stop_patience: int = 10,
               verbose: bool = True) -> Dict[str, List[float]]:
    """
    Train a pose estimation model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (cpu/cuda/mps)
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        use_scheduler: Whether to use learning rate scheduling
        early_stopping: Whether to use early stopping
        early_stop_patience: Patience for early stopping
        verbose: Whether to print training progress
    
    Returns:
        Dictionary with 'train_losses' and 'val_losses' lists
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for frames_batch, poses_batch in progress_bar:
            frames_batch = frames_batch.to(device)
            poses_batch = poses_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames_batch)
            loss = criterion(outputs, poses_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for frames_batch, poses_batch in val_loader:
                frames_batch = frames_batch.to(device)
                poses_batch = poses_batch.to(device)
                
                outputs = model(frames_batch)
                loss = criterion(outputs, poses_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        if use_scheduler:
            scheduler.step(val_loss)
        
        # Early stopping check
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        if verbose:
            tqdm.write(f"Epoch {epoch+1:2d}/{num_epochs} - "
                      f"Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def evaluate_model(model: nn.Module,
                  test_loader: DataLoader,
                  device: torch.device,
                  aerial_image_shape: Tuple[int, int],
                  frame_size: Tuple[int, int] = (224, 224)) -> Dict[str, np.ndarray]:
    """
    Evaluate model on test set and return predictions and errors.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        aerial_image_shape: Shape of original aerial image (h, w)
        frame_size: Frame dimensions (h, w)
    
    Returns:
        Dictionary with 'predictions', 'targets', and 'errors' (in pixels)
    """
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for frames_batch, poses_batch in test_loader:
            frames_batch = frames_batch.to(device)
            outputs = model(frames_batch).cpu().numpy()
            preds.append(outputs)
            targets.append(poses_batch.numpy())
    
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    
    # Calculate pixel errors
    h, w = aerial_image_shape
    frame_w, frame_h = frame_size
    
    pred_x_px = preds[:, 0] * (w - frame_w) + frame_w // 2
    pred_y_px = preds[:, 1] * (h - frame_h) + frame_h // 2
    
    target_x_px = targets[:, 0] * (w - frame_w) + frame_w // 2
    target_y_px = targets[:, 1] * (h - frame_h) + frame_h // 2
    
    errors = np.sqrt((pred_x_px - target_x_px)**2 + (pred_y_px - target_y_px)**2)
    
    return {
        'predictions': preds,
        'targets': targets,
        'errors': errors
    }


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device for training.
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
    
    Returns:
        torch.device object
    """
    if not prefer_gpu:
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Artifact Caching Utilities
# ============================================================================

class ArtifactCache:
    """
    Manages caching of intermediate results to speed up development.
    
    Supports saving/loading:
    - Generated datasets
    - Trained models
    - Evaluation results
    - Training history
    
    Example:
        cache = ArtifactCache('chapters/4/artifacts')
        
        # Save dataset
        cache.save_dataset('flight_1000', frames, poses)
        
        # Load dataset
        frames, poses = cache.load_dataset('flight_1000')
        
        # Save trained model
        cache.save_model('simple_model', model, metadata={'epochs': 20})
        
        # Load trained model
        model = SimplePoseNet()
        cache.load_model('simple_model', model)
    """
    
    def __init__(self, cache_dir: str = 'artifacts'):
        """
        Initialize artifact cache.
        
        Args:
            cache_dir: Directory to store cached artifacts
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, name: str, suffix: str) -> Path:
        """Get full path for an artifact."""
        return self.cache_dir / f"{name}{suffix}"
    
    def exists(self, name: str, artifact_type: str = 'dataset') -> bool:
        """
        Check if artifact exists.
        
        Args:
            name: Artifact name
            artifact_type: Type of artifact ('dataset', 'model', 'results')
        
        Returns:
            True if artifact exists
        """
        if artifact_type == 'dataset':
            return self._get_path(name, '_dataset.pkl').exists()
        elif artifact_type == 'model':
            return self._get_path(name, '_model.pth').exists()
        elif artifact_type == 'results':
            return self._get_path(name, '_results.pkl').exists()
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    # Dataset caching
    def save_dataset(self, name: str, frames: List[np.ndarray], 
                    poses: np.ndarray, metadata: Optional[Dict] = None):
        """
        Save generated dataset.
        
        Args:
            name: Dataset identifier
            frames: List of image frames
            poses: Pose array
            metadata: Optional metadata (num_samples, frame_size, etc.)
        """
        data = {
            'frames': frames,
            'poses': poses,
            'metadata': metadata or {}
        }
        
        path = self._get_path(name, '_dataset.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Saved dataset to {path}")
    
    def load_dataset(self, name: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load cached dataset.
        
        Args:
            name: Dataset identifier
        
        Returns:
            frames, poses tuple
        """
        path = self._get_path(name, '_dataset.pkl')
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded dataset from {path}")
        return data['frames'], data['poses']
    
    def get_dataset_metadata(self, name: str) -> Dict:
        """Get metadata for a cached dataset."""
        path = self._get_path(name, '_dataset.pkl')
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        return data.get('metadata', {})
    
    # Model caching
    def save_model(self, name: str, model: nn.Module, 
                  metadata: Optional[Dict] = None):
        """
        Save trained model.
        
        Args:
            name: Model identifier
            model: Trained PyTorch model
            metadata: Optional metadata (epochs, loss, hyperparams, etc.)
        """
        # Save model weights
        model_path = self._get_path(name, '_model.pth')
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        if metadata:
            meta_path = self._get_path(name, '_model_meta.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved model to {model_path}")
    
    def load_model(self, name: str, model: nn.Module) -> nn.Module:
        """
        Load trained model weights.
        
        Args:
            name: Model identifier
            model: Model instance to load weights into
        
        Returns:
            Model with loaded weights
        """
        model_path = self._get_path(name, '_model.pth')
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"✓ Loaded model from {model_path}")
        
        return model
    
    def get_model_metadata(self, name: str) -> Dict:
        """Get metadata for a cached model."""
        meta_path = self._get_path(name, '_model_meta.json')
        
        if not meta_path.exists():
            return {}
        
        with open(meta_path, 'r') as f:
            return json.load(f)
    
    # Results caching
    def save_results(self, name: str, results: Dict[str, Any]):
        """
        Save evaluation results or other computed results.
        
        Args:
            name: Results identifier
            results: Dictionary of results to save
        """
        path = self._get_path(name, '_results.pkl')
        
        with open(path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"✓ Saved results to {path}")
    
    def load_results(self, name: str) -> Dict[str, Any]:
        """
        Load cached results.
        
        Args:
            name: Results identifier
        
        Returns:
            Dictionary of results
        """
        path = self._get_path(name, '_results.pkl')
        
        if not path.exists():
            raise FileNotFoundError(f"Results not found: {path}")
        
        with open(path, 'rb') as f:
            results = pickle.load(f)
        
        print(f"✓ Loaded results from {path}")
        return results
    
    # Training history caching
    def save_history(self, name: str, history: Dict[str, List[float]]):
        """
        Save training history.
        
        Args:
            name: History identifier
            history: Dictionary with 'train_losses' and 'val_losses'
        """
        path = self._get_path(name, '_history.json')
        
        # Convert to JSON-serializable format
        history_json = {
            k: [float(v) for v in vals]
            for k, vals in history.items()
        }
        
        with open(path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"✓ Saved history to {path}")
    
    def load_history(self, name: str) -> Dict[str, List[float]]:
        """
        Load training history.
        
        Args:
            name: History identifier
        
        Returns:
            Dictionary with training history
        """
        path = self._get_path(name, '_history.json')
        
        if not path.exists():
            raise FileNotFoundError(f"History not found: {path}")
        
        with open(path, 'r') as f:
            history = json.load(f)
        
        print(f"✓ Loaded history from {path}")
        return history
    
    # Utility methods
    def list_artifacts(self) -> Dict[str, List[str]]:
        """
        List all cached artifacts.
        
        Returns:
            Dictionary with lists of dataset, model, and result names
        """
        datasets = []
        models = []
        results = []
        histories = []
        
        for path in self.cache_dir.glob('*'):
            name = path.stem
            
            if path.suffix == '.pkl':
                if '_dataset' in name:
                    datasets.append(name.replace('_dataset', ''))
                elif '_results' in name:
                    results.append(name.replace('_results', ''))
            elif path.suffix == '.pth':
                models.append(name.replace('_model', ''))
            elif path.suffix == '.json' and '_history' in name:
                histories.append(name.replace('_history', ''))
        
        return {
            'datasets': sorted(set(datasets)),
            'models': sorted(set(models)),
            'results': sorted(set(results)),
            'histories': sorted(set(histories))
        }
    
    def clear(self, artifact_type: Optional[str] = None):
        """
        Clear cached artifacts.
        
        Args:
            artifact_type: Type to clear ('dataset', 'model', 'results', 'all', or None for all)
        """
        if artifact_type is None or artifact_type == 'all':
            patterns = ['*_dataset.pkl', '*_model.pth', '*_model_meta.json', 
                       '*_results.pkl', '*_history.json']
        elif artifact_type == 'dataset':
            patterns = ['*_dataset.pkl']
        elif artifact_type == 'model':
            patterns = ['*_model.pth', '*_model_meta.json']
        elif artifact_type == 'results':
            patterns = ['*_results.pkl']
        elif artifact_type == 'history':
            patterns = ['*_history.json']
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        count = 0
        for pattern in patterns:
            for path in self.cache_dir.glob(pattern):
                path.unlink()
                count += 1
        
        print(f"✓ Cleared {count} artifact(s)")


def create_or_load_dataset(cache: ArtifactCache, name: str,
                           image: np.ndarray, num_samples: int,
                           frame_size: Tuple[int, int] = (224, 224),
                           force_regenerate: bool = False,
                           seed: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Create or load cached dataset.
    
    Args:
        cache: ArtifactCache instance
        name: Dataset name
        image: Aerial image
        num_samples: Number of samples
        frame_size: Frame dimensions
        force_regenerate: Force regeneration even if cached
        seed: Random seed
    
    Returns:
        frames, poses tuple
    """
    if not force_regenerate and cache.exists(name, 'dataset'):
        print(f"Loading cached dataset '{name}'...")
        return cache.load_dataset(name)
    
    print(f"Generating dataset '{name}' ({num_samples} samples)...")
    frames, poses = generate_flight_dataset(
        image, num_samples=num_samples, frame_size=frame_size, seed=seed
    )
    
    metadata = {
        'num_samples': num_samples,
        'frame_size': frame_size,
        'seed': seed,
        'image_shape': image.shape
    }
    
    cache.save_dataset(name, frames, poses, metadata)
    return frames, poses


def train_or_load_model(cache: ArtifactCache, name: str,
                       model: nn.Module, train_loader: DataLoader,
                       val_loader: DataLoader, device: torch.device,
                       force_retrain: bool = False,
                       **train_kwargs) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train or load cached model.
    
    Args:
        cache: ArtifactCache instance
        name: Model name
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        force_retrain: Force retraining even if cached
        **train_kwargs: Additional arguments for train_model
    
    Returns:
        Tuple of (trained model, training history)
    """
    if not force_retrain and cache.exists(name, 'model'):
        print(f"Loading cached model '{name}'...")
        model = cache.load_model(name, model)
        
        # Try to load history too
        try:
            history = cache.load_history(name)
        except FileNotFoundError:
            history = {'train_losses': [], 'val_losses': []}
        
        return model, history
    
    print(f"Training model '{name}'...")
    history = train_model(model, train_loader, val_loader, device, **train_kwargs)
    
    metadata = {
        'num_epochs': len(history['train_losses']),
        'final_train_loss': history['train_losses'][-1],
        'final_val_loss': history['val_losses'][-1],
        **train_kwargs
    }
    
    cache.save_model(name, model, metadata)
    cache.save_history(name, history)
    
    return model, history

