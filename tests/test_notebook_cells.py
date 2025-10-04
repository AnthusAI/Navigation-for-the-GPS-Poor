"""
Test every notebook cell operation to catch failures BEFORE the user tries to run it.
This is what should have been done from the start.
"""

import sys
sys.path.insert(0, '..')

import pytest
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import tempfile
import shutil

from src.navigation.deep_learning import (
    PoseNet, ImprovedPoseNet, SmallPoseNet, MediumPoseNet, LargePoseNet,
    ResNetPoseNet, CoordConvPoseNet, AttentionPoseNet,
    FlightDataset, generate_flight_dataset,
    train_model, evaluate_model,
    ArtifactCache, create_or_load_dataset, train_or_load_model,
    get_device, count_parameters
)


@pytest.fixture
def sample_image():
    """Create a sample aerial image."""
    return np.random.randint(0, 255, (1024, 1536, 3), dtype=np.uint8)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    cache_dir = tempfile.mkdtemp()
    yield cache_dir
    shutil.rmtree(cache_dir)


class TestNotebookCell1_Setup:
    """Test Cell 1: Imports and setup"""
    
    def test_imports(self):
        """All imports should work."""
        # This test passing means imports work
        assert PoseNet is not None
        assert FlightDataset is not None
        
    def test_device_detection(self):
        """Device detection should work."""
        device = get_device()
        assert device.type in ['cuda', 'mps', 'cpu']
        
    def test_cache_creation(self, temp_cache_dir):
        """ArtifactCache should initialize."""
        cache = ArtifactCache(temp_cache_dir)
        assert cache.cache_dir.exists()


class TestNotebookCell3_LoadImage:
    """Test Cell 3: Load aerial image"""
    
    def test_image_loading(self, sample_image):
        """Image should be loaded correctly."""
        assert sample_image.shape == (1024, 1536, 3)
        assert sample_image.dtype == np.uint8


class TestNotebookCell5_DatasetGeneration:
    """Test Cell 5: Generate flight dataset"""
    
    def test_dataset_generation_basic(self, sample_image):
        """Basic dataset generation should work."""
        frames, poses = generate_flight_dataset(sample_image, num_samples=10, seed=42)
        assert len(frames) == 10
        assert poses.shape == (10, 2)
        
    def test_dataset_generation_cached(self, sample_image, temp_cache_dir):
        """Cached dataset generation should work."""
        cache = ArtifactCache(temp_cache_dir)
        frames, poses = create_or_load_dataset(
            cache, 'test_100', sample_image,
            num_samples=100, seed=42
        )
        assert len(frames) == 100
        assert poses.shape == (100, 2)
        
        # Second call should load from cache
        frames2, poses2 = create_or_load_dataset(
            cache, 'test_100', sample_image,
            num_samples=100, seed=42
        )
        assert len(frames2) == 100


class TestNotebookCell8_DataLoaders:
    """Test Cell 8: Create datasets and dataloaders - THIS IS WHERE IT FAILED"""
    
    def test_flight_dataset_no_transform(self, sample_image):
        """FlightDataset should work without transforms."""
        frames, poses = generate_flight_dataset(sample_image, num_samples=10, seed=42)
        dataset = FlightDataset(frames, poses, transform=None)
        
        frame, pose = dataset[0]
        assert isinstance(frame, torch.Tensor)
        assert isinstance(pose, torch.Tensor)
        assert frame.shape[0] == 3  # Channels first
        
    def test_flight_dataset_with_basic_transform(self, sample_image):
        """FlightDataset should work with basic transforms."""
        frames, poses = generate_flight_dataset(sample_image, num_samples=10, seed=42)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FlightDataset(frames, poses, transform=transform)
        frame, pose = dataset[0]
        
        assert isinstance(frame, torch.Tensor)
        assert isinstance(pose, torch.Tensor)
        assert frame.shape == (3, 224, 224)
        
    def test_flight_dataset_with_augmentation(self, sample_image):
        """FlightDataset should work with data augmentation - EXACT NOTEBOOK CONFIG."""
        frames, poses = generate_flight_dataset(sample_image, num_samples=10, seed=42)
        
        # This is the EXACT transform from the notebook that was failing
        transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FlightDataset(frames, poses, transform=transform)
        frame, pose = dataset[0]
        
        assert isinstance(frame, torch.Tensor)
        assert isinstance(pose, torch.Tensor)
        assert frame.shape == (3, 224, 224)
        
    def test_dataloader_iteration(self, sample_image):
        """DataLoader iteration should work - EXACT FAILURE POINT."""
        frames, poses = generate_flight_dataset(sample_image, num_samples=20, seed=42)
        
        # Use augmentation like the notebook
        transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FlightDataset(frames, poses, transform=transform)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # This is where it failed - iterating over the dataloader
        for batch_frames, batch_poses in loader:
            assert batch_frames.shape == (4, 3, 224, 224)
            assert batch_poses.shape == (4, 2)
            break  # Just test first batch


class TestNotebookCell10_ModelCreation:
    """Test Cell 10: Create model"""
    
    def test_posenet_creation(self):
        """PoseNet should be created correctly."""
        model = PoseNet()
        assert isinstance(model, nn.Module)
        
    def test_parameter_counting(self):
        """count_parameters should work."""
        model = PoseNet()
        params = count_parameters(model)
        assert params > 0
        assert params == 14_518_658  # Known value


class TestNotebookCell12_Training:
    """Test Cell 12+: Training loops"""
    
    def test_basic_training_loop(self, sample_image):
        """Basic training should work."""
        device = get_device()
        
        # Small dataset for speed
        frames, poses = generate_flight_dataset(sample_image, num_samples=20, seed=42)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FlightDataset(frames, poses, transform=transform)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        model = SmallPoseNet().to(device)
        
        # Train for 1 epoch
        history = train_model(
            model, loader, loader, device,
            num_epochs=1, learning_rate=0.001, verbose=False
        )
        
        assert 'train_losses' in history
        assert 'val_losses' in history
        assert len(history['train_losses']) == 1
        
    def test_cached_training(self, sample_image, temp_cache_dir):
        """Cached training should work."""
        cache = ArtifactCache(temp_cache_dir)
        device = get_device()
        
        frames, poses = create_or_load_dataset(
            cache, 'train_test', sample_image,
            num_samples=20, seed=42
        )
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FlightDataset(frames, poses, transform=transform)
        loader = DataLoader(dataset, batch_size=4)
        
        model = SmallPoseNet().to(device)
        
        model, history = train_or_load_model(
            cache, 'test_model', model,
            loader, loader, device,
            num_epochs=1, verbose=False
        )
        
        assert history is not None


class TestNotebookAllModels:
    """Test that ALL models from the notebook work."""
    
    @pytest.mark.parametrize("model_class", [
        PoseNet,
        ImprovedPoseNet,
        SmallPoseNet,
        MediumPoseNet,
        LargePoseNet,
        ResNetPoseNet,
        CoordConvPoseNet,
        AttentionPoseNet,
    ])
    def test_model_forward_pass(self, model_class):
        """Every model should handle forward pass."""
        device = get_device()
        model = model_class().to(device)
        
        # Create dummy input
        x = torch.randn(2, 3, 224, 224).to(device)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (2, 2)  # (batch_size, 2) for (x, y)
        
    @pytest.mark.parametrize("model_class", [
        PoseNet,
        ImprovedPoseNet,
        SmallPoseNet,
        MediumPoseNet,
        LargePoseNet,
        ResNetPoseNet,
        CoordConvPoseNet,
        AttentionPoseNet,
    ])
    def test_model_with_dataloader(self, model_class, sample_image):
        """Every model should work with DataLoader."""
        device = get_device()
        model = model_class().to(device)
        
        # Create small dataset
        frames, poses = generate_flight_dataset(sample_image, num_samples=8, seed=42)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FlightDataset(frames, poses, transform=transform)
        loader = DataLoader(dataset, batch_size=4)
        
        # Try one training step
        model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for batch_frames, batch_poses in loader:
            batch_frames = batch_frames.to(device)
            batch_poses = batch_poses.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_frames)
            loss = criterion(outputs, batch_poses)
            loss.backward()
            optimizer.step()
            
            break  # Just test one iteration


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

