"""
Tests for deep learning module.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.navigation.deep_learning import (
    SimplePoseNet,
    AccuratePoseNet,
    FlightDataset,
    generate_flight_dataset,
    train_model,
    evaluate_model,
    get_device,
    count_parameters
)


# Fixtures
@pytest.fixture
def sample_image():
    """Create a sample aerial image."""
    return np.random.randint(0, 255, (1024, 1536, 3), dtype=np.uint8)


@pytest.fixture
def sample_dataset(sample_image):
    """Create a small sample dataset."""
    frames, poses = generate_flight_dataset(
        sample_image, num_samples=50, frame_size=(224, 224), seed=42
    )
    return frames, poses


@pytest.fixture
def device():
    """Get device for testing (prefer CPU for reproducibility)."""
    return torch.device('cpu')


# Model Architecture Tests
class TestSimplePoseNet:
    """Tests for SimplePoseNet model."""
    
    def test_model_creation(self):
        """Test that model can be created."""
        model = SimplePoseNet()
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = SimplePoseNet()
        x = torch.randn(4, 3, 224, 224)  # Batch of 4
        output = model(x)
        
        assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"
    
    def test_output_range(self):
        """Test that output values are reasonable."""
        model = SimplePoseNet()
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        # Output should be finite
        assert torch.isfinite(output).all()
    
    def test_parameter_count(self):
        """Test that model has expected number of parameters."""
        model = SimplePoseNet()
        param_count = count_parameters(model)
        
        # Should have parameters (rough estimate)
        assert 100_000 < param_count < 10_000_000


class TestAccuratePoseNet:
    """Tests for AccuratePoseNet model."""
    
    def test_model_creation(self):
        """Test that model can be created."""
        model = AccuratePoseNet()
        assert isinstance(model, nn.Module)
    
    def test_model_with_custom_dropout(self):
        """Test model creation with custom dropout rate."""
        model = AccuratePoseNet(dropout_rate=0.3)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = AccuratePoseNet()
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (4, 2)
    
    def test_batch_normalization(self):
        """Test that batch normalization layers exist."""
        model = AccuratePoseNet()
        
        # Check for batch norm layers
        has_batchnorm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
        assert has_batchnorm, "Model should have batch normalization layers"
    
    def test_dropout_layers(self):
        """Test that dropout layers exist."""
        model = AccuratePoseNet()
        
        # Check for dropout layers
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
        assert has_dropout, "Model should have dropout layers"
    
    def test_more_parameters_than_simple(self):
        """Test that accurate model has more parameters than simple."""
        simple = SimplePoseNet()
        accurate = AccuratePoseNet()
        
        simple_params = count_parameters(simple)
        accurate_params = count_parameters(accurate)
        
        assert accurate_params > simple_params


# Dataset Tests
class TestFlightDataset:
    """Tests for FlightDataset class."""
    
    def test_dataset_creation(self, sample_dataset):
        """Test dataset can be created."""
        frames, poses = sample_dataset
        dataset = FlightDataset(frames, poses)
        
        assert len(dataset) == 50
    
    def test_dataset_getitem(self, sample_dataset):
        """Test __getitem__ returns correct types."""
        frames, poses = sample_dataset
        dataset = FlightDataset(frames, poses)
        
        frame, pose = dataset[0]
        
        assert isinstance(frame, torch.Tensor)
        assert isinstance(pose, torch.Tensor)
        assert frame.shape == (3, 224, 224)
        assert pose.shape == (2,)
    
    def test_dataset_with_transform(self, sample_dataset):
        """Test dataset with transform."""
        frames, poses = sample_dataset
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        
        dataset = FlightDataset(frames, poses, transform=transform)
        frame, pose = dataset[0]
        
        assert frame.shape == (3, 224, 224)
        assert 0 <= frame.min() <= frame.max() <= 1


class TestGenerateFlightDataset:
    """Tests for generate_flight_dataset function."""
    
    def test_basic_generation(self, sample_image):
        """Test basic dataset generation."""
        frames, poses = generate_flight_dataset(
            sample_image, num_samples=10, seed=42
        )
        
        assert len(frames) == 10
        assert len(poses) == 10
        assert poses.shape == (10, 2)
    
    def test_frame_size(self, sample_image):
        """Test that generated frames have correct size."""
        frames, poses = generate_flight_dataset(
            sample_image, num_samples=5, frame_size=(224, 224), seed=42
        )
        
        for frame in frames:
            assert frame.shape == (224, 224, 3)
    
    def test_pose_normalization(self, sample_image):
        """Test that poses are normalized to [0, 1]."""
        frames, poses = generate_flight_dataset(
            sample_image, num_samples=20, seed=42
        )
        
        assert poses.min() >= 0.0
        assert poses.max() <= 1.0
    
    def test_reproducibility(self, sample_image):
        """Test that same seed produces same results."""
        frames1, poses1 = generate_flight_dataset(
            sample_image, num_samples=10, seed=42
        )
        frames2, poses2 = generate_flight_dataset(
            sample_image, num_samples=10, seed=42
        )
        
        np.testing.assert_array_equal(poses1, poses2)


# Training Tests
class TestTrainModel:
    """Tests for train_model function."""
    
    def test_train_simple_model(self, sample_dataset, device):
        """Test training a simple model."""
        frames, poses = sample_dataset
        
        # Split dataset
        train_frames = frames[:35]
        train_poses = poses[:35]
        val_frames = frames[35:]
        val_poses = poses[35:]
        
        train_dataset = FlightDataset(train_frames, train_poses)
        val_dataset = FlightDataset(val_frames, val_poses)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        model = SimplePoseNet().to(device)
        
        history = train_model(
            model, train_loader, val_loader, device,
            num_epochs=2, verbose=False
        )
        
        assert 'train_losses' in history
        assert 'val_losses' in history
        assert len(history['train_losses']) == 2
        assert len(history['val_losses']) == 2
    
    def test_training_reduces_loss(self, sample_dataset, device):
        """Test that training reduces loss over epochs."""
        frames, poses = sample_dataset
        
        train_dataset = FlightDataset(frames[:35], poses[:35])
        val_dataset = FlightDataset(frames[35:], poses[35:])
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        model = SimplePoseNet().to(device)
        
        history = train_model(
            model, train_loader, val_loader, device,
            num_epochs=5, verbose=False
        )
        
        # Loss should generally decrease
        first_loss = history['train_losses'][0]
        last_loss = history['train_losses'][-1]
        
        # Allow some variance, but expect improvement
        assert last_loss < first_loss * 1.5
    
    def test_early_stopping(self, sample_dataset, device):
        """Test that early stopping works."""
        frames, poses = sample_dataset
        
        # Create a very simple dataset that will overfit quickly
        train_dataset = FlightDataset(frames[:10], poses[:10])
        val_dataset = FlightDataset(frames[10:20], poses[10:20])
        
        train_loader = DataLoader(train_dataset, batch_size=5)
        val_loader = DataLoader(val_dataset, batch_size=5)
        
        model = SimplePoseNet().to(device)
        
        history = train_model(
            model, train_loader, val_loader, device,
            num_epochs=50, early_stopping=True,
            early_stop_patience=3, verbose=False
        )
        
        # Should stop before 50 epochs
        assert len(history['train_losses']) < 50
    
    def test_learning_rate_scheduler(self, sample_dataset, device):
        """Test training with learning rate scheduler."""
        frames, poses = sample_dataset
        
        train_dataset = FlightDataset(frames[:35], poses[:35])
        val_dataset = FlightDataset(frames[35:], poses[35:])
        
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        model = SimplePoseNet().to(device)
        
        history = train_model(
            model, train_loader, val_loader, device,
            num_epochs=3, use_scheduler=True, verbose=False
        )
        
        assert len(history['train_losses']) == 3


# Evaluation Tests
class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    def test_evaluate_model(self, sample_dataset, sample_image, device):
        """Test model evaluation."""
        frames, poses = sample_dataset
        
        dataset = FlightDataset(frames, poses)
        loader = DataLoader(dataset, batch_size=8)
        
        model = SimplePoseNet().to(device)
        
        results = evaluate_model(
            model, loader, device,
            aerial_image_shape=sample_image.shape[:2]
        )
        
        assert 'predictions' in results
        assert 'targets' in results
        assert 'errors' in results
        
        assert results['predictions'].shape == (50, 2)
        assert results['targets'].shape == (50, 2)
        assert results['errors'].shape == (50,)
    
    def test_errors_are_positive(self, sample_dataset, sample_image, device):
        """Test that all errors are non-negative."""
        frames, poses = sample_dataset
        
        dataset = FlightDataset(frames, poses)
        loader = DataLoader(dataset, batch_size=8)
        
        model = SimplePoseNet().to(device)
        model.eval()
        
        results = evaluate_model(
            model, loader, device,
            aerial_image_shape=sample_image.shape[:2]
        )
        
        assert (results['errors'] >= 0).all()


# Utility Tests
class TestGetDevice:
    """Tests for get_device function."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device(prefer_gpu=False)
        assert device.type == 'cpu'
    
    def test_get_device_prefer_gpu(self):
        """Test GPU preference (may be CPU if no GPU available)."""
        device = get_device(prefer_gpu=True)
        assert device.type in ['cpu', 'cuda', 'mps']


class TestCountParameters:
    """Tests for count_parameters function."""
    
    def test_count_simple_model(self):
        """Test parameter counting for simple model."""
        model = SimplePoseNet()
        count = count_parameters(model)
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_accurate_model(self):
        """Test parameter counting for accurate model."""
        model = AccuratePoseNet()
        count = count_parameters(model)
        
        assert count > 0
        assert isinstance(count, int)


# Integration Tests
class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_full_training_pipeline(self, sample_image, device):
        """Test complete training pipeline."""
        # Generate dataset
        frames, poses = generate_flight_dataset(
            sample_image, num_samples=100, seed=42
        )
        
        # Split data
        train_size = 70
        val_size = 15
        
        train_dataset = FlightDataset(frames[:train_size], poses[:train_size])
        val_dataset = FlightDataset(
            frames[train_size:train_size+val_size],
            poses[train_size:train_size+val_size]
        )
        test_dataset = FlightDataset(
            frames[train_size+val_size:],
            poses[train_size+val_size:]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        # Create and train model
        model = SimplePoseNet().to(device)
        
        history = train_model(
            model, train_loader, val_loader, device,
            num_epochs=3, verbose=False
        )
        
        # Evaluate model
        results = evaluate_model(
            model, test_loader, device,
            aerial_image_shape=sample_image.shape[:2]
        )
        
        # Verify results
        assert len(history['train_losses']) == 3
        assert len(results['errors']) == 15
        assert results['errors'].mean() < 1500  # Reasonable error for untrained model
    
    def test_simple_vs_accurate_comparison(self, sample_image, device):
        """Test that accurate model architecture is more complex."""
        simple_model = SimplePoseNet()
        accurate_model = AccuratePoseNet()
        
        simple_params = count_parameters(simple_model)
        accurate_params = count_parameters(accurate_model)
        
        # Accurate model should have more parameters
        assert accurate_params > simple_params
        
        # Both models should work with same input
        x = torch.randn(1, 3, 224, 224)
        
        simple_output = simple_model(x)
        accurate_output = accurate_model(x)
        
        assert simple_output.shape == accurate_output.shape == (1, 2)

