"""
Tests for artifact caching utilities.
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path

from src.navigation.deep_learning import (
    ArtifactCache,
    SimplePoseNet,
    create_or_load_dataset,
    train_or_load_model,
    FlightDataset,
    generate_flight_dataset
)
from torch.utils.data import DataLoader, Subset


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def cache(temp_cache_dir):
    """Create an ArtifactCache instance."""
    return ArtifactCache(temp_cache_dir)


@pytest.fixture
def sample_image():
    """Create a sample aerial image."""
    return np.random.randint(0, 255, (512, 768, 3), dtype=np.uint8)


@pytest.fixture
def sample_dataset(sample_image):
    """Create a small sample dataset."""
    frames, poses = generate_flight_dataset(
        sample_image, num_samples=20, frame_size=(224, 224), seed=42
    )
    return frames, poses


class TestArtifactCache:
    """Tests for ArtifactCache class."""
    
    def test_cache_creation(self, temp_cache_dir):
        """Test that cache directory is created."""
        cache = ArtifactCache(temp_cache_dir)
        assert cache.cache_dir.exists()
    
    def test_exists_dataset(self, cache):
        """Test checking if dataset exists."""
        assert not cache.exists('test', 'dataset')
    
    def test_exists_model(self, cache):
        """Test checking if model exists."""
        assert not cache.exists('test', 'model')
    
    def test_exists_results(self, cache):
        """Test checking if results exist."""
        assert not cache.exists('test', 'results')


class TestDatasetCaching:
    """Tests for dataset caching."""
    
    def test_save_and_load_dataset(self, cache, sample_dataset):
        """Test saving and loading a dataset."""
        frames, poses = sample_dataset
        
        # Save dataset
        cache.save_dataset('test_dataset', frames, poses)
        
        # Check it exists
        assert cache.exists('test_dataset', 'dataset')
        
        # Load dataset
        loaded_frames, loaded_poses = cache.load_dataset('test_dataset')
        
        # Verify data is the same
        assert len(loaded_frames) == len(frames)
        np.testing.assert_array_equal(loaded_poses, poses)
    
    def test_save_dataset_with_metadata(self, cache, sample_dataset):
        """Test saving dataset with metadata."""
        frames, poses = sample_dataset
        
        metadata = {
            'num_samples': len(frames),
            'frame_size': (224, 224),
            'seed': 42
        }
        
        cache.save_dataset('test_dataset', frames, poses, metadata)
        
        loaded_metadata = cache.get_dataset_metadata('test_dataset')
        assert loaded_metadata['num_samples'] == 20
        assert loaded_metadata['seed'] == 42
    
    def test_load_nonexistent_dataset(self, cache):
        """Test loading a dataset that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            cache.load_dataset('nonexistent')


class TestModelCaching:
    """Tests for model caching."""
    
    def test_save_and_load_model(self, cache):
        """Test saving and loading a model."""
        # Create and save model
        model = SimplePoseNet()
        original_state = model.state_dict().copy()
        
        cache.save_model('test_model', model)
        
        # Check it exists
        assert cache.exists('test_model', 'model')
        
        # Create new model and load weights
        new_model = SimplePoseNet()
        cache.load_model('test_model', new_model)
        
        # Verify weights are the same
        for key in original_state:
            assert torch.allclose(original_state[key], new_model.state_dict()[key])
    
    def test_save_model_with_metadata(self, cache):
        """Test saving model with metadata."""
        model = SimplePoseNet()
        
        metadata = {
            'epochs': 20,
            'final_loss': 0.123,
            'learning_rate': 0.001
        }
        
        cache.save_model('test_model', model, metadata)
        
        loaded_metadata = cache.get_model_metadata('test_model')
        assert loaded_metadata['epochs'] == 20
        assert loaded_metadata['final_loss'] == 0.123
    
    def test_load_nonexistent_model(self, cache):
        """Test loading a model that doesn't exist."""
        model = SimplePoseNet()
        
        with pytest.raises(FileNotFoundError):
            cache.load_model('nonexistent', model)


class TestResultsCaching:
    """Tests for results caching."""
    
    def test_save_and_load_results(self, cache):
        """Test saving and loading results."""
        results = {
            'predictions': np.random.randn(10, 2),
            'targets': np.random.randn(10, 2),
            'errors': np.random.randn(10),
            'mean_error': 123.45
        }
        
        cache.save_results('test_results', results)
        
        # Check it exists
        assert cache.exists('test_results', 'results')
        
        # Load results
        loaded_results = cache.load_results('test_results')
        
        assert loaded_results['mean_error'] == 123.45
        np.testing.assert_array_equal(loaded_results['errors'], results['errors'])
    
    def test_load_nonexistent_results(self, cache):
        """Test loading results that don't exist."""
        with pytest.raises(FileNotFoundError):
            cache.load_results('nonexistent')


class TestHistoryCaching:
    """Tests for training history caching."""
    
    def test_save_and_load_history(self, cache):
        """Test saving and loading training history."""
        history = {
            'train_losses': [0.5, 0.4, 0.3, 0.2],
            'val_losses': [0.6, 0.5, 0.4, 0.3]
        }
        
        cache.save_history('test_history', history)
        
        # Load history
        loaded_history = cache.load_history('test_history')
        
        assert loaded_history['train_losses'] == history['train_losses']
        assert loaded_history['val_losses'] == history['val_losses']
    
    def test_load_nonexistent_history(self, cache):
        """Test loading history that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            cache.load_history('nonexistent')


class TestCacheUtilities:
    """Tests for cache utility methods."""
    
    def test_list_artifacts_empty(self, cache):
        """Test listing artifacts when cache is empty."""
        artifacts = cache.list_artifacts()
        
        assert artifacts['datasets'] == []
        assert artifacts['models'] == []
        assert artifacts['results'] == []
        assert artifacts['histories'] == []
    
    def test_list_artifacts(self, cache, sample_dataset):
        """Test listing artifacts after adding some."""
        frames, poses = sample_dataset
        model = SimplePoseNet()
        
        # Add various artifacts
        cache.save_dataset('dataset1', frames, poses)
        cache.save_dataset('dataset2', frames, poses)
        cache.save_model('model1', model)
        cache.save_results('results1', {'error': 123})
        cache.save_history('model1', {'train_losses': [0.1, 0.2]})
        
        artifacts = cache.list_artifacts()
        
        assert 'dataset1' in artifacts['datasets']
        assert 'dataset2' in artifacts['datasets']
        assert 'model1' in artifacts['models']
        assert 'results1' in artifacts['results']
        assert 'model1' in artifacts['histories']
    
    def test_clear_all(self, cache, sample_dataset):
        """Test clearing all artifacts."""
        frames, poses = sample_dataset
        model = SimplePoseNet()
        
        # Add artifacts
        cache.save_dataset('dataset1', frames, poses)
        cache.save_model('model1', model)
        cache.save_results('results1', {'error': 123})
        
        # Clear all
        cache.clear('all')
        
        artifacts = cache.list_artifacts()
        assert len(artifacts['datasets']) == 0
        assert len(artifacts['models']) == 0
        assert len(artifacts['results']) == 0
    
    def test_clear_datasets_only(self, cache, sample_dataset):
        """Test clearing only datasets."""
        frames, poses = sample_dataset
        model = SimplePoseNet()
        
        cache.save_dataset('dataset1', frames, poses)
        cache.save_model('model1', model)
        
        cache.clear('dataset')
        
        artifacts = cache.list_artifacts()
        assert len(artifacts['datasets']) == 0
        assert len(artifacts['models']) == 1


class TestCreateOrLoadDataset:
    """Tests for create_or_load_dataset helper."""
    
    def test_create_new_dataset(self, cache, sample_image):
        """Test creating a new dataset."""
        frames, poses = create_or_load_dataset(
            cache, 'new_dataset', sample_image, num_samples=10, seed=42
        )
        
        assert len(frames) == 10
        assert cache.exists('new_dataset', 'dataset')
    
    def test_load_cached_dataset(self, cache, sample_image):
        """Test loading cached dataset instead of regenerating."""
        # Create dataset first
        frames1, poses1 = create_or_load_dataset(
            cache, 'cached_dataset', sample_image, num_samples=10, seed=42
        )
        
        # Load it again (should use cache)
        frames2, poses2 = create_or_load_dataset(
            cache, 'cached_dataset', sample_image, num_samples=10, seed=42
        )
        
        # Should be the same data
        np.testing.assert_array_equal(poses1, poses2)
    
    def test_force_regenerate(self, cache, sample_image):
        """Test forcing regeneration even with cache."""
        # Create dataset
        create_or_load_dataset(
            cache, 'force_dataset', sample_image, num_samples=10, seed=42
        )
        
        # Force regenerate with different seed
        frames, poses = create_or_load_dataset(
            cache, 'force_dataset', sample_image, num_samples=10,
            seed=99, force_regenerate=True
        )
        
        assert len(frames) == 10


class TestTrainOrLoadModel:
    """Tests for train_or_load_model helper."""
    
    def test_train_new_model(self, cache, sample_dataset):
        """Test training a new model."""
        frames, poses = sample_dataset
        
        dataset = FlightDataset(frames, poses)
        train_dataset = Subset(dataset, range(15))
        val_dataset = Subset(dataset, range(15, 20))
        train_loader = DataLoader(train_dataset, batch_size=5)
        val_loader = DataLoader(val_dataset, batch_size=5)
        
        model = SimplePoseNet()
        device = torch.device('cpu')
        
        trained_model, history = train_or_load_model(
            cache, 'new_model', model, train_loader, val_loader, device,
            num_epochs=2, verbose=False
        )
        
        assert cache.exists('new_model', 'model')
        assert len(history['train_losses']) == 2
    
    def test_load_cached_model(self, cache, sample_dataset):
        """Test loading cached model instead of retraining."""
        frames, poses = sample_dataset
        
        dataset = FlightDataset(frames, poses)
        train_dataset = Subset(dataset, range(15))
        val_dataset = Subset(dataset, range(15, 20))
        train_loader = DataLoader(train_dataset, batch_size=5)
        val_loader = DataLoader(val_dataset, batch_size=5)
        
        device = torch.device('cpu')
        
        # Train first time
        model1 = SimplePoseNet()
        trained_model1, history1 = train_or_load_model(
            cache, 'cached_model', model1, train_loader, val_loader, device,
            num_epochs=2, verbose=False
        )
        
        # "Train" again (should load from cache)
        model2 = SimplePoseNet()
        trained_model2, history2 = train_or_load_model(
            cache, 'cached_model', model2, train_loader, val_loader, device,
            num_epochs=2, verbose=False
        )
        
        # Should have same weights
        for key in trained_model1.state_dict():
            assert torch.allclose(
                trained_model1.state_dict()[key],
                trained_model2.state_dict()[key]
            )
    
    def test_force_retrain(self, cache, sample_dataset):
        """Test forcing retraining even with cache."""
        frames, poses = sample_dataset
        
        dataset = FlightDataset(frames, poses)
        train_dataset = Subset(dataset, range(15))
        val_dataset = Subset(dataset, range(15, 20))
        train_loader = DataLoader(train_dataset, batch_size=5)
        val_loader = DataLoader(val_dataset, batch_size=5)
        
        device = torch.device('cpu')
        
        # Train first time
        model1 = SimplePoseNet()
        train_or_load_model(
            cache, 'retrain_model', model1, train_loader, val_loader, device,
            num_epochs=2, verbose=False
        )
        
        # Force retrain
        model2 = SimplePoseNet()
        trained_model2, history2 = train_or_load_model(
            cache, 'retrain_model', model2, train_loader, val_loader, device,
            num_epochs=2, force_retrain=True, verbose=False
        )
        
        assert len(history2['train_losses']) == 2

