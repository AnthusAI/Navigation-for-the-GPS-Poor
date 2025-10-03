"""
Tests for dataset fetching utilities.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets import KITTIDatasetFetcher


class TestKITTIDatasetFetcher:
    """Test KITTI dataset fetching functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.fetcher = KITTIDatasetFetcher(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test fetcher initialization."""
        assert self.fetcher.data_dir.exists()
        assert self.fetcher.BASE_URL == "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_"
    
    def test_data_dir_creation(self):
        """Test that data directory is created."""
        new_temp_dir = tempfile.mkdtemp()
        shutil.rmtree(new_temp_dir)  # Remove it to test creation
        
        fetcher = KITTIDatasetFetcher(data_dir=new_temp_dir)
        assert Path(new_temp_dir).exists()
        
        shutil.rmtree(new_temp_dir)
    
    @patch('src.datasets.requests.get')
    def test_download_file_success(self, mock_get):
        """Test successful file download."""
        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content.return_value = [b'test_data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        test_file = Path(self.temp_dir) / "test_file.txt"
        self.fetcher._download_file("http://test.com/file.txt", test_file)
        
        assert test_file.exists()
        assert test_file.read_text() == "test_data"
    
    def test_load_poses_file_not_found(self):
        """Test load_poses when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.fetcher.load_poses("00")
    
    def test_load_calibration_file_not_found(self):
        """Test load_calibration when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.fetcher.load_calibration("00")
    
    def test_get_image_paths_dir_not_found(self):
        """Test get_image_paths when directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.fetcher.get_image_paths("00")
    
    def test_load_poses_format(self):
        """Test poses loading with mock data."""
        # Create mock poses file
        seq_dir = Path(self.temp_dir) / "kitti" / "sequence_00" / "poses"
        seq_dir.mkdir(parents=True)
        
        poses_file = seq_dir / "00.txt"
        # Create sample pose data (12 values per line for 3x4 transformation)
        poses_data = np.array([
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # Identity transformation
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],  # Translation in x
        ])
        np.savetxt(poses_file, poses_data)
        
        poses = self.fetcher.load_poses("00")
        
        assert poses.shape == (2, 4, 4)
        assert np.allclose(poses[0], np.eye(4))
        
        expected_second = np.eye(4)
        expected_second[0, 3] = 1
        assert np.allclose(poses[1], expected_second)
    
    def test_load_calibration_format(self):
        """Test calibration loading with mock data."""
        # Create mock calibration file
        seq_dir = Path(self.temp_dir) / "kitti" / "sequence_00" / "sequences" / "00"
        seq_dir.mkdir(parents=True)
        
        calib_file = seq_dir / "calib.txt"
        calib_content = """P0: 718.856 0.0 607.1928 0.0 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0
P1: 718.856 0.0 607.1928 -386.1448 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0
P2: 718.856 0.0 607.1928 0.0 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0
P3: 718.856 0.0 607.1928 -386.1448 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0"""
        
        calib_file.write_text(calib_content)
        
        calib = self.fetcher.load_calibration("00")
        
        assert 'P0' in calib
        assert 'P1' in calib
        assert calib['P0'].shape == (3, 4)
        assert calib['P1'].shape == (3, 4)
        
        # Check some values
        assert np.isclose(calib['P0'][0, 0], 718.856)
        assert np.isclose(calib['P1'][0, 3], -386.1448)
    
    def test_get_image_paths(self):
        """Test getting image paths."""
        # Create mock image directory
        img_dir = Path(self.temp_dir) / "kitti" / "sequence_00" / "sequences" / "00" / "image_0"
        img_dir.mkdir(parents=True)
        
        # Create some mock image files
        for i in range(3):
            (img_dir / f"{i:06d}.png").touch()
        
        paths = self.fetcher.get_image_paths("00")
        
        assert len(paths) == 3
        assert all(p.endswith('.png') for p in paths)
        assert paths == sorted(paths)  # Should be sorted


def test_download_kitti_sequence_convenience():
    """Test the convenience function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(KITTIDatasetFetcher, 'download_sequence') as mock_download:
            mock_download.return_value = Path(temp_dir) / "kitti" / "sequence_00"
            
            from src.datasets import download_kitti_sequence
            result = download_kitti_sequence("00", temp_dir)
            
            mock_download.assert_called_once_with("00")
            assert result == Path(temp_dir) / "kitti" / "sequence_00"
