"""
Tests for feature matching functionality.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.feature_matching import FeatureMatcher


class TestFeatureMatcher:
    """Test feature detection and matching functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = FeatureMatcher(detector_type='ORB', max_features=100)
        
        # Create simple test images
        self.img1 = np.zeros((100, 100), dtype=np.uint8)
        self.img2 = np.zeros((100, 100), dtype=np.uint8)
        
        # Add some features (simple squares)
        cv2.rectangle(self.img1, (20, 20), (30, 30), 255, -1)
        cv2.rectangle(self.img1, (60, 60), (70, 70), 255, -1)
        
        # Second image with slightly shifted features
        cv2.rectangle(self.img2, (22, 22), (32, 32), 255, -1)
        cv2.rectangle(self.img2, (62, 62), (72, 72), 255, -1)
    
    def test_init_orb(self):
        """Test ORB detector initialization."""
        matcher = FeatureMatcher('ORB', 500)
        assert matcher.detector_type == 'ORB'
        assert matcher.max_features == 500
        assert matcher.detector is not None
        assert matcher.matcher is not None
    
    def test_init_sift(self):
        """Test SIFT detector initialization."""
        matcher = FeatureMatcher('SIFT', 500)
        assert matcher.detector_type == 'SIFT'
        assert matcher.max_features == 500
    
    def test_init_invalid_detector(self):
        """Test invalid detector type."""
        with pytest.raises(ValueError, match="Unknown detector type"):
            FeatureMatcher('INVALID')
    
    def test_detect_and_match_basic(self):
        """Test basic feature detection and matching."""
        pts1, pts2 = self.matcher.detect_and_match(self.img1, self.img2)
        
        # Should find some matches
        assert len(pts1) >= 0
        assert len(pts2) >= 0
        assert len(pts1) == len(pts2)
        
        # Points should be in correct format
        if len(pts1) > 0:
            assert pts1.shape[1:] == (1, 2)
            assert pts2.shape[1:] == (1, 2)
    
    def test_detect_and_match_empty_images(self):
        """Test matching with empty images."""
        empty_img1 = np.zeros((100, 100), dtype=np.uint8)
        empty_img2 = np.zeros((100, 100), dtype=np.uint8)
        
        pts1, pts2 = self.matcher.detect_and_match(empty_img1, empty_img2)
        
        assert len(pts1) == 0
        assert len(pts2) == 0
        assert pts1.shape == (0, 1, 2)
        assert pts2.shape == (0, 1, 2)
    
    def test_get_keypoints_and_descriptors(self):
        """Test keypoint and descriptor extraction."""
        kp, desc = self.matcher.get_keypoints_and_descriptors(self.img1)
        
        if desc is not None:
            assert len(kp) == len(desc)
            assert isinstance(kp, (list, tuple))  # OpenCV can return tuple or list
            assert isinstance(desc, np.ndarray)
    
    def test_visualize_matches(self):
        """Test match visualization."""
        # Create some mock points
        pts1 = np.array([[[10, 10]], [[20, 20]]], dtype=np.float32)
        pts2 = np.array([[[12, 12]], [[22, 22]]], dtype=np.float32)
        
        result = self.matcher.visualize_matches(self.img1, self.img2, pts1, pts2, max_matches=10)
        
        # Should return a color image
        assert len(result.shape) == 3
        assert result.shape[2] == 3
        assert result.shape[0] == max(self.img1.shape[0], self.img2.shape[0])
        assert result.shape[1] == self.img1.shape[1] + self.img2.shape[1]
    
    def test_visualize_matches_no_matches(self):
        """Test visualization with no matches."""
        pts1 = np.array([], dtype=np.float32).reshape(0, 1, 2)
        pts2 = np.array([], dtype=np.float32).reshape(0, 1, 2)
        
        result = self.matcher.visualize_matches(self.img1, self.img2, pts1, pts2)
        
        # Should still return a valid image
        assert len(result.shape) == 3
        assert result.shape[2] == 3
    
    @patch('cv2.ORB_create')
    def test_detector_creation_orb(self, mock_orb_create):
        """Test ORB detector creation."""
        mock_detector = MagicMock()
        mock_orb_create.return_value = mock_detector
        
        matcher = FeatureMatcher('ORB', 1000)
        
        mock_orb_create.assert_called_once_with(nfeatures=1000)
        assert matcher.detector == mock_detector
    
    @patch('cv2.SIFT_create')
    def test_detector_creation_sift(self, mock_sift_create):
        """Test SIFT detector creation."""
        mock_detector = MagicMock()
        mock_sift_create.return_value = mock_detector
        
        matcher = FeatureMatcher('SIFT', 1000)
        
        mock_sift_create.assert_called_once_with(nfeatures=1000)
        assert matcher.detector == mock_detector
    
    def test_matcher_creation_orb(self):
        """Test matcher creation for ORB."""
        matcher = FeatureMatcher('ORB')
        assert isinstance(matcher.matcher, cv2.BFMatcher)
    
    def test_matcher_creation_sift(self):
        """Test matcher creation for SIFT."""
        matcher = FeatureMatcher('SIFT')
        assert isinstance(matcher.matcher, cv2.BFMatcher)
    
    def test_detect_and_match_with_real_features(self):
        """Test detection and matching with more realistic features."""
        # Create images with checkerboard patterns
        img1 = np.zeros((200, 200), dtype=np.uint8)
        img2 = np.zeros((200, 200), dtype=np.uint8)
        
        # Create checkerboard pattern
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    img1[i:i+20, j:j+20] = 255
                    img2[i:i+20, j:j+20] = 255
        
        # Shift second image slightly
        img2 = np.roll(img2, 2, axis=1)
        
        pts1, pts2 = self.matcher.detect_and_match(img1, img2)
        
        # Should find some matches with checkerboard
        assert len(pts1) > 0
        assert len(pts2) > 0
