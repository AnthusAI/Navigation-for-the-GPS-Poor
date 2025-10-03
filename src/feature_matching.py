"""
Feature detection and matching utilities for visual odometry.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class FeatureMatcher:
    """Feature detection and matching for visual odometry."""
    
    def __init__(self, detector_type: str = 'ORB', max_features: int = 1000):
        """
        Initialize feature matcher.
        
        Args:
            detector_type: Type of feature detector ('ORB', 'SIFT', 'SURF')
            max_features: Maximum number of features to detect
        """
        self.detector_type = detector_type
        self.max_features = max_features
        
        if self.detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif self.detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=self.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            raise ValueError("Unsupported detector type. Use 'ORB' or 'SIFT'.")
    
    def _create_detector(self):
        """Create feature detector based on type."""
        if self.detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=self.max_features)
        elif self.detector_type == 'SIFT':
            return cv2.SIFT_create(nfeatures=self.max_features)
        elif self.detector_type == 'SURF':
            return cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")
    
    def _create_matcher(self):
        """Create feature matcher based on detector type."""
        if self.detector_type == 'ORB':
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def detect_and_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and match features between two images.
        
        Args:
            img1: First image (grayscale)
            img2: Second image (grayscale)
            
        Returns:
            Tuple of (matched_points1, matched_points2)
        """
        # Detect keypoints and descriptors
        kp1, des1 = self.get_keypoints_and_descriptors(img1)
        kp2, des2 = self.get_keypoints_and_descriptors(img2)
        
        if des1 is None or des2 is None:
            return np.array([]), np.array([])

        des1 = np.uint8(des1)
        des2 = np.uint8(des2)

        matches = self.matcher.match(des2, des1)
        matches = sorted(matches, key=lambda x: x.distance)[:100] # Get top 100 matches
        
        pts1 = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        return pts1, pts2
        
    def match_descriptors(self, query_des, train_des):
        """Match two sets of descriptors."""
        if query_des is None or train_des is None:
            return []
            
        query_des = np.uint8(query_des)
        train_des = np.uint8(train_des)
        
        matches = self.matcher.knnMatch(query_des, train_des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return good_matches

    def get_keypoints_and_descriptors(self, img: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Get keypoints and descriptors for an image.
        
        Args:
            img: Input image (grayscale)
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        return self.detector.detectAndCompute(img, None)
    
    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray, 
                         pts1: np.ndarray, pts2: np.ndarray, 
                         max_matches: int = 50) -> np.ndarray:
        """
        Visualize feature matches between two images (vertical layout).
        
        Args:
            img1: First image
            img2: Second image
            pts1: Points in first image
            pts2: Points in second image
            max_matches: Maximum number of matches to display
            
        Returns:
            Combined image with matches drawn
        """
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        
        # Create top-to-bottom image
        combined = np.zeros((h1 + h2, max(w1, w2)), dtype=np.uint8)
        combined[:h1, :w1] = img1
        combined[h1:h1+h2, :w2] = img2
        offset = np.array([0, h1])
        
        # Convert to color for drawing
        combined_color = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
        
        # Draw matches (limit to max_matches for clarity)
        n_matches = min(len(pts1), max_matches)
        for i in range(n_matches):
            pt1 = tuple(map(int, pts1[i][0]))
            pt2 = tuple(map(int, pts2[i][0] + offset))
            
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(combined_color, pt1, 3, color, -1)
            cv2.circle(combined_color, pt2, 3, color, -1)
            cv2.line(combined_color, pt1, pt2, color, 1)
        
        return combined_color
