"""
SLAM (Simultaneous Localization and Mapping) implementation for RGB-D cameras.

This module implements a basic visual SLAM system that builds a map of 3D landmarks
while tracking camera pose, with loop closure detection and graph optimization.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from scipy.optimize import least_squares

from .feature_matching import FeatureMatcher
from .pose_estimation import PoseEstimator
from .utils import load_image


class Landmark:
    """Represents a 3D landmark in the map."""
    
    def __init__(self, landmark_id: int, position: np.ndarray, descriptor: np.ndarray):
        """
        Initialize a landmark.
        
        Args:
            landmark_id: Unique identifier for this landmark
            position: 3D position as (x, y, z) numpy array
            descriptor: Feature descriptor for matching
        """
        self.id = landmark_id
        self.position = position.copy()
        self.descriptor = descriptor.copy()
        
        # Track observations
        self.observations = []  # List of (frame_id, keypoint_2d)
        self.times_observed = 0
        self.creation_frame = None
        
    def add_observation(self, frame_id: int, keypoint_2d: np.ndarray):
        """Record that this landmark was observed in a frame."""
        self.observations.append((frame_id, keypoint_2d.copy()))
        self.times_observed += 1
        
        if self.creation_frame is None:
            self.creation_frame = frame_id


class RGBDSlam:
    """RGB-D SLAM system with loop closure detection."""
    
    def __init__(self, camera_matrix: np.ndarray,
                 detector_type: str = 'ORB',
                 max_features: int = 1000,
                 loop_closure_threshold: float = 30.0):
        """
        Initialize SLAM system.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            detector_type: Type of feature detector ('ORB', 'SIFT', 'SURF')
            max_features: Maximum number of features to detect
            loop_closure_threshold: Distance threshold for loop closure detection
        """
        self.K = camera_matrix
        self.feature_matcher = FeatureMatcher(detector_type, max_features)
        self.pose_estimator = PoseEstimator(camera_matrix)
        
        # Pose tracking
        self.current_pose = np.eye(4)
        self.poses = [self.current_pose.copy()]
        
        # Landmark map
        self.landmarks: Dict[int, Landmark] = {}
        self.next_landmark_id = 0
        
        # Frame tracking
        self.frame_count = 0
        self.prev_image = None
        self.prev_depth = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Loop closure
        self.loop_closure_threshold = loop_closure_threshold
        self.loop_closures = []  # List of (frame_i, frame_j) tuples
        
        # Statistics
        self.total_matches = 0
        self.total_landmarks_created = 0
        self.total_landmarks_matched = 0
        
    def depth_to_3d(self, keypoint: np.ndarray, depth: float) -> Optional[np.ndarray]:
        """
        Convert a 2D keypoint and depth to 3D point in camera frame.
        
        Args:
            keypoint: 2D keypoint (u, v)
            depth: Depth value in meters
            
        Returns:
            3D point (x, y, z) or None if depth is invalid
        """
        if depth <= 0 or depth > 10.0:  # Invalid or too far
            return None
        
        # Unproject using camera intrinsics
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        u, v = keypoint
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z])
    
    def transform_point(self, point_3d: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Transform a 3D point by a 4x4 transformation matrix."""
        point_homogeneous = np.append(point_3d, 1.0)
        transformed = transform @ point_homogeneous
        return transformed[:3]
    
    def create_landmarks_from_frame(self, frame_id: int, keypoints: list,
                                   descriptors: np.ndarray, depth_map: np.ndarray):
        """
        Create new landmarks from keypoints with valid depth.
        
        Args:
            frame_id: Current frame ID
            keypoints: List of cv2.KeyPoint objects
            descriptors: Feature descriptors
            depth_map: Depth map (same size as image)
        """
        current_pose = self.poses[frame_id]
        
        for kp, desc in zip(keypoints, descriptors):
            # Get keypoint coordinates
            u, v = int(kp.pt[0]), int(kp.pt[1])
            if v >= depth_map.shape[0] or u >= depth_map.shape[1]:
                continue
                
            depth = depth_map[v, u] / 5000.0  # TUM depth is in mm, convert to meters
            
            # Convert to 3D in camera frame
            point_3d_camera = self.depth_to_3d(np.array(kp.pt), depth)
            if point_3d_camera is None:
                continue
            
            # Transform to world frame
            point_3d_world = self.transform_point(point_3d_camera, current_pose)
            
            # Create landmark
            landmark = Landmark(self.next_landmark_id, point_3d_world, desc)
            landmark.add_observation(frame_id, np.array(kp.pt))
            
            self.landmarks[self.next_landmark_id] = landmark
            self.next_landmark_id += 1
            self.total_landmarks_created += 1
    
    def match_to_map(self, descriptors: np.ndarray, 
                    max_distance: float = 50.0) -> List[Tuple[int, int]]:
        """
        Match new descriptors to existing landmarks in the map.
        
        Args:
            descriptors: New descriptors to match
            max_distance: Maximum descriptor distance for a valid match
            
        Returns:
            List of (descriptor_idx, landmark_id) tuples
        """
        if not self.landmarks:
            return []
        
        # Get all landmark descriptors
        landmark_ids = list(self.landmarks.keys())
        map_descriptors = np.array([self.landmarks[lid].descriptor 
                                   for lid in landmark_ids])
        
        # Match using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        raw_matches = bf.match(descriptors, map_descriptors)
        
        # Filter by distance
        matches = []
        for m in raw_matches:
            if m.distance < max_distance:
                descriptor_idx = m.queryIdx
                landmark_id = landmark_ids[m.trainIdx]
                matches.append((descriptor_idx, landmark_id))
        
        return matches
    
    def detect_loop_closure(self, frame_id: int) -> Optional[int]:
        """
        Detect if current frame closes a loop with a previous frame.
        
        Args:
            frame_id: Current frame ID
            
        Returns:
            Previous frame ID if loop closure detected, None otherwise
        """
        # Simple loop closure: check if we're close to a previous pose
        # Skip recent frames (must be at least 50 frames apart)
        if frame_id < 50:
            return None
        
        current_pos = self.poses[frame_id][:3, 3]
        
        for i in range(0, frame_id - 50):
            prev_pos = self.poses[i][:3, 3]
            distance = np.linalg.norm(current_pos - prev_pos)
            
            if distance < self.loop_closure_threshold:
                return i
        
        return None
    
    def process_frame(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> bool:
        """
        Process a new RGB-D frame.
        
        Args:
            rgb_image: RGB image
            depth_image: Depth map (uint16, millimeters)
            
        Returns:
            True if processing was successful
        """
        self.frame_count += 1
        frame_id = len(self.poses) - 1  # Current frame index
        
        # Convert to grayscale for feature detection
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image
        
        # First frame: just initialize
        if self.prev_image is None:
            self.prev_image = gray.copy()
            self.prev_depth = depth_image.copy()
            kp, desc = self.feature_matcher.get_keypoints_and_descriptors(gray)
            self.prev_keypoints = kp
            self.prev_descriptors = desc
            
            # Create initial landmarks
            self.create_landmarks_from_frame(frame_id, kp, desc, depth_image)
            return True
        
        # Detect and match features
        pts1, pts2 = self.feature_matcher.detect_and_match(self.prev_image, gray)
        
        if len(pts1) < 8:
            print(f"Warning: Not enough matches ({len(pts1)}) in frame {self.frame_count}")
            return False
        
        self.total_matches += len(pts1)
        
        # Get 3D points for matched features using depth
        points_3d = []
        points_2d = []
        
        for i, pt1 in enumerate(pts1):
            u, v = int(pt1[0, 0]), int(pt1[0, 1])
            if v >= self.prev_depth.shape[0] or u >= self.prev_depth.shape[1]:
                continue
                
            depth = self.prev_depth[v, u] / 5000.0  # Convert to meters
            point_3d = self.depth_to_3d(pt1[0], depth)
            
            if point_3d is not None:
                points_3d.append(point_3d)
                points_2d.append(pts2[i][0])
        
        if len(points_3d) < 6:
            print(f"Warning: Not enough 3D points ({len(points_3d)}) in frame {self.frame_count}")
            return False
        
        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)
        
        # Solve PnP to get pose
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, self.K, None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            print(f"Warning: PnP failed in frame {self.frame_count}")
            return False
        
        # Convert to transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        
        # Create relative transformation
        T_relative = np.eye(4)
        T_relative[:3, :3] = R
        T_relative[:3, 3] = t
        
        # Update global pose
        # This was the bug: The relative transform must be inverted before
        # being applied to the global pose.
        self.current_pose = self.current_pose @ np.linalg.inv(T_relative)
        self.poses.append(self.current_pose.copy())
        
        # Create new landmarks from current frame
        current_kp, current_desc = self.feature_matcher.get_keypoints_and_descriptors(gray)
        self.create_landmarks_from_frame(len(self.poses) - 1, current_kp, 
                                        current_desc, depth_image)
        
        # Check for loop closure
        loop_frame = self.detect_loop_closure(len(self.poses) - 1)
        if loop_frame is not None:
            print(f"Loop closure detected: frame {len(self.poses) - 1} -> frame {loop_frame}")
            self.loop_closures.append((loop_frame, len(self.poses) - 1))
        
        # Update previous frame data
        self.prev_image = gray.copy()
        self.prev_depth = depth_image.copy()
        self.prev_keypoints = current_kp
        self.prev_descriptors = current_desc
        
        return True
    
    def process_sequence(self, rgb_paths: List[str], depth_paths: List[str],
                        max_frames: Optional[int] = None) -> np.ndarray:
        """
        Process a sequence of RGB-D frames.
        
        Args:
            rgb_paths: List of paths to RGB images
            depth_paths: List of paths to depth images
            max_frames: Maximum number of frames to process
            
        Returns:
            Array of estimated poses (N, 4, 4)
        """
        from tqdm import tqdm
        
        n_frames = len(rgb_paths)
        if max_frames is not None:
            n_frames = min(n_frames, max_frames)
        
        for i in tqdm(range(n_frames), desc="Processing frames"):
            rgb = cv2.imread(rgb_paths[i])
            depth = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)
            
            success = self.process_frame(rgb, depth)
            if not success:
                print(f"Failed to process frame {i}")
        
        return np.array(self.poses)
    
    def get_landmark_positions(self) -> np.ndarray:
        """Get all landmark positions as an array."""
        if not self.landmarks:
            return np.zeros((0, 3))
        
        positions = [lm.position for lm in self.landmarks.values()]
        return np.array(positions)
    
    def get_statistics(self) -> dict:
        """Get SLAM statistics."""
        return {
            'frames_processed': self.frame_count,
            'total_landmarks': len(self.landmarks),
            'landmarks_created': self.total_landmarks_created,
            'total_matches': self.total_matches,
            'loop_closures': len(self.loop_closures),
            'loop_closures_list': self.loop_closures,
            'avg_matches_per_frame': self.total_matches / max(self.frame_count, 1)
        }

