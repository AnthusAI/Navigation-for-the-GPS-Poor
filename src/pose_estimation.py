"""
Pose estimation utilities for visual odometry and SLAM.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class PoseEstimator:
    """Camera pose estimation from feature correspondences."""
    
    def __init__(self, camera_matrix: np.ndarray):
        """
        Initialize pose estimator.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.K = camera_matrix
    
    def estimate_pose_essential_matrix(self, pts1: np.ndarray, pts2: np.ndarray,
                                     ransac_threshold: float = 1.0,
                                     confidence: float = 0.999) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate camera pose using essential matrix decomposition.
        
        Args:
            pts1: Points in first image (Nx1x2)
            pts2: Points in second image (Nx1x2)
            ransac_threshold: RANSAC threshold for outlier rejection
            confidence: RANSAC confidence level
            
        Returns:
            Tuple of (rotation_matrix, translation_vector, inlier_mask)
        """
        if len(pts1) < 8:
            return np.eye(3), np.zeros((3, 1)), np.array([])
        
        # Compute essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, 
            method=cv2.RANSAC,
            prob=confidence,
            threshold=ransac_threshold
        )
        
        if E is None:
            return np.eye(3), np.zeros((3, 1)), np.array([])
        
        # Recover pose from essential matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        return R, t, mask_pose
    
    def estimate_pose_pnp(self, points_3d: np.ndarray, points_2d: np.ndarray,
                         use_ransac: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate camera pose using Perspective-n-Point (PnP).
        
        Args:
            points_3d: 3D points in world coordinates (Nx3)
            points_2d: Corresponding 2D points in image (Nx2)
            use_ransac: Whether to use RANSAC for robust estimation
            
        Returns:
            Tuple of (rotation_vector, translation_vector, inlier_mask)
        """
        if len(points_3d) < 4:
            return np.zeros(3), np.zeros(3), np.array([])
        
        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros(4)
        
        if use_ransac:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, points_2d, self.K, dist_coeffs
            )
            if not success:
                return np.zeros(3), np.zeros(3), np.array([])
            return rvec.flatten(), tvec.flatten(), inliers.flatten()
        else:
            success, rvec, tvec = cv2.solvePnP(
                points_3d, points_2d, self.K, dist_coeffs
            )
            if not success:
                return np.zeros(3), np.zeros(3), np.array([])
            return rvec.flatten(), tvec.flatten(), np.arange(len(points_3d))
    
    def triangulate_points(self, pts1: np.ndarray, pts2: np.ndarray,
                          R1: np.ndarray, t1: np.ndarray,
                          R2: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from stereo correspondences.
        
        Args:
            pts1: Points in first image (Nx1x2 or Nx2)
            pts2: Points in second image (Nx1x2 or Nx2)
            R1, t1: Pose of first camera
            R2, t2: Pose of second camera
            
        Returns:
            3D points in homogeneous coordinates
        """
        # Reshape points to (N, 2) if needed
        pts1 = pts1.reshape(-1, 2)
        pts2 = pts2.reshape(-1, 2)
        
        # Create projection matrices
        P1 = self.K @ np.hstack([R1, t1.reshape(-1, 1)])
        P2 = self.K @ np.hstack([R2, t2.reshape(-1, 1)])
        
        # Triangulate points - needs (2, N) shape
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Convert to 3D by dividing by homogeneous coordinate
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def check_pose_validity(self, R: np.ndarray, t: np.ndarray,
                           pts1: np.ndarray, pts2: np.ndarray) -> bool:
        """
        Check if estimated pose is geometrically valid.
        
        Args:
            R: Rotation matrix
            t: Translation vector
            pts1: Points in first image
            pts2: Points in second image
            
        Returns:
            True if pose is valid, False otherwise
        """
        # Check if rotation matrix is valid
        if not self._is_rotation_matrix(R):
            return False
        
        # Check if translation is not zero
        if np.linalg.norm(t) < 1e-6:
            return False
        
        # Triangulate points and check if they are in front of both cameras
        try:
            points_3d = self.triangulate_points(
                pts1, pts2,
                np.eye(3), np.zeros(3),
                R, t.flatten()
            )
            
            # Check if points are in front of first camera (positive Z)
            # Use lenient 30% threshold - monocular VO can work with partial visibility
            if np.sum(points_3d[:, 2] > 0) < len(points_3d) * 0.3:
                return False
            
            # Transform points to second camera frame and check
            points_3d_cam2 = (R @ points_3d.T + t.reshape(-1, 1)).T
            if np.sum(points_3d_cam2[:, 2] > 0) < len(points_3d_cam2) * 0.3:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_rotation_matrix(self, R: np.ndarray, tolerance: float = 1e-3) -> bool:
        """Check if matrix is a valid rotation matrix."""
        # Check if matrix is 3x3
        if R.shape != (3, 3):
            return False
        
        # Check if R * R.T = I
        should_be_identity = np.dot(R, R.T)
        identity = np.eye(3)
        if not np.allclose(should_be_identity, identity, atol=tolerance):
            return False
        
        # Check if determinant is 1
        if not np.isclose(np.linalg.det(R), 1.0, atol=tolerance):
            return False
        
        return True
    
    def compute_reprojection_error(self, points_3d: np.ndarray, points_2d: np.ndarray,
                                  R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Compute reprojection error for pose validation.
        
        Args:
            points_3d: 3D points
            points_2d: Corresponding 2D points
            R: Rotation matrix
            t: Translation vector
            
        Returns:
            Array of reprojection errors for each point
        """
        # Transform 3D points to camera frame
        points_cam = (R @ points_3d.T + t.reshape(-1, 1)).T
        
        # Project to image plane
        points_proj = (self.K @ points_cam.T).T
        points_proj = points_proj[:, :2] / points_proj[:, 2:3]
        
        # Compute errors
        errors = np.linalg.norm(points_2d - points_proj, axis=1)
        
        return errors
