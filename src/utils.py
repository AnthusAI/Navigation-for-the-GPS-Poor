"""
Common utility functions for computer vision and navigation tasks.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from pathlib import Path


def load_image(image_path: str, grayscale: bool = True) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file
        grayscale: Whether to load as grayscale
        
    Returns:
        Loaded image array
    """
    if grayscale:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_and_match_features(img1: np.ndarray, img2: np.ndarray,
                             detector_type: str = 'ORB',
                             max_features: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect and match features between two images.
    
    Args:
        img1: First image
        img2: Second image  
        detector_type: Type of feature detector ('ORB', 'SIFT', 'SURF')
        max_features: Maximum number of features to detect
        
    Returns:
        Tuple of (matched_points1, matched_points2)
    """
    # Create detector
    if detector_type == 'ORB':
        detector = cv2.ORB_create(nfeatures=max_features)
    elif detector_type == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif detector_type == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    # Detect keypoints and descriptors
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    
    if desc1 is None or desc2 is None:
        return np.array([]), np.array([])
    
    # Match features
    if detector_type == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    return pts1, pts2


def estimate_pose_from_essential_matrix(pts1: np.ndarray, pts2: np.ndarray,
                                      K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate camera pose from matched points using essential matrix.
    
    Args:
        pts1: Points in first image
        pts2: Points in second image
        K: Camera intrinsic matrix
        
    Returns:
        Tuple of (rotation_matrix, translation_vector, inlier_mask)
    """
    # Compute essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, 
                                   prob=0.999, threshold=1.0)
    
    if E is None:
        return np.eye(3), np.zeros((3, 1)), np.array([])
    
    # Recover pose from essential matrix
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    return R, t, mask_pose


def triangulate_points(pts1: np.ndarray, pts2: np.ndarray,
                      P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D points from stereo correspondences.
    
    Args:
        pts1: Points in first image
        pts2: Points in second image
        P1: Projection matrix for first camera
        P2: Projection matrix for second camera
        
    Returns:
        3D points in homogeneous coordinates
    """
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T


def plot_trajectory(poses: np.ndarray, ground_truth: Optional[np.ndarray] = None,
                   title: str = "Camera Trajectory",
                   save_path: Optional[str] = None) -> None:
    """
    Plot camera trajectory in 3D.
    
    Args:
        poses: Array of 4x4 transformation matrices
        ground_truth: Optional ground truth poses for comparison
        title: Plot title
        save_path: If provided, saves the plot to this path instead of showing it.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    positions = poses[:, :3, 3]
    
    # KITTI coordinates: X=right, Y=down, Z=forward.
    # To get an intuitive plot, we'll map:
    # X -> plot's X axis (East/West)
    # Z -> plot's Y axis (North/South - Forward)
    # -Y -> plot's Z axis (Altitude - Up)
    ax.plot(positions[:, 0], positions[:, 2], -positions[:, 1], 
            'b-', label='Estimated', linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 2], -positions[0, 1], 
               c='green', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 2], -positions[-1, 1], 
               c='red', s=100, label='End')
    
    # Plot ground truth if provided
    if ground_truth is not None:
        gt_positions = ground_truth[:, :3, 3]
        ax.plot(gt_positions[:, 0], gt_positions[:, 2], -gt_positions[:, 1], 
                'r--', label='Ground Truth', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m) - Forward')
    ax.set_zlabel('-Y (m) - Altitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Set a more intuitive viewing angle
    ax.view_init(elev=20, azim=-60)
    
    if save_path:
        # Use a more aggressive tight_layout to ensure all labels are visible before saving
        fig.tight_layout(pad=3.0)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        # Use tight_layout with padding for notebook display
        fig.tight_layout(pad=2.0)
        plt.show()


def plot_top_down_trajectory(estimated_poses: np.ndarray,
                           ground_truth_poses: np.ndarray,
                           title: str = "Top-Down Trajectory View",
                           save_path: Optional[Path] = None,
                           color: str = 'b',
                           label: str = 'Estimated'):
    """
    Plots the top-down (X-Z plane) trajectory.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Extract positions (X and Z coordinates for top-down view)
    est_positions = estimated_poses[:, :3, 3]

    # Plot trajectories (X vs Z, which is the ground plane)
    ax.plot(est_positions[:, 0], est_positions[:, 2], color=color, label=label)
    
    if "Estimated" in label: # Only plot start/end/GT for the main estimate
        # Plot start and end points
        ax.scatter(est_positions[0, 0], est_positions[0, 2],
               c='green', s=100, label='Start')
        ax.scatter(est_positions[-1, 0], est_positions[-1, 2],
               c='red', s=100, label='End')

        # Plot ground truth
        gt_x = ground_truth_poses[:, 0, 3]
        gt_z = ground_truth_poses[:, 2, 3]
        ax.plot(gt_x, gt_z, '--', color='k', label='Ground Truth')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m) - Forward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
        plt.show()


def compute_trajectory_error(estimated_poses: np.ndarray, 
                           ground_truth_poses: np.ndarray) -> dict:
    """
    Compute trajectory estimation errors.
    
    Args:
        estimated_poses: Estimated 4x4 transformation matrices
        ground_truth_poses: Ground truth 4x4 transformation matrices
        
    Returns:
        Dictionary containing various error metrics
    """
    # Ensure same length
    min_len = min(len(estimated_poses), len(ground_truth_poses))
    est_pos = estimated_poses[:min_len, :3, 3]
    gt_pos = ground_truth_poses[:min_len, :3, 3]
    
    # Compute errors
    position_errors = np.linalg.norm(est_pos - gt_pos, axis=1)
    
    # Compute relative pose errors (RPE)
    rpe_trans = []
    rpe_rot = []
    
    for i in range(1, min_len):
        # Relative transformations
        rel_est = np.linalg.inv(estimated_poses[i-1]) @ estimated_poses[i]
        rel_gt = np.linalg.inv(ground_truth_poses[i-1]) @ ground_truth_poses[i]
        
        # Error in relative transformation
        rel_error = np.linalg.inv(rel_gt) @ rel_est
        
        # Translation error
        trans_error = np.linalg.norm(rel_error[:3, 3])
        rpe_trans.append(trans_error)
        
        # Rotation error (angle of rotation matrix)
        rot_error = np.arccos(np.clip((np.trace(rel_error[:3, :3]) - 1) / 2, -1, 1))
        rpe_rot.append(rot_error)
    
    return {
        'absolute_position_error': position_errors,
        'mean_position_error': np.mean(position_errors),
        'rms_position_error': np.sqrt(np.mean(position_errors**2)),
        'relative_position_error': np.array(rpe_trans),
        'relative_rotation_error': np.array(rpe_rot),
        'mean_rpe_trans': np.mean(rpe_trans),
        'mean_rpe_rot': np.mean(rpe_rot)
    }


def create_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Create projection matrix from intrinsics and extrinsics.
    
    Args:
        K: 3x3 intrinsic matrix
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        
    Returns:
        3x4 projection matrix
    """
    return K @ np.hstack([R, t.reshape(-1, 1)])


def normalize_points(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Normalize image points using camera intrinsics.
    
    Args:
        points: Image points
        K: Camera intrinsic matrix
        
    Returns:
        Normalized points
    """
    K_inv = np.linalg.inv(K)
    points_homo = np.hstack([points.reshape(-1, 2), np.ones((len(points), 1))])
    normalized = (K_inv @ points_homo.T).T
    return normalized[:, :2]
