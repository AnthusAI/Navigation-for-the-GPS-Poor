"""
Visual odometry implementation for monocular and stereo cameras.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple
from pathlib import Path
from multiprocessing import Pool, cpu_count

from .feature_matching import FeatureMatcher
from .pose_estimation import PoseEstimator
from .utils import load_image
from tqdm import tqdm


class SimpleVisualOdometry:
    """Simple monocular visual odometry implementation."""
    
    def __init__(self, camera_matrix: np.ndarray, 
                 detector_type: str = 'ORB',
                 max_features: int = 1000):
        """
        Initialize visual odometry system.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            detector_type: Type of feature detector ('ORB', 'SIFT', 'SURF')
            max_features: Maximum number of features to detect
        """
        self.K = camera_matrix
        self.feature_matcher = FeatureMatcher(detector_type, max_features)
        self.pose_estimator = PoseEstimator(camera_matrix)
        
        # Initialize pose tracking
        self.current_pose = np.eye(4)
        self.poses = [self.current_pose.copy()]
        
        # Scale factor (monocular VO has scale ambiguity)
        self.scale = 1.0
        
        # Previous frame data
        self.prev_image = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Statistics
        self.frame_count = 0
        self.total_matches = 0
        self.total_inliers = 0
    
    def process_frame(self, image: np.ndarray, 
                     ground_truth_scale: Optional[float] = None) -> bool:
        """
        Process a new frame and update pose estimate.
        
        Args:
            image: Input image (grayscale)
            ground_truth_scale: Optional ground truth scale for this frame
            
        Returns:
            True if processing was successful, False otherwise
        """
        self.frame_count += 1
        
        # For first frame, just store it
        if self.prev_image is None:
            self.prev_image = image.copy()
            self.prev_keypoints, self.prev_descriptors = \
                self.feature_matcher.get_keypoints_and_descriptors(image)
            return True
        
        # Detect and match features
        pts1, pts2 = self.feature_matcher.detect_and_match(self.prev_image, image)
        
        if len(pts1) < 8:  # Need at least 8 points for essential matrix
            print(f"Warning: Not enough feature matches ({len(pts1)}) in frame {self.frame_count}")
            return False
        
        self.total_matches += len(pts1)
        
        # Estimate pose
        R, t, mask = self.pose_estimator.estimate_pose_essential_matrix(pts1, pts2)
        
        if mask is not None:
            self.total_inliers += np.sum(mask)
        
        # Validate pose
        # TODO: The validity check is currently too strict and causes frame skips
        # that accumulate significant error. Needs investigation and better criteria.
        # Temporarily disabled to match original Chapter 1 performance.
        # if not self.pose_estimator.check_pose_validity(R, t, pts1, pts2):
        #     print(f"Warning: Invalid pose estimate in frame {self.frame_count}")
        #     return False
        
        # Handle scale (monocular VO scale ambiguity)
        if ground_truth_scale is not None:
            self.scale = ground_truth_scale
        
        # Update pose - THIS IS THE CRITICAL PART
        # Ground truth gives: inv(pose[i-1]) @ pose[i] = relative_motion
        # So: pose[i] = pose[i-1] @ relative_motion  (right multiplication)
        # 
        # cv2.recoverPose gives us R,t where: pts_cam2 = R @ pts_cam1 + t
        # For WORLD-to-CAMERA matrices, this is the transformation we want directly
        # BUT KITTI poses are CAMERA-to-WORLD, so we need the inverse
        
        t_scaled = t * self.scale
        
        # Build the INVERSE transformation (world motion not camera motion)
        relative_transform = np.eye(4)
        relative_transform[:3, :3] = R.T
        relative_transform[:3, 3] = (-R.T @ t_scaled).flatten()
        
        # Right-multiply to match ground truth computation
        self.current_pose = self.current_pose @ relative_transform
        self.poses.append(self.current_pose.copy())
        
        # Update previous frame data
        self.prev_image = image.copy()
        
        return True
    
    def process_image_sequence(self, image_paths: List[str],
                             ground_truth_poses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process a sequence of images.
        
        Args:
            image_paths: List of paths to images
            ground_truth_poses: Optional ground truth poses for scale
            
        Returns:
            Array of estimated poses
        """
        print(f"Processing {len(image_paths)} images...")
        failed_frames = 0
        
        for i, img_path in enumerate(image_paths):
            # Load image
            image = load_image(img_path, grayscale=True)
            
            # Get ground truth scale if available
            gt_scale = None
            if ground_truth_poses is not None and i > 0:
                gt_relative = np.linalg.inv(ground_truth_poses[i-1]) @ ground_truth_poses[i]
                gt_scale = np.linalg.norm(gt_relative[:3, 3])
            
            # Process frame
            success = self.process_frame(image, gt_scale)
            
            if not success:
                # Skip this frame and continue with previous pose
                failed_frames += 1
                self.poses.append(self.current_pose.copy())  # Duplicate previous pose
                # Update previous image so next frame can continue
                self.prev_image = image.copy()
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} frames ({failed_frames} skipped)")
        
        if failed_frames > 0:
            print(f"Completed with {failed_frames} frames skipped due to pose estimation failures")
        
        return self.get_poses()
    
    def get_poses(self) -> np.ndarray:
        """Get all estimated poses as 4x4 transformation matrices."""
        return np.array(self.poses)
    
    def get_trajectory(self) -> np.ndarray:
        """Get trajectory as array of 3D positions."""
        poses = self.get_poses()
        return poses[:, :3, 3]
    
    def get_statistics(self) -> dict:
        """Get processing statistics."""
        avg_matches = self.total_matches / max(1, self.frame_count - 1)
        avg_inliers = self.total_inliers / max(1, self.frame_count - 1)
        inlier_ratio = self.total_inliers / max(1, self.total_matches)
        
        return {
            'frames_processed': self.frame_count,
            'total_matches': self.total_matches,
            'total_inliers': self.total_inliers,
            'avg_matches_per_frame': avg_matches,
            'avg_inliers_per_frame': avg_inliers,
            'inlier_ratio': inlier_ratio
        }
    
    def reset(self):
        """Reset the visual odometry system."""
        self.current_pose = np.eye(4)
        self.poses = [self.current_pose.copy()]
        self.scale = 1.0
        self.prev_image = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.frame_count = 0
        self.total_matches = 0
        self.total_inliers = 0


class StereoVisualOdometry:
    """
    Stereo visual odometry implementation.
    
    Uses two cameras with known baseline to resolve scale ambiguity.
    Optimized for CPU performance with efficient matching strategies.
    """
    
    def __init__(self, camera_matrix_left: np.ndarray,
                 camera_matrix_right: np.ndarray,
                 baseline: float,
                 detector_type: str = 'ORB',
                 max_features: int = 1000):
        """
        Initialize stereo visual odometry.
        
        Args:
            camera_matrix_left: Left camera intrinsic matrix
            camera_matrix_right: Right camera intrinsic matrix
            baseline: Stereo baseline distance (meters)
            detector_type: Feature detector ('ORB', 'SIFT')
            max_features: Maximum features to detect (lower = faster)
        """
        self.K_left = camera_matrix_left
        self.K_right = camera_matrix_right
        self.baseline = baseline
        self.detector_type = detector_type
        self.max_features = max_features
        
        # Initialize feature matcher and pose estimator
        self.feature_matcher = FeatureMatcher(self.detector_type, self.max_features)
        self.pose_estimator = PoseEstimator(camera_matrix_left)
        
        # State tracking
        self.current_pose = np.eye(4)
        self.poses = [self.current_pose.copy()]
        
        # Previous frame data
        self.prev_left_image = None
        self.prev_kp_left = None
        self.prev_des_left = None
        self.prev_points_3d = None
        
        # Statistics
        self.frame_count = 0
        self.total_stereo_matches = 0
        self.total_temporal_matches = 0
        self.total_inliers = 0
    
    def triangulate_stereo_points(self, pts_left: np.ndarray, pts_right: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from stereo correspondences.
        
        Args:
            pts_left: Points in left image (Nx2)
            pts_right: Corresponding points in right image (Nx2)
            
        Returns:
            3D points in left camera frame (Nx3)
        """
        # Create projection matrices for stereo pair
        # Left camera at origin
        P_left = self.K_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # Right camera shifted by baseline along X-axis
        R_right = np.eye(3)
        t_right = np.array([[-self.baseline], [0], [0]]) # Use -baseline for KITTI
        P_right = self.K_right @ np.hstack([R_right, t_right])
        
        # Correct triangulation call for stereo
        points_4d = cv2.triangulatePoints(P_left, P_right, pts_left.T, pts_right.T)
        
        # Convert to 3D and filter invalid points
        points_3d = points_4d[:3] / (points_4d[3] + 1e-8) # Add epsilon to avoid division by zero
        
        return points_3d.T
    
    def process_stereo_pair(self, img_left: np.ndarray, img_right: np.ndarray) -> bool:
        """
        Process a stereo image pair and update pose.
        
        Args:
            img_left: Left camera image (grayscale)
            img_right: Right camera image (grayscale)
            
        Returns:
            True if processing successful, False otherwise
        """
        self.frame_count += 1
        
        # Step 1: Match features between left and right (stereo matching)
        pts_left, pts_right = self.feature_matcher.detect_and_match(img_left, img_right)
        
        if len(pts_left) < 8:
            print(f"Warning: Not enough stereo matches ({len(pts_left)}) in frame {self.frame_count}")
            return False
        
        self.total_stereo_matches += len(pts_left)
        
        # Step 2: Triangulate to get 3D points
        points_3d = self.triangulate_stereo_points(pts_left, pts_right)
        
        # Filter out points that are too close or too far (likely errors)
        valid_mask = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 100)
        points_3d = points_3d[valid_mask]
        pts_left = pts_left[valid_mask]
        
        if len(points_3d) < 8:
            print(f"Warning: Not enough valid 3D points ({len(points_3d)}) in frame {self.frame_count}")
            return False
        
        # For first frame, just store the data
        if self.prev_left_image is None:
            self.prev_left_image = img_left.copy()
            kp, des = self.feature_matcher.get_keypoints_and_descriptors(img_left)
            self.prev_kp_left = kp
            self.prev_des_left = des
            self.prev_points_3d = points_3d
            return True
        
        # Step 3: Match current left image with previous left image (temporal matching)
        kp_curr, des_curr = self.feature_matcher.get_keypoints_and_descriptors(img_left)
        
        if des_curr is None or self.prev_des_left is None:
            return False

        matches = self.feature_matcher.matcher.knnMatch(des_curr, self.prev_des_left, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) < 8: return False

        # Step 4: Get 3D-2D correspondences
        # qryIdx = current frame, trainIdx = previous frame
        matched_3d = self.prev_points_3d[[m.trainIdx for m in good_matches]]
        matched_2d = np.array([kp_curr[m.queryIdx].pt for m in good_matches])
        
        # Step 5: Estimate pose using PnP
        rvec, tvec, inliers = self.pose_estimator.estimate_pose_pnp(matched_3d, matched_2d)
        
        if inliers is None or len(inliers) < 8: return False
        
        R, _ = cv2.Rodrigues(rvec)
        
        # Step 6: Update pose
        # PnP returns the transformation from world to camera frame.
        # The camera's pose in the world is the inverse of this.
        transform_cam_to_world = np.eye(4)
        transform_cam_to_world[:3, :3] = R.T
        transform_cam_to_world[:3, 3] = (-R.T @ tvec).flatten()
        
        # This gives the pose of the current camera *relative to the previous camera's frame*.
        # We need to chain this with the previous world pose.
        self.current_pose = self.current_pose @ transform_cam_to_world
        self.poses.append(self.current_pose.copy())
        
        # Update state for next iteration
        # The NEW 3D points are in the CURRENT camera's frame. We must transform
        # them into the WORLD frame before storing them for the next iteration.
        new_points_3d_world = (self.current_pose @ np.hstack([points_3d, np.ones((points_3d.shape[0], 1))]).T).T[:, :3]

        self.prev_left_image = img_left.copy()
        self.prev_points_3d = new_points_3d_world
        kp, des = self.feature_matcher.get_keypoints_and_descriptors(img_left)
        self.prev_kp_left = kp
        self.prev_des_left = des
        
        return True
    
    def process_image_sequence(self, left_image_paths: List[str],
                              right_image_paths: List[str]) -> np.ndarray:
        """
        Process a sequence of stereo image pairs.
        
        Args:
            left_image_paths: List of left camera image paths
            right_image_paths: List of right camera image paths
            
        Returns:
            Array of estimated poses (Nx4x4)
        """
        print(f"Processing {len(left_image_paths)} stereo pairs using {cpu_count()} CPU cores...")
        failed_frames = 0
        
        # Create a multiprocessing pool
        with Pool(processes=cpu_count()) as pool:
            
            # Prepare arguments for parallel processing
            tasks = []
            for i in range(len(left_image_paths)):
                tasks.append((
                    left_image_paths[i], right_image_paths[i],
                    self.detector_type, self.max_features,
                    self.K_left, self.K_right, self.baseline
                ))

            # Run processing in parallel
            results = list(tqdm(pool.imap(_process_pair_task, tasks), total=len(tasks), desc="Stereo Processing"))

        # Now, process results sequentially to build the trajectory
        for i, (success, result_data) in enumerate(results):
            if not success:
                failed_frames += 1
                self.poses.append(self.current_pose.copy())
                # For the next frame to work, we need an image to match against.
                # Since we don't have good 3D points, we just update the image.
                if i > 0:
                    self.prev_left_image = results[i-1][1]['img_left'] if results[i-1][0] else self.prev_left_image
                continue

            # Unpack results
            img_left = result_data['img_left']
            points_3d = result_data['points_3d']
            des_left = result_data['des_left']
            kp_left_pts = result_data['kp_left_pts'] # Now a numpy array

            if self.prev_des_left is None: # First frame
                self.prev_left_image = img_left
                self.prev_points_3d = points_3d
                self.prev_des_left = des_left
                self.prev_kp_left_pts = kp_left_pts # Store the filtered keypoints
                continue
            
            # Now, `prev_points_3d` and `prev_des_left` are guaranteed to be in sync
            matches = self.feature_matcher.matcher.knnMatch(des_left, self.prev_des_left, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) < 8: continue

            matched_3d = self.prev_points_3d[[m.trainIdx for m in good_matches]]
            matched_2d = kp_left_pts[[m.queryIdx for m in good_matches]] # Correct indexing!
            
            rvec, tvec, inliers = self.pose_estimator.estimate_pose_pnp(matched_3d, matched_2d)

            if inliers is None or len(inliers) < 8:
                self.poses.append(self.current_pose.copy())
                failed_frames += 1
                continue
                
            R, _ = cv2.Rodrigues(rvec)
            
            # PnP gives the transformation from world to the CURRENT camera frame.
            # The pose of the CURRENT camera in the world is the inverse of this.
            self.current_pose = np.eye(4)
            self.current_pose[:3, :3] = R.T
            self.current_pose[:3, 3] = (-R.T @ tvec).flatten()
            self.poses.append(self.current_pose.copy())
            
            # Update state for next iteration
            # Transform the NEW 3D points (which are in the CURRENT camera's frame) to the WORLD frame.
            new_points_3d_world = (self.current_pose @ np.hstack([points_3d, np.ones((points_3d.shape[0], 1))]).T).T[:, :3]
            
            self.prev_left_image = img_left
            self.prev_points_3d = new_points_3d_world
            self.prev_kp_left_pts = kp_left_pts
            self.prev_des_left = des_left


        if failed_frames > 0:
            print(f"Completed with {failed_frames} / {len(left_image_paths)} frames skipped.")
        
        return self.get_poses()

    def get_poses(self) -> np.ndarray:
        """Get all estimated poses as 4x4 transformation matrices."""
        return np.array(self.poses)
    
    def get_trajectory(self) -> np.ndarray:
        """Get trajectory as array of 3D positions."""
        poses = self.get_poses()
        return poses[:, :3, 3]
    
    def reset(self):
        """Reset the visual odometry system."""
        self.current_pose = np.eye(4)
        self.poses = [self.current_pose.copy()]
        self.prev_left_image = None
        self.prev_kp_left = None
        self.prev_des_left = None
        self.prev_points_3d = None
        self.frame_count = 0
        self.total_stereo_matches = 0
        self.total_temporal_matches = 0
        self.total_inliers = 0


def _process_pair_task(args):
    """Helper function for parallel processing. Performs stereo matching and triangulation."""
    left_path, right_path, detector_type, max_features, K_left, K_right, baseline = args
    
    # Each worker process must have its own instances of these objects
    feature_matcher = FeatureMatcher(detector_type, max_features)

    img_left = load_image(left_path, grayscale=True)
    img_right = load_image(right_path, grayscale=True)
    
    # Match left and right to get corresponding 2D points
    pts_left, pts_right = feature_matcher.detect_and_match(img_left, img_right)
    
    if len(pts_left) < 8: return False, {}
    
    # Reshape to (N, 2) immediately after matching
    pts_left = pts_left.reshape(-1, 2)
    pts_right = pts_right.reshape(-1, 2)

    # --- Add validation step to remove non-finite points ---
    finite_mask = np.isfinite(pts_left).all(axis=1) & np.isfinite(pts_right).all(axis=1)
    pts_left, pts_right = pts_left[finite_mask], pts_right[finite_mask]
    
    if len(pts_left) < 8:
        return False, {}

    # Triangulation logic needs to be self-contained here
    P_left = K_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_right = K_right @ np.hstack([np.eye(3), np.array([[-baseline], [0], [0]])]) # Use -baseline
    points_4d = cv2.triangulatePoints(P_left, P_right, pts_left.T, pts_right.T)
    
    # Check for valid triangulation result
    if points_4d is None or points_4d.shape[1] == 0:
        return False, {}
        
    points_3d = (points_4d[:3] / (points_4d[3] + 1e-8)).T
    
    if points_3d.shape[1] < 3:
        return False, {}
    
    valid_mask = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 100)
    points_3d = points_3d[valid_mask]
    pts_left = pts_left[valid_mask]

    if len(points_3d) < 8:
        return False, {}
        
    # Get all descriptors for the left image
    kp_left_all, des_left_all = feature_matcher.get_keypoints_and_descriptors(img_left)
    
    # Find the descriptors that correspond to our successfully matched pts_left
    from scipy.spatial import cKDTree
    tree = cKDTree(np.array([kp.pt for kp in kp_left_all]))
    _, indices = tree.query(pts_left.reshape(-1, 2), k=1)
    
    # Filter descriptors and 3D points based on valid triangulation
    valid_mask = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 100)
    
    final_indices = indices[valid_mask]
    final_points_3d = points_3d[valid_mask]
    
    final_des_left = des_left_all[final_indices]
    final_kp_left = np.array(kp_left_all)[final_indices] # Convert to array for indexing
    
    final_kp_left_array = np.array([kp.pt for kp in final_kp_left])
    
    if len(final_points_3d) < 8: return False, {}
        
    result_data = {
        'img_left': img_left,
        'points_3d': final_points_3d,
        'des_left': final_des_left,
        'kp_left_pts': final_kp_left_array # Return pickleable numpy array
    }
    return True, result_data


def run_vo_pipeline(camera_matrix: np.ndarray,
                   image_paths: List[str],
                   ground_truth_poses: Optional[np.ndarray] = None,
                   detector_type: str = 'ORB',
                   max_features: int = 2000) -> np.ndarray:
    """
    Convenience function to run monocular VO pipeline.
    
    Args:
        camera_matrix: 3x3 camera intrinsic matrix
        image_paths: List of image file paths
        ground_truth_poses: Optional ground truth for scale
        detector_type: Feature detector type
        max_features: Maximum features to detect
        
    Returns:
        Array of estimated poses
    """
    vo = SimpleVisualOdometry(camera_matrix, detector_type, max_features)
    return vo.process_image_sequence(image_paths, ground_truth_poses)


def run_stereo_vo_pipeline(camera_matrix_left: np.ndarray,
                           camera_matrix_right: np.ndarray,
                           baseline: float,
                           left_image_paths: List[str],
                           right_image_paths: List[str],
                           detector_type: str = 'ORB',
                           max_features: int = 1000) -> np.ndarray:
    """
    Convenience function to run stereo VO pipeline.
    
    Args:
        camera_matrix_left: Left camera intrinsic matrix
        camera_matrix_right: Right camera intrinsic matrix
        baseline: Stereo baseline (meters)
        left_image_paths: List of left image paths
        right_image_paths: List of right image paths
        detector_type: Feature detector type
        max_features: Maximum features (lower for stereo = faster)
        
    Returns:
        Array of estimated poses
    """
    vo = StereoVisualOdometry(camera_matrix_left, camera_matrix_right, baseline,
                              detector_type, max_features)
    return vo.process_image_sequence(left_image_paths, right_image_paths)
