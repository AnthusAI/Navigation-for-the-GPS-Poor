"""
Generate visualizations for Chapter 2: Stereo Visual Odometry

This script creates:
1. Stereo pair examples
2. Stereo feature matching visualizations
3. Disparity visualizations
4. Trajectory comparison (Stereo vs Mono vs Ground Truth)
5. Error analysis plots

Run from project root:
    python chapters/2/generate_visualizations.py
"""
import sys
sys.path.append('.')

from src.datasets import KITTIDatasetFetcher
from src.feature_matching import FeatureMatcher
from src.pose_estimation import PoseEstimator
from src.visual_odometry import run_vo_pipeline, run_stereo_vo_pipeline
from src.utils import load_image, plot_trajectory, compute_trajectory_error, plot_top_down_trajectory
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
SEQUENCE = "00"
N_FRAMES = 1000  # Process first 1000 frames for stereo comparison
OUTPUT_DIR = Path(f"chapters/2/images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors
COLOR_PURPLE = '#9400D3'
COLOR_BLUE = '#0077BE'
COLOR_RED = '#D95319'


def main():
    print("📸 Generating Chapter 2 Visualizations...")
    print("=" * 50)
    
    # --- 1. Load Data ---
    print("1. Loading KITTI data...")
    fetcher = KITTIDatasetFetcher()
    
    # Download if needed
    fetcher.download_sequence(SEQUENCE)
    
    left_image_paths = fetcher.get_image_paths(SEQUENCE, camera="image_0")
    right_image_paths = fetcher.get_image_paths(SEQUENCE, camera="image_1")
    gt_poses = fetcher.load_poses(SEQUENCE)
    calib = fetcher.load_calibration(SEQUENCE)
    
    # Get camera matrices and baseline
    K_left = calib['P2'][:3, :3]
    K_right = calib['P3'][:3, :3]
    
    # Baseline is the difference in the x-translation of the two cameras
    P2 = calib['P2']
    P3 = calib['P3']
    baseline = abs(P2[0, 3] / P2[0, 0] - P3[0, 3] / P3[0, 0])

    print(f"   ✅ Loaded {len(left_image_paths)} image pairs, {len(gt_poses)} poses.")
    print(f"   - Left Camera K:\n{K_left}")
    print(f"   - Stereo Baseline: {baseline:.4f} meters")

    # --- 2. Stereo Pair Example ---
    print("\n2. Visualizing a stereo pair...")
    img_left_color = load_image(left_image_paths[0], grayscale=False)
    img_right_color = load_image(right_image_paths[0], grayscale=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.imshow(img_left_color)
    ax1.set_title("Left Camera Image")
    ax1.axis('off')
    ax2.imshow(img_right_color)
    ax2.set_title("Right Camera Image")
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stereo_pair_example.png', dpi=150)
    print("   ✅ Saved: stereo_pair_example.png")

    # --- 2b. Stereo Feature Matching ---
    print("\n2b. Visualizing stereo feature matches...")
    feature_matcher = FeatureMatcher(max_features=500)
    img_left_gray = cv2.cvtColor(img_left_color, cv2.COLOR_RGB2GRAY)
    img_right_gray = cv2.cvtColor(img_right_color, cv2.COLOR_RGB2GRAY)

    kp1, des1 = feature_matcher.get_keypoints_and_descriptors(img_left_gray)
    kp2, des2 = feature_matcher.get_keypoints_and_descriptors(img_right_gray)
    
    matches = feature_matcher.matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50] # Top 50 matches

    img_matches = cv2.drawMatches(img_left_color, kp1, img_right_color, kp2, matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title("Top 50 Feature Matches Between Stereo Pair")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stereo_matches.png', dpi=150)
    print("   ✅ Saved: stereo_matches.png")

    # --- 2c. Disparity Visualization ---
    print("\n2c. Visualizing disparity map...")
    stereo = cv2.StereoBM_create(numDisparities=16*6, blockSize=21)
    disparity = stereo.compute(img_left_gray, img_right_gray)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(disparity, 'plasma')
    ax.set_title('Disparity Map (Closer = Brighter)', fontsize=14)
    ax.axis('off')
    
    # Add colorbar with custom size
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.6)
    cbar.set_label('Disparity (pixels)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'disparity_visualization.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: disparity_visualization.png")


    # --- 3. Run Stereo and Mono VO ---
    print(f"\n3. Running Stereo and Monocular VO on first {N_FRAMES} frames...")
    
    # Caching logic
    stereo_poses_cache = OUTPUT_DIR / f'stereo_poses_{N_FRAMES}.npy'
    mono_poses_cache = OUTPUT_DIR / f'mono_poses_{N_FRAMES}.npy'

    if stereo_poses_cache.exists():
        print("   Loading stereo poses from cache...")
        stereo_poses = np.load(stereo_poses_cache)
    else:
        print("   Running Stereo VO (this may take a while)...")
        stereo_poses = run_stereo_vo_pipeline(
            K_left, K_right, baseline,
            left_image_paths[:N_FRAMES],
            right_image_paths[:N_FRAMES],
            max_features=1000
        )
        np.save(stereo_poses_cache, stereo_poses)
    
    if mono_poses_cache.exists():
        print("   Loading monocular poses from cache...")
        mono_poses = np.load(mono_poses_cache)
    else:
        print("   Running Monocular VO for comparison...")
        mono_poses = run_vo_pipeline(
            K_left,
            left_image_paths[:N_FRAMES],
            gt_poses[:N_FRAMES],
            max_features=2000
        )
        np.save(mono_poses_cache, mono_poses)
        
    print("   ✅ VO processing complete.")

    # --- 4. Trajectory Comparison ---
    print("\n4. Generating trajectory comparison plots...")
    
    # Top-down comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot estimated trajectories
    ax.plot(mono_poses[:, 0, 3], mono_poses[:, 2, 3], color=COLOR_PURPLE, label='Monocular VO')
    ax.plot(stereo_poses[:, 0, 3], stereo_poses[:, 2, 3], color=COLOR_BLUE, label='Stereo VO')
    
    # Plot ground truth and start/end points once
    gt_x = gt_poses[:N_FRAMES, 0, 3]
    gt_z = gt_poses[:N_FRAMES, 2, 3]
    ax.plot(gt_x, gt_z, '--', color=COLOR_RED, label='Ground Truth')
    ax.scatter(gt_x[0], gt_z[0], c='g', s=100, label='Start', zorder=5)
    ax.scatter(gt_x[-1], gt_z[-1], c='r', s=100, label='End', zorder=5)
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m) - Forward")
    ax.set_title(f'Stereo vs Monocular VO Trajectory (First {N_FRAMES} frames)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    plt.savefig(OUTPUT_DIR / f'trajectory_comparison_{N_FRAMES}.png', dpi=150)
    print(f"   ✅ Saved: trajectory_comparison_{N_FRAMES}.png")

    # 3D comparison
    plot_trajectory(
        poses=stereo_poses,
        ground_truth=gt_poses[:N_FRAMES],
        title=f'3D Stereo VO Trajectory (First {N_FRAMES} frames)',
        save_path=OUTPUT_DIR / f'trajectory_comparison_3d_{N_FRAMES}.png'
    )
    print(f"   ✅ Saved: trajectory_comparison_3d_{N_FRAMES}.png")

    # --- 5. Error Analysis ---
    print("\n5. Generating error analysis plots...")
    
    mono_errors = compute_trajectory_error(mono_poses, gt_poses[:N_FRAMES])
    stereo_errors = compute_trajectory_error(stereo_poses, gt_poses[:N_FRAMES])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(mono_errors['absolute_position_error'], color=COLOR_PURPLE, label='Monocular')
    ax1.plot(stereo_errors['absolute_position_error'], color=COLOR_BLUE, label='Stereo')
    ax1.set_title('Absolute Position Error vs. Time')
    ax1.set_ylabel('Position Error (m)')
    ax1.legend()
    
    ax2.plot(np.rad2deg(mono_errors['relative_rotation_error']), color=COLOR_PURPLE, label='Monocular')
    ax2.plot(np.rad2deg(stereo_errors['relative_rotation_error']), color=COLOR_BLUE, label='Stereo')
    ax2.set_title('Relative Rotation Error (Frame-to-Frame)')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Rotation Error (degrees)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'error_comparison_{N_FRAMES}.png', dpi=150)
    print(f"   ✅ Saved: error_comparison_{N_FRAMES}.png")
    
    print("\n--- Error Summary ---")
    print(f"Monocular VO (1000 frames):")
    print(f"  - Mean Pos Error: {mono_errors['mean_position_error']:.3f} m")
    print(f"  - RMS Pos Error:  {mono_errors['rms_position_error']:.3f} m")
    print(f"Stereo VO (1000 frames):")
    print(f"  - Mean Pos Error: {stereo_errors['mean_position_error']:.3f} m")
    print(f"  - RMS Pos Error:  {stereo_errors['rms_position_error']:.3f} m")
    
    improvement = (mono_errors['rms_position_error'] - stereo_errors['rms_position_error']) / mono_errors['rms_position_error'] * 100
    print(f"\nImprovement in RMS Error: {improvement:.2f}%")


if __name__ == "__main__":
    main()
