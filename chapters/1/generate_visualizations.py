"""
Generate visualizations for Chapter 1: Visual Odometry

This script creates annotated images showing:
1. Original KITTI frames
2. Detected features (corners)
3. Feature matches between frames
4. Trajectory visualization

Run from project root:
    python chapters/1/generate_visualizations.py
"""

import sys
sys.path.append('.')

from src.datasets import KITTIDatasetFetcher
from src.feature_matching import FeatureMatcher
from src.pose_estimation import PoseEstimator
from src.visual_odometry import SimpleVisualOdometry
from src.utils import load_image, plot_trajectory, plot_top_down_trajectory, compute_trajectory_error
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import matplotlib.patches as mpatches

# Miami color theme -> Updated to Purple/Blue
COLOR_PURPLE = '#9400D3' # Bright Purple
COLOR_BLUE = '#007BFF'   # A more visible Blue
COLOR_BLUE_BGR = (255, 123, 0)
COLOR_PURPLE_BGR = (211, 0, 148)


# Create output directory
output_dir = Path('chapters/1/images')
output_dir.mkdir(exist_ok=True, parents=True)

print('📸 Generating Chapter 1 Visualizations...')
print('=' * 60)

# Load KITTI data
kitti = KITTIDatasetFetcher('data')
image_paths = kitti.get_image_paths('00', camera='image_0')
calib = kitti.load_calibration('00')
K = calib['P0'][:, :3]
gt_poses = kitti.load_poses('00')

print(f'✅ Loaded {len(image_paths)} images from KITTI sequence 00')
print()

# 1. Original Frame
print('1. Saving original KITTI frames...')
img = cv2.imread(image_paths[0])
cv2.imwrite(str(output_dir / 'frame_000.png'), img)
img = cv2.imread(image_paths[1])
cv2.imwrite(str(output_dir / 'frame_001.png'), img)
img = cv2.imread(image_paths[10])
cv2.imwrite(str(output_dir / 'frame_010.png'), img)
print(f'   ✅ Saved 3 sample frames')

# 2. Features Detected
print('2. Detecting and visualizing features...')
matcher = FeatureMatcher('ORB', max_features=1000)
img_gray = load_image(image_paths[0], grayscale=True)
kp, desc = matcher.get_keypoints_and_descriptors(img_gray)

# Draw keypoints on image
img_with_keypoints = cv2.imread(image_paths[0])
for i in range(min(len(kp), 200)):  # Show first 200 features
    pt = (int(kp[i].pt[0]), int(kp[i].pt[1]))
    cv2.circle(img_with_keypoints, pt, 3, COLOR_PURPLE_BGR, -1)
    cv2.circle(img_with_keypoints, pt, 6, COLOR_PURPLE_BGR, 1)

cv2.imwrite(str(output_dir / 'features_detected.png'), img_with_keypoints)
print(f'   ✅ Detected {len(kp)} features, visualized 200')

# 3. Feature Matches
print('3. Matching features between consecutive frames...')
img1_gray = load_image(image_paths[0], grayscale=True)
img2_gray = load_image(image_paths[1], grayscale=True)
pts1, pts2 = matcher.detect_and_match(img1_gray, img2_gray)

matches_img = matcher.visualize_matches(img1_gray, img2_gray, pts1, pts2, max_matches=50)
cv2.imwrite(str(output_dir / 'feature_matches.png'), cv2.cvtColor(matches_img, cv2.COLOR_RGB2BGR))
print(f'   ✅ Matched 607 features, visualized 50')

# 4. Trajectory Visualization
print('4. Running visual odometry and plotting trajectory...')

# --- Caching logic for the full run ---
poses_cache_file = output_dir / 'mono_poses_full.npy'
n_frames = len(image_paths) # Use all available frames

if poses_cache_file.exists():
    print(f'   Loading {n_frames} poses from cache...')
    estimated_poses = np.load(poses_cache_file)
    print('   ✅ Loaded poses from cache.')
else:
    print(f'   Processing all {n_frames} frames (this will take a while)...')
    vo = SimpleVisualOdometry(K, detector_type='SIFT', max_features=2000)
    estimated_poses = vo.process_image_sequence(
        image_paths,
        ground_truth_poses=gt_poses
    )
    np.save(poses_cache_file, estimated_poses)
    print(f'   ✅ Processing complete. Saved {len(estimated_poses)} poses to cache.')

# If we didn't get enough poses, print warning but continue
if len(estimated_poses) < 10:
    print(f'   ⚠️  Warning: Only processed {len(estimated_poses)} frames successfully')
    print('   Continuing with available data...')
    n_frames = len(estimated_poses)

# Generate TWO sets of visualizations:
# 1. First 1000 frames - shows good tracking and altitude analysis
# 2. All frames - shows long-term drift accumulation

print('   Generating trajectory visualizations...')

# 1000-frame visualization (for 3D view - shows altitude clearly)
n_frames_3d = min(200, len(estimated_poses))
plot_trajectory(
    estimated_poses[:n_frames_3d],
    gt_poses[:n_frames_3d],
    title=f'3D Visual Odometry Trajectory (First {n_frames_3d} frames)',
    save_path=output_dir / 'trajectory_3d_200.png'
)

# 1000-frame top-down (shows good tracking)
n_frames_1k = min(1000, len(estimated_poses))
plot_top_down_trajectory(
    estimated_poses[:n_frames_1k],
    gt_poses[:n_frames_1k],
    title=f'Top-Down Trajectory View (First {n_frames_1k} frames)',
    save_path=output_dir / 'trajectory_topdown_1000.png'
)

# Full sequence top-down (shows long-term drift - chaotic but educational)
plot_top_down_trajectory(
    estimated_poses,
    gt_poses[:len(estimated_poses)],
    title=f'Top-Down Trajectory View (Full Sequence - {len(estimated_poses)} frames)',
    save_path=output_dir / 'trajectory_topdown_full.png'
)

print(f"   ✅ Trajectory images saved:")
print(f"      - trajectory_3d_200.png (first 200 frames, shows altitude failure)")
print(f"      - trajectory_topdown_1000.png (first 1000 frames, good tracking)")
print(f"      - trajectory_topdown_full.png (all {len(estimated_poses)} frames, shows drift)")

# 5. Computing and visualizing errors...
print('5. Computing and visualizing errors...')

# Compute errors for both 1000-frame and full sequence
errors_1k = compute_trajectory_error(estimated_poses[:n_frames_1k], gt_poses[:n_frames_1k])
errors_full = compute_trajectory_error(estimated_poses, gt_poses)

# Plot error analysis for full sequence
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(errors_full['absolute_position_error'], color=COLOR_PURPLE)
axes[0].set_title('Translation Error Over Time (Full Sequence)')
axes[0].set_ylabel('Position Error (m)')
axes[0].grid(True)

axes[1].plot(np.rad2deg(errors_full['relative_rotation_error']), color=COLOR_BLUE)
axes[1].set_title('Rotational Error (Frame-to-Frame)')
axes[1].set_xlabel('Frame')
axes[1].set_ylabel('Rotation Error (degrees)')
axes[1].grid(True)
plt.tight_layout()
plt.savefig(output_dir / 'error_analysis_full.png', dpi=150, bbox_inches='tight')

# Plot error analysis for first 1000 frames (cleaner, more readable)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(errors_1k['absolute_position_error'], color=COLOR_PURPLE)
axes[0].set_title(f'Translation Error Over Time (First {n_frames_1k} frames)')
axes[0].set_ylabel('Position Error (m)')
axes[0].grid(True)

axes[1].plot(np.rad2deg(errors_1k['relative_rotation_error']), color=COLOR_BLUE)
axes[1].set_title('Rotational Error (Frame-to-Frame)')
axes[1].set_xlabel('Frame')
axes[1].set_ylabel('Rotation Error (degrees)')
axes[1].grid(True)
plt.tight_layout()
plt.savefig(output_dir / 'error_analysis_1000.png', dpi=150, bbox_inches='tight')

print("   ✅ Error analysis plots saved")
print(f"   First 1000 frames:")
print(f"      - Mean position error: {errors_1k['mean_position_error']:.3f} m")
print(f"      - RMS position error: {errors_1k['rms_position_error']:.3f} m")
print(f"   Full sequence ({len(estimated_poses)} frames):")
print(f"      - Mean position error: {errors_full['mean_position_error']:.3f} m")
print(f"      - RMS position error: {errors_full['rms_position_error']:.3f} m")

# 6. Generating statistics summary...
print('6. Generating statistics summary...')
stats_text = f"""
Visual Odometry Statistics
{'=' * 40}
- **Sequence**: 00
- **Frames Processed**: {len(estimated_poses)} / {len(image_paths)}
- **Feature Detector**: ORB
- **Max Features**: 2000

## Performance (First 1000 frames)
- **Mean Position Error**: {errors_1k['mean_position_error']:.3f} m
- **Root Mean Square (RMS) Error**: {errors_1k['rms_position_error']:.3f} m
- **Mean Rotational Error (RPE)**: {np.rad2deg(errors_1k['mean_rpe_rot']):.3f} degrees/frame

## Performance (Full Sequence - {len(estimated_poses)} frames)
- **Mean Position Error**: {errors_full['mean_position_error']:.3f} m
- **Root Mean Square (RMS) Error**: {errors_full['rms_position_error']:.3f} m
- **Mean Rotational Error (RPE)**: {np.rad2deg(errors_full['mean_rpe_rot']):.3f} degrees/frame

## Interpretation
The first 1000 frames show good tracking with reasonable drift.
The full sequence shows significant drift accumulation - this is expected
for monocular VO over long distances without loop closure correction.
"""
with open(output_dir / 'statistics.txt', 'w') as f:
    f.write(stats_text)
print('   ✅ Statistics saved')

# 7. Animated GIFs: raw frames
print('7. Creating animated GIF from raw frames...')

def _resize_rgb(img_rgb: np.ndarray, target_width: int) -> np.ndarray:
    """Resize an RGB image to the target width while keeping aspect ratio."""
    h, w = img_rgb.shape[:2]
    if w <= target_width:
        return img_rgb
    scale = target_width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_AREA)

def generate_gif_from_images(paths: list, output_path: Path, step: int = 5,
                             max_frames: int = 300, resize_width: int = 800,
                             duration_ms: int = 80) -> None:
    frames = []
    count = 0
    for idx in range(0, len(paths), step):
        if count >= max_frames:
            break
        bgr = cv2.imread(paths[idx])
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = _resize_rgb(rgb, resize_width)
        frames.append(Image.fromarray(rgb))
        count += 1
    if not frames:
        print('   ⚠️  No frames collected for raw GIF, skipping...')
        return
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    print(f"   ✅ Saved raw sequence GIF: {output_path.name} ({len(frames)} frames)")

raw_gif_path = output_dir / 'sequence_00_raw.gif'
generate_gif_from_images(image_paths, raw_gif_path, step=5, max_frames=300, resize_width=800, duration_ms=80)

# 8. Animated GIFs: feature-annotated frames
print('8. Creating animated GIF with detected features...')

def generate_feature_annotated_gif(paths: list, output_path: Path, step: int = 5,
                                   max_frames: int = 300, resize_width: int = 800,
                                   detector_type: str = 'ORB', max_features: int = 1000,
                                   draw_limit: int = 300, duration_ms: int = 80) -> None:
    fm = FeatureMatcher(detector_type=detector_type, max_features=max_features)
    frames = []
    count = 0
    for idx in range(0, len(paths), step):
        if count >= max_frames:
            break
        gray = load_image(paths[idx], grayscale=True)
        if gray is None:
            continue
        kp, _ = fm.get_keypoints_and_descriptors(gray)
        # Draw up to draw_limit keypoints for clarity
        draw_kp = kp[:draw_limit] if kp is not None else []
        color_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        annotated = cv2.drawKeypoints(color_bgr, draw_kp, None,
                                      color=COLOR_PURPLE_BGR, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        rgb = _resize_rgb(rgb, resize_width)
        frames.append(Image.fromarray(rgb))
        count += 1
    if not frames:
        print('   ⚠️  No frames collected for feature GIF, skipping...')
        return
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    print(f"   ✅ Saved feature-annotated GIF: {output_path.name} ({len(frames)} frames)")

features_gif_path = output_dir / 'sequence_00_features.gif'
generate_feature_annotated_gif(image_paths, features_gif_path, step=5, max_frames=300, resize_width=800,
                               detector_type='SIFT', max_features=2000, draw_limit=300, duration_ms=80)

# 9. NEW: Simple Analogy (Parallax) Visualization
print('9. Creating simple analogy illustration (parallax)...')

def generate_parallax_analogy(output_path: Path):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # Top-down view
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_title("Top-Down View: A car drives past a building", fontsize=16)
    # Building
    ax_top.add_patch(plt.Rectangle((4, 2), 2, 1, facecolor='gray', edgecolor='black', label='Building'))
    # Road
    ax_top.axhline(0, color='black', linestyle='--')
    # Car positions
    car1_pos = (2, 0)
    car2_pos = (8, 0)
    ax_top.add_patch(plt.Rectangle((car1_pos[0]-0.5, -0.25), 1, 0.5, facecolor=COLOR_PURPLE, alpha=0.6, edgecolor='black', label='Car Position A'))
    ax_top.text(car1_pos[0], car1_pos[1] - 0.5, 'A', ha='center', fontsize=14, fontweight='bold')
    ax_top.add_patch(plt.Rectangle((car2_pos[0]-0.5, -0.25), 1, 0.5, facecolor=COLOR_BLUE, alpha=0.6, edgecolor='black', label='Car Position B'))
    ax_top.text(car2_pos[0], car2_pos[1] - 0.5, 'B', ha='center', fontsize=14, fontweight='bold')
    ax_top.arrow(3, 0, 4, 0, head_width=0.2, head_length=0.4, fc='black', ec='black')
    # Lines of sight
    building_center = (5, 2.5)
    ax_top.plot([car1_pos[0], building_center[0]], [car1_pos[1], building_center[1]], color=COLOR_PURPLE, linestyle='--', alpha=0.7)
    ax_top.plot([car2_pos[0], building_center[0]], [car2_pos[1], building_center[1]], color=COLOR_BLUE, linestyle='--', alpha=0.7)
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(-1, 4)
    ax_top.set_aspect('equal')
    ax_top.legend(loc='upper left')
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    # Camera View A
    ax_cam1 = fig.add_subplot(gs[1, 0])
    ax_cam1.set_title("View from Position A", fontsize=14)
    ax_cam1.add_patch(plt.Rectangle((0.6, 0.2), 0.3, 0.6, facecolor='gray', edgecolor='black'))
    ax_cam1.set_xlim(0, 1)
    ax_cam1.set_ylim(0, 1)
    ax_cam1.set_xticks([])
    ax_cam1.set_yticks([])
    ax_cam1.spines['top'].set_visible(False)
    ax_cam1.spines['right'].set_visible(False)
    ax_cam1.spines['bottom'].set_visible(False)
    ax_cam1.spines['left'].set_visible(False)
    ax_cam1.set_facecolor('#d3d3d3')


    # Camera View B
    ax_cam2 = fig.add_subplot(gs[1, 1])
    ax_cam2.set_title("View from Position B", fontsize=14)
    ax_cam2.add_patch(plt.Rectangle((0.1, 0.2), 0.3, 0.6, facecolor='gray', edgecolor='black'))
    ax_cam2.set_xlim(0, 1)
    ax_cam2.set_ylim(0, 1)
    ax_cam2.set_xticks([])
    ax_cam2.set_yticks([])
    ax_cam2.spines['top'].set_visible(False)
    ax_cam2.spines['right'].set_visible(False)
    ax_cam2.spines['bottom'].set_visible(False)
    ax_cam2.spines['left'].set_visible(False)
    ax_cam2.set_facecolor('#d3d3d3')

    fig.suptitle("The building appears to move across the field of view as the car drives past", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f'   ✅ Simple analogy illustration saved as {output_path.name}')

generate_parallax_analogy(output_dir / 'simple_analogy_parallax.png')


# 10. NEW: Camera Calibration (Distortion) Visualization
print('10. Creating camera calibration illustration (distortion)...')

def generate_calibration_analogy(output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    # Create a grid
    x = np.linspace(-1, 1, 15)
    y = np.linspace(-1, 1, 15)
    xx, yy = np.meshgrid(x, y)
    
    # Apply barrel distortion
    k1 = 0.2  # Distortion coefficient
    r2 = xx**2 + yy**2
    x_distorted = xx * (1 + k1 * r2)
    y_distorted = yy * (1 + k1 * r2)

    # Plot Distorted Grid
    ax1.set_title("1. Distorted Image from Camera Lens", fontsize=16)
    for i in range(len(x)):
        ax1.plot(x_distorted[i, :], y_distorted[i, :], 'k-')
        ax1.plot(x_distorted[:, i], y_distorted[:, i], 'k-')
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Plot Corrected Grid
    ax2.set_title("2. Corrected Image After Calibration", fontsize=16)
    for i in range(len(x)):
        ax2.plot(xx[i, :], yy[i, :], 'k-')
        ax2.plot(xx[:, i], yy[:, i], 'k-')
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Arrow and text box, redesigned for clarity
    fig.text(0.5, 0.4, 'Calibration\nFinds the Math to Fix This',
             ha='center', va='center', fontsize=20,
             bbox=dict(boxstyle='round,pad=0.5', fc=COLOR_BLUE, alpha=0.8))
    
    arrow = mpatches.FancyArrowPatch((0.45, 0.55), (0.55, 0.55),
                                     mutation_scale=40, transform=fig.transFigure,
                                     color=COLOR_PURPLE)
    fig.patches.append(arrow)

    fig.suptitle("Camera Calibration Corrects Lens Distortion", fontsize=20, y=0.95)
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f'   ✅ Camera calibration illustration saved as {output_path.name}')

generate_calibration_analogy(output_dir / 'camera_calibration_distortion.png')

# 11. NEW: Triangulation 3D Diagram (for Step 3)
print('11. Creating new 3D triangulation diagram...')

def look_at(eye, target, up):
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    R = np.vstack([x_axis, y_axis, z_axis])
    return R

def generate_triangulation_diagram(output_path: Path):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Config
    up_vector = np.array([0, 1, 0])
    point_3d = np.array([0, 0.5, 10])
    cam_a_pos = np.array([-2, 0, 0])
    cam_b_pos = np.array([2, 0, 0])

    # Calculate rotations to look at the point
    R_a = look_at(cam_a_pos, point_3d, up_vector)
    R_b = look_at(cam_b_pos, point_3d, up_vector)

    # Create poses
    pose1 = np.eye(4)
    pose1[:3, :3] = R_a.T
    pose1[:3, 3] = cam_a_pos
    
    pose2 = np.eye(4)
    pose2[:3, :3] = R_b.T
    pose2[:3, 3] = cam_b_pos

    # Draw camera frustums and projections
    for pose, color, label in [(pose1, COLOR_PURPLE, 'A'), (pose2, COLOR_BLUE, 'B')]:
        center = pose[:3, 3]
        R = pose[:3, :3]
        
        # Frustum
        frustum_pts = np.array([[-1, -0.75, 2], [1, -0.75, 2], [1, 0.75, 2], [-1, 0.75, 2], [0,0,0]])
        frustum_world = (R @ frustum_pts.T).T + center
        
        for i in range(4):
            ax.plot([frustum_world[4,0], frustum_world[i,0]], [frustum_world[4,1], frustum_world[i,1]], [frustum_world[4,2], frustum_world[i,2]], color=color, alpha=0.4)
        ax.plot(frustum_world[[0,1,2,3,0],0], frustum_world[[0,1,2,3,0],1], frustum_world[[0,1,2,3,0],2], color=color)

        # Projection
        pt_cam = np.linalg.inv(R) @ (point_3d - center)
        proj_pt = (pt_cam[:2] / pt_cam[2]) * 2 # f=2
        proj_pt_world = (R @ np.array([proj_pt[0], proj_pt[1], 2])).T + center
        ax.scatter(proj_pt_world[0], proj_pt_world[1], proj_pt_world[2], c=color, marker='x', s=100)
        
        # Ray
        ax.plot([center[0], point_3d[0]], [center[1], point_3d[1]], [center[2], point_3d[2]], '--', color=color, alpha=0.8)
        
        ax.text(center[0], center[1] - 0.5, center[2], f'Camera {label}', color=color, fontsize=12)

    ax.scatter(point_3d[0], point_3d[1], point_3d[2], c='red', s=150, label='Real-world Point (X)')
    ax.text(point_3d[0], point_3d[1]+0.3, point_3d[2], 'X', color='red', fontsize=14)

    ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
    ax.view_init(elev=25., azim=-70)
    ax.legend()
    ax.set_title('Triangulation: Locating a 3D point from two different views', fontsize=18)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f'   ✅ New 3D triangulation diagram saved as {output_path.name}')

generate_triangulation_diagram(output_dir / 'triangulation_3d_illustration.png')


# 12. NEW: Camera Intrinsics Diagram (for Calibration section)
print('12. Creating new camera intrinsics diagram...')

def generate_intrinsics_diagram(output_path: Path):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2)

    # 3D Pinhole Model View
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_title('1. Camera projects 3D world onto 2D plane', fontsize=14)
    
    # 3D point
    pt_3d = np.array([2, 1, 5])
    ax1.scatter(pt_3d[0], pt_3d[1], pt_3d[2], s=100, c=COLOR_PURPLE, label='3D Point')
    
    # Camera center and image plane
    f = 1  # Focal length
    img_plane = plt.Rectangle((-1.5, -1), 3, 2, color=COLOR_BLUE, alpha=0.3)
    ax1.add_patch(img_plane)
    from mpl_toolkits.mplot3d import art3d
    art3d.patch_2d_to_3d(img_plane, z=f, zdir='z')
    
    # Projection
    pt_2d = (pt_3d[:2] / pt_3d[2]) * f
    ax1.scatter(pt_2d[0], pt_2d[1], f, s=80, c=COLOR_BLUE, marker='x', label='Projected Point')
    
    # Rays
    ax1.plot([0, pt_3d[0]], [0, pt_3d[1]], [0, pt_3d[2]], '--', color=COLOR_PURPLE, alpha=0.8)
    ax1.plot([0, pt_2d[0]], [0, pt_2d[1]], [0, f], color=COLOR_BLUE)
    
    # Axes and labels
    ax1.plot([0, 0], [0, 0], [0, f], 'k-', label='Focal Length (f)')
    ax1.text(0.1, 0.1, f/2, 'f')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z (Optical Axis)')
    ax1.view_init(elev=20, azim=-70)
    ax1.legend()

    # 2D Pixel Coordinate View
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('2. Image plane is converted to pixel coordinates', fontsize=14)
    
    h, w = 480, 640
    cx, cy = w / 2, h / 2
    
    # Pixel grid
    ax2.set_xlim(0, w); ax2.set_ylim(h, 0)
    ax2.set_aspect('equal')
    ax2.set_xlabel('u (pixels)')
    ax2.set_ylabel('v (pixels)')
    ax2.xaxis.tick_top(); ax2.xaxis.set_label_position('top')
    
    # Principal point
    ax2.plot(cx, cy, '+', markersize=15, color=COLOR_PURPLE, label=f'Principal Point (cx, cy) = ({cx}, {cy})')
    
    # Projected point in pixels
    fx, fy = 500, 500 # Focal length in pixels
    u = fx * pt_2d[0] + cx
    v = fy * pt_2d[1] + cy
    ax2.plot(u, v, 'x', markersize=12, color=COLOR_BLUE, label=f'Pixel Coords (u, v) = ({u:.0f}, {v:.0f})')
    
    ax2.legend()
    fig.suptitle('Camera Intrinsics: How a 3D point becomes a pixel', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f'   ✅ New camera intrinsics diagram saved as {output_path.name}')

generate_intrinsics_diagram(output_dir / 'camera_intrinsics_illustration.png')


print()
print('✅ All visualizations generated!')
print(f'📁 Output directory: {output_dir.absolute()}')
print()
print('Generated files:')
for file in sorted(output_dir.glob('*.png')):
    print(f'   - {file.name}')
for file in sorted(output_dir.glob('*.gif')):
    print(f'   - {file.name}')
print(f'   - statistics.txt')
print()
print('Attribution for KITTI images:')
print('   Source: KITTI Vision Benchmark Suite')
print('   URL: http://www.cvlibs.net/datasets/kitti/')
print('   Citation: Geiger et al., "Are we ready for Autonomous Driving?"')
print('             CVPR 2012')

