"""
Generate visualizations for Chapter 3: SLAM Fundamentals

This script creates annotated images showing:
1. Raw RGB-D sequence (GIF)
2. Sample RGB and depth images
3. SLAM trajectory vs visual odometry
4. 3D landmark map visualization
5. Loop closure detection

Run from project root:
    python chapters/3/generate_visualizations.py
"""

import sys
sys.path.append('.')

from src.datasets import TUMDatasetFetcher
from src.slam import RGBDSlam
from src.visual_odometry import SimpleVisualOdometry
from src.utils import plot_trajectory, plot_top_down_trajectory, compute_trajectory_error
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from tqdm import tqdm
import matplotlib.patches as mpatches

# Color theme (consistent with Chapter 1 & 2)
COLOR_PURPLE = '#9400D3'
COLOR_BLUE = '#007BFF'
COLOR_GREEN = '#00D900'
COLOR_RED = '#FF0000'

def create_loop_closure_diagram(output_path: Path):
    """Create a diagram illustrating the concept of loop closure."""
    if output_path.exists():
        print(f"'{output_path.name}' already exists. Skipping generation.")
        return

    print("   Creating loop closure diagram...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Drifting Trajectory (a spiral)
    t = np.linspace(0, 4 * np.pi, 100)
    x = t * np.cos(t) * 0.1
    y = t * np.sin(t) * 0.1
    ax.plot(x, y, '--', color='gray', label='Odometry Estimate (with drift)')

    # 2. Key Poses
    pose_5 = (x[5], y[5])
    pose_95 = (x[95], y[95])
    ax.scatter([pose_5[0]], [pose_5[1]], c=COLOR_BLUE, s=150, zorder=5, label='Pose 5')
    ax.scatter([pose_95[0]], [pose_95[1]], c=COLOR_RED, s=150, zorder=5, label='Pose 95')

    # 3. Landmark
    landmark_pos = (0.5, 2.0)
    ax.scatter([landmark_pos[0]], [landmark_pos[1]], c='orange', marker='*', s=300, zorder=10, label='Landmark')

    # 4. Observations
    ax.plot([pose_5[0], landmark_pos[0]], [pose_5[1], landmark_pos[1]], ':', color=COLOR_BLUE, linewidth=2)
    ax.plot([pose_95[0], landmark_pos[0]], [pose_95[1], landmark_pos[1]], ':', color=COLOR_RED, linewidth=2)

    # 5. Loop Closure Constraint
    ax.add_patch(mpatches.FancyArrowPatch(
        pose_5, pose_95,
        connectionstyle="arc3,rad=-0.3",
        arrowstyle='<->,head_width=10,head_length=10',
        color=COLOR_PURPLE,
        linewidth=3,
        zorder=15,
        label='Loop Closure Constraint'
    ))
    
    # 6. Corrected Trajectory
    x_corr = x - (x[95] - x[5]) * np.linspace(0, 1, 100)**2
    y_corr = y - (y[95] - y[5]) * np.linspace(0, 1, 100)**2
    ax.plot(x_corr, y_corr, '-', color=COLOR_GREEN, linewidth=3, label='Corrected Trajectory')


    ax.set_title('The "Aha!" Moment: How Loop Closure Corrects Drift', fontsize=16, pad=20)
    ax.legend(fontsize=11)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved '{output_path.name}'")


def create_slam_buildup_animation(output_path: Path, slam: RGBDSlam, rgb_paths, depth_paths, num_frames=100):
    """Create an animated GIF showing the SLAM map being built."""
    if output_path.exists():
        print(f"'{output_path.name}' already exists. Skipping generation.")
        return

    print("   Creating SLAM map buildup animation (this is slow)...")
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=60)

    # Pre-process to find axis limits
    print("      Pre-processing to determine map boundaries...")
    slam_temp = RGBDSlam(slam.K, detector_type='ORB')
    pre_poses = slam_temp.process_sequence(rgb_paths, depth_paths, max_frames=num_frames)
    all_landmarks = slam_temp.get_landmark_positions()
    
    if len(all_landmarks) == 0:
        print("      ⚠️ No landmarks found, cannot generate animation.")
        return

    max_range = np.array([all_landmarks[:, 0].max()-all_landmarks[:, 0].min(),
                         all_landmarks[:, 1].max()-all_landmarks[:, 1].min(),
                         all_landmarks[:, 2].max()-all_landmarks[:, 2].min()]).max() / 2.0
    mid_x = (all_landmarks[:, 0].max()+all_landmarks[:, 0].min()) * 0.5
    mid_y = (all_landmarks[:, 1].max()+all_landmarks[:, 1].min()) * 0.5
    mid_z = (all_landmarks[:, 2].max()+all_landmarks[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

    # Initialize plot elements
    landmark_plot = ax.scatter([], [], [], c='lightblue', marker='.', s=1)
    traj_plot, = ax.plot([], [], [], 'purple', linewidth=2)
    start_plot = ax.scatter([], [], [], c='green', marker='o', s=100, edgecolors='black')
    end_plot = ax.scatter([], [], [], c='red', marker='o', s=100, edgecolors='black')

    # Reset SLAM for animation run
    slam_anim = RGBDSlam(slam.K, detector_type='ORB')
    pbar = tqdm(total=num_frames, desc="      Animating frames")

    def update(frame_index):
        rgb = cv2.imread(rgb_paths[frame_index])
        depth = cv2.imread(depth_paths[frame_index], cv2.IMREAD_UNCHANGED)
        slam_anim.process_frame(rgb, depth)

        landmarks = slam_anim.get_landmark_positions()
        if len(landmarks) > 0:
            landmark_plot._offsets3d = (landmarks[:,0], landmarks[:,1], landmarks[:,2])

        poses = np.array(slam_anim.poses)
        traj = poses[:, :3, 3]
        traj_plot.set_data(traj[:, 0], traj[:, 1])
        traj_plot.set_3d_properties(traj[:, 2])

        if frame_index == 0:
            start_plot._offsets3d = ([traj[0,0]], [traj[0,1]], [traj[0,2]])
        
        end_plot._offsets3d = ([traj[-1,0]], [traj[-1,1]], [traj[-1,2]])
        
        ax.set_title(f'Building the Map | Frame: {frame_index+1}/{num_frames} | Landmarks: {len(landmarks):,}', fontsize=14)
        pbar.update(1)
        return landmark_plot, traj_plot, start_plot, end_plot

    ani = FuncAnimation(fig, update, frames=num_frames, blit=False, interval=50)
    writer = PillowWriter(fps=15)
    ani.save(output_path, writer=writer)

    pbar.close()
    plt.close(fig)
    print(f"   ✅ Saved '{output_path.name}'")


# Create output directory
output_dir = Path('chapters/3/images')
output_dir.mkdir(exist_ok=True, parents=True)

print('📸 Generating Chapter 3 Visualizations (SLAM)...')
print('=' * 60)

# Load TUM RGB-D data
print('Loading TUM RGB-D dataset...')
fetcher = TUMDatasetFetcher('data')
sequence = 'fr1/desk'
fetcher.download_sequence(sequence)

timestamps, rgb_paths, depth_paths = fetcher.load_associations(sequence)
gt_timestamps, gt_poses = fetcher.load_poses(sequence)
K = fetcher.get_camera_intrinsics(sequence)

print(f'✅ Loaded {len(rgb_paths)} RGB-D pairs from TUM {sequence}')
print()

# 1. New Visualizations for Storytelling
print('1. Generating narrative visualizations...')
create_loop_closure_diagram(output_dir / 'loop_closure_diagram.png')

# Create a slimmed down SLAM instance for the animation
slam_for_anim = RGBDSlam(K, detector_type='ORB', max_features=500)
create_slam_buildup_animation(
    output_dir / 'slam_buildup_animation.gif',
    slam=slam_for_anim,
    rgb_paths=rgb_paths,
    depth_paths=depth_paths,
    num_frames=150
)
print()

# 2. Raw Sequence GIF (if it doesn't exist)
print('2. Generating raw sequence GIF...')
gif_path = output_dir / 'tum_sequence_raw.gif'
if gif_path.exists():
    print('   ✅ GIF already exists, skipping generation')
else:
    print('   Creating animation (this may take a moment)...')
    num_frames = 200
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    first_img = cv2.imread(rgb_paths[0])
    first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
    im = ax.imshow(first_img)
    
    pbar = tqdm(total=num_frames, desc="   Processing frames")
    
    def update(frame_index):
        img = cv2.imread(rgb_paths[frame_index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im.set_array(img)
        pbar.update(1)
        return [im]
    
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    writer = PillowWriter(fps=20)
    ani.save(gif_path, writer=writer)
    
    pbar.close()
    plt.close(fig)
    print('   ✅ Saved tum_sequence_raw.gif')

# 3. Sample RGB-D pair (if it doesn't exist)
print('3. Creating sample RGB-D pair visualization...')
sample_path = output_dir / 'tum_rgbd_comparison.png'
if not sample_path.exists():
    idx = len(rgb_paths) // 3
    rgb = cv2.imread(rgb_paths[idx])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_paths[idx], cv2.IMREAD_UNCHANGED)
    depth_meters = depth.astype(np.float32) / 5000.0
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image', fontsize=14, pad=10)
    axes[0].axis('off')
    
    depth_vis = depth_meters.copy()
    depth_vis[depth_vis == 0] = np.nan
    im = axes[1].imshow(depth_vis, cmap='plasma', vmin=0, vmax=4.0)
    axes[1].set_title('Depth Map', fontsize=14, pad=10)
    axes[1].axis('off')
    
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth (meters)', fontsize=11)
    
    plt.suptitle('TUM RGB-D Dataset - Synchronized RGB-D Pair', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('   ✅ Saved tum_rgbd_comparison.png')
else:
    print('   ✅ Sample already exists, skipping')

# 4. Run SLAM and generate trajectory comparison
print('4. Running SLAM system...')

slam_cache_file = output_dir / 'slam_poses_200.npy'
landmarks_cache_file = output_dir / 'slam_landmarks_200.npy'
n_frames = 200

if slam_cache_file.exists() and landmarks_cache_file.exists():
    print(f'   Loading {n_frames} SLAM poses from cache...')
    slam_poses = np.load(slam_cache_file)
    landmarks = np.load(landmarks_cache_file)
    print('   ✅ Loaded SLAM results from cache')
else:
    print(f'   Processing {n_frames} frames with SLAM (this will take a few minutes)...')
    slam = RGBDSlam(
        camera_matrix=K,
        detector_type='ORB',
        max_features=1000,
        loop_closure_threshold=0.5
    )
    
    slam_poses = slam.process_sequence(rgb_paths, depth_paths, max_frames=n_frames)
    landmarks = slam.get_landmark_positions()
    
    # Save to cache
    np.save(slam_cache_file, slam_poses)
    np.save(landmarks_cache_file, landmarks)
    
    stats = slam.get_statistics()
    print(f'   ✅ SLAM complete:')
    print(f'      - Poses: {len(slam_poses)}')
    print(f'      - Landmarks: {len(landmarks)}')
    print(f'      - Loop closures: {stats["loop_closures"]}')

# Match ground truth poses to our timestamps
print('5. Generating trajectory visualizations...')
gt_poses_matched = []
for i in range(len(slam_poses)):
    ts = timestamps[i]
    idx = np.argmin(np.abs(gt_timestamps - ts))
    gt_poses_matched.append(gt_poses[idx])
gt_poses_matched = np.array(gt_poses_matched)

# Plot SLAM trajectory comparison
plot_top_down_trajectory(
    slam_poses,
    gt_poses_matched,
    title=f'SLAM Trajectory (First {n_frames} frames)',
    save_path=output_dir / f'slam_trajectory_{n_frames}.png'
)
print(f'   ✅ Saved slam_trajectory_{n_frames}.png')

# 6. 3D Landmark Map Visualization
print('6. Generating 3D landmark map...')
map_path = output_dir / f'slam_map_3d_{n_frames}.png'

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot landmarks (subsample for clarity)
if len(landmarks) > 5000:
    indices = np.random.choice(len(landmarks), 5000, replace=False)
    landmarks_plot = landmarks[indices]
else:
    landmarks_plot = landmarks

ax.scatter(landmarks_plot[:, 0], landmarks_plot[:, 1], landmarks_plot[:, 2], 
           c='lightblue', marker='.', s=1, alpha=0.3, label='Landmarks')

# Plot trajectory
traj = slam_poses[:, :3, 3]
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
        'purple', linewidth=3, label='Camera Path', alpha=0.8)

# Mark start and end
ax.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], 
           c='green', marker='o', s=200, label='Start', edgecolors='black', linewidths=2)
ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], 
           c='red', marker='o', s=200, label='End', edgecolors='black', linewidths=2)

ax.set_xlabel('X (meters)', fontsize=12, labelpad=10)
ax.set_ylabel('Y (meters)', fontsize=12, labelpad=10)
ax.set_zlabel('Z (meters)', fontsize=12, labelpad=10)
ax.set_title(f'SLAM Map: {len(landmarks):,} Landmarks', fontsize=16, pad=20)
ax.legend(loc='upper right', fontsize=11)

# Set viewing angle
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig(map_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'   ✅ Saved slam_map_3d_{n_frames}.png')

# 7. Compute and save statistics
print('7. Computing error statistics...')
errors = compute_trajectory_error(slam_poses, gt_poses_matched)

stats_file = output_dir / 'statistics.txt'
with open(stats_file, 'w') as f:
    f.write('Chapter 3: SLAM Performance Statistics\n')
    f.write('=' * 50 + '\n\n')
    f.write(f'Dataset: TUM RGB-D {sequence}\n')
    f.write(f'Frames processed: {len(slam_poses)}\n')
    f.write(f'Landmarks in map: {len(landmarks):,}\n\n')
    f.write('Trajectory Errors:\n')
    f.write(f'  Mean position error: {errors["mean_position_error"]:.3f} m\n')
    f.write(f'  RMS position error: {errors["rms_position_error"]:.3f} m\n')
    f.write(f'  Mean RPE translation: {errors["mean_rpe_trans"]:.3f} m\n')
    f.write(f'  Mean RPE rotation: {np.rad2deg(errors["mean_rpe_rot"]):.3f}°\n\n')
    f.write('Source: TUM RGB-D Benchmark\n')
    f.write('   https://cvg.cit.tum.de/data/datasets/rgbd-dataset\n')

print('   ✅ Saved statistics.txt')

print()
print('=' * 60)
print('✅ All Chapter 3 visualizations generated successfully!')
print()
print('Generated files:')
print(f'   - loop_closure_diagram.png')
print(f'   - slam_buildup_animation.gif')
print(f'   - tum_sequence_raw.gif')
print(f'   - tum_rgbd_comparison.png')
print(f'   - slam_trajectory_{n_frames}.png')
print(f'   - slam_map_3d_{n_frames}.png')
print(f'   - statistics.txt')
print()
print('   Source: TUM RGB-D Benchmark')
print('   https://cvg.cit.tum.de/data/datasets/rgbd-dataset')
