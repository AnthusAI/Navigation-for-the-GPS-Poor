"""
Create the full trajectory visualization for the improved CoordConv model.
This is equivalent to the baseline's predicted_vs_ground_truth_trajectory.png
showing all predictions and errors over the complete flight path.
"""
import sys
sys.path.append('../../..')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from matplotlib.patches import Circle

def create_improved_trajectory_full():
    """Create full trajectory plot for improved CoordConv model."""
    print("--- Creating Improved Model Full Trajectory ---")

    # Load improved model results
    results_path = '../artifacts/improved_model_results.pkl'
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    # Handle different result formats
    if 'ground_truth' in results:
        gt_coords = results['ground_truth']
        pred_coords = results['predictions']
        errors = results['errors']
    else:
        # Convert from normalized format
        map_path = '../../../data/boneyard/davis_monthan_stitched_map.jpg'
        full_map = Image.open(map_path).convert('RGB')
        map_width, map_height = full_map.size

        targets_norm = results['targets']
        pred_norm = results['predictions']

        # Denormalize to pixel coordinates
        gt_coords = targets_norm * np.array([map_width, map_height])
        pred_coords = pred_norm * np.array([map_width, map_height])
        errors = results['errors']

    # Configuration
    map_path = '../../../data/boneyard/davis_monthan_stitched_map.jpg'
    full_map = Image.open(map_path).convert('RGB')
    map_width, map_height = full_map.size

    # Create figure with 16:9 aspect ratio
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Define zoom bounds around the flight path
    buffer = 800
    min_x = min(np.min(gt_coords[:, 0]), np.min(pred_coords[:, 0])) - buffer
    max_x = max(np.max(gt_coords[:, 0]), np.max(pred_coords[:, 0])) + buffer
    min_y = min(np.min(gt_coords[:, 1]), np.min(pred_coords[:, 1])) - buffer
    max_y = max(np.max(gt_coords[:, 1]), np.max(pred_coords[:, 1])) + buffer

    # Force 16:9 aspect ratio
    current_width = max_x - min_x
    current_height = max_y - min_y
    current_aspect = current_width / current_height
    target_aspect = 16.0 / 9.0

    if current_aspect > target_aspect:
        # Too wide - increase height
        new_height = current_width / target_aspect
        height_increase = (new_height - current_height) / 2
        min_y -= height_increase
        max_y += height_increase
    else:
        # Too tall - increase width
        new_width = current_height * target_aspect
        width_increase = (new_width - current_width) / 2
        min_x -= width_increase
        max_x += width_increase

    # Crop and display map
    crop_bounds = (int(min_x), int(min_y), int(max_x), int(max_y))
    cropped_map = full_map.crop(crop_bounds)
    ax.imshow(cropped_map, extent=[min_x, max_x, max_y, min_y], alpha=0.8)

    # Plot ground truth path
    ax.plot(gt_coords[:, 0], gt_coords[:, 1], 'g-', linewidth=4,
            label='Ground Truth Path', alpha=0.9)

    # Color scheme for errors
    colors = plt.cm.RdYlGn_r(errors / np.max(errors))

    # Plot each prediction as a circle with radius = actual error distance
    for i in range(len(pred_coords)):
        circle = Circle((pred_coords[i, 0], pred_coords[i, 1]),
                      radius=errors[i], facecolor=colors[i],
                      edgecolor='white', alpha=0.6, linewidth=1)
        ax.add_patch(circle)

    # Mark start and end points
    ax.plot(gt_coords[0, 0], gt_coords[0, 1], 'o', color='lime', markersize=20,
            markeredgecolor='black', markeredgewidth=3, label='Start Point')
    ax.plot(gt_coords[-1, 0], gt_coords[-1, 1], 's', color='red', markersize=20,
            markeredgecolor='black', markeredgewidth=3, label='End Point')

    # Set limits and styling
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)  # Flip Y for image coordinates
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=0, vmax=np.max(errors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                       fraction=0.04, pad=0.02, aspect=50)
    cbar.set_label('Prediction Error (pixels)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add title with consistent size (fontsize=14)
    ax.text(0.5, 0.04, 'CoordConv Terrain Navigation: Individual Position Predictions',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='bottom', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8, edgecolor='white'))

    # Save
    output_path = '../images/improved_predicted_vs_ground_truth_trajectory.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"✅ Improved full trajectory saved to {output_path}")
    plt.close()

    # Print summary
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    print(f"\nCoordConv Model Performance Summary:")
    print(f"Mean error: {mean_error:.1f}px")
    print(f"Median error: {median_error:.1f}px")
    print(f"Max error: {max_error:.1f}px")
    print(f"Predictions within 200px: {np.sum(errors < 200)}/150 ({(np.sum(errors < 200)/150)*100:.1f}%)")

if __name__ == "__main__":
    create_improved_trajectory_full()