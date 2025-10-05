"""
Create stacked vertical trajectory comparison.
Original on top, improved on bottom, full frame with no padding.
"""
import sys
sys.path.append('../../..')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from matplotlib.patches import Circle

def load_model_results():
    """Load results from both models."""
    # Load baseline results
    baseline_path = '../artifacts/flight_evaluation_results.pkl'
    if not os.path.exists(baseline_path):
        baseline_path = 'artifacts/flight_evaluation_results.pkl'

    with open(baseline_path, 'rb') as f:
        baseline_results = pickle.load(f)

    # Load improved results
    improved_path = '../artifacts/improved_model_results.pkl'
    if not os.path.exists(improved_path):
        improved_path = 'artifacts/improved_model_results.pkl'

    with open(improved_path, 'rb') as f:
        improved_results = pickle.load(f)

    return baseline_results, improved_results

def create_stacked_trajectory_comparison():
    """Create stacked vertical trajectory comparison."""
    print("--- Creating Stacked Trajectory Comparison ---")

    baseline_results, improved_results = load_model_results()

    # Get trajectory data
    gt_coords = baseline_results['ground_truth']  # Same for both
    baseline_pred = baseline_results['predictions']
    baseline_errors = baseline_results['errors']

    # Handle improved results format
    if 'ground_truth' in improved_results:
        improved_pred = improved_results['predictions']
        improved_errors = improved_results['errors']
    else:
        # Convert from normalized format
        map_path = '../../../data/boneyard/davis_monthan_stitched_map.jpg'
        full_map = Image.open(map_path).convert('RGB')
        map_width, map_height = full_map.size

        targets_norm = improved_results['targets']
        pred_norm = improved_results['predictions']

        improved_pred = pred_norm * np.array([map_width, map_height])
        improved_errors = improved_results['errors']

    # Configuration
    map_path = '../../../data/boneyard/davis_monthan_stitched_map.jpg'
    full_map = Image.open(map_path).convert('RGB')
    map_width, map_height = full_map.size

    # Create figure with two vertically stacked subplots, no spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0)

    # Define zoom bounds (same for both)
    buffer = 800
    min_x = min(np.min(gt_coords[:, 0]), np.min(baseline_pred[:, 0]), np.min(improved_pred[:, 0])) - buffer
    max_x = max(np.max(gt_coords[:, 0]), np.max(baseline_pred[:, 0]), np.max(improved_pred[:, 0])) + buffer
    min_y = min(np.min(gt_coords[:, 1]), np.min(baseline_pred[:, 1]), np.min(improved_pred[:, 1])) - buffer
    max_y = max(np.max(gt_coords[:, 1]), np.max(baseline_pred[:, 1]), np.max(improved_pred[:, 1])) + buffer

    # Force 16:9 aspect ratio for each subplot
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

    crop_bounds = (int(min_x), int(min_y), int(max_x), int(max_y))

    for ax, pred_coords, errors, title, model_name in [
        (ax1, baseline_pred, baseline_errors, 'Baseline CorridorCNN Performance', 'Baseline'),
        (ax2, improved_pred, improved_errors, 'CoordConv Model Performance', 'Improved')
    ]:
        # Crop and display map
        cropped_map = full_map.crop(crop_bounds)
        ax.imshow(cropped_map, extent=[min_x, max_x, max_y, min_y], alpha=0.8)

        # Plot ground truth path
        ax.plot(gt_coords[:, 0], gt_coords[:, 1], 'g-', linewidth=4,
                label='Ground Truth Path', alpha=0.9)

        # Color scheme for errors
        max_error = max(np.max(baseline_errors), np.max(improved_errors))
        colors = plt.cm.RdYlGn_r(errors / max_error)

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

        # Add performance statistics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        max_error_single = np.max(errors)

        stats_text = f"""PERFORMANCE METRICS

Mean Error: {mean_error:.1f} px
Median Error: {median_error:.1f} px
Max Error: {max_error_single:.1f} px

Predictions < 100px: {np.sum(errors < 100)}/150
Predictions < 200px: {np.sum(errors < 200)}/150
Success Rate: {(np.sum(errors < 200)/150)*100:.1f}%"""

        color = 'lightgreen' if mean_error < 150 else 'lightyellow' if mean_error < 250 else 'lightcoral'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.95, edgecolor='black'))

        # Add title
        ax.text(0.5, 0.04, title,
                transform=ax.transAxes, fontsize=16, fontweight='bold',
                ha='center', va='bottom', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8, edgecolor='white'))

    # Add shared colorbar at bottom
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                              norm=plt.Normalize(vmin=0, vmax=max(np.max(baseline_errors), np.max(improved_errors))))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal',
                       fraction=0.02, pad=0.01, aspect=100)
    cbar.set_label('Prediction Error (pixels)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    # Save comparison
    output_path = '../images/stacked_trajectory_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"✅ Stacked trajectory comparison saved to {output_path}")
    plt.close()

    # Print comparison stats
    baseline_mean = np.mean(baseline_errors)
    improved_mean = np.mean(improved_errors)
    print(f"\nComparison Summary:")
    print(f"Baseline mean error: {baseline_mean:.1f}px")
    print(f"CoordConv mean error: {improved_mean:.1f}px")
    print(f"Performance change: {((improved_mean - baseline_mean) / baseline_mean) * 100:.1f}%")

if __name__ == "__main__":
    create_stacked_trajectory_comparison()