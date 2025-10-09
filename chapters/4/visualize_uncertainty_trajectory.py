#!/usr/bin/env python3
"""
Create flight path visualization WITH uncertainty circles.
Like the standard trajectory but showing confidence for each prediction.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from pathlib import Path
import pickle

sys.path.append(str(Path(__file__).parent))
from navigation.terrain_window import TerrainWindow


def create_trajectory_with_uncertainty_circles(eval_results_path: str = "artifacts/uncertainty_evaluation_results.pkl",
                                               save_path: str = "images/navigation_uncertainty_trajectory.png"):
    """
    Create the standard flight path visualization with uncertainty circles.

    Shows:
    - Satellite background
    - Green circles: ground truth
    - Red X: predictions
    - Blue circles: uncertainty bounds (1σ radius)
    - Gray lines: errors
    """
    print("🎨 Creating Flight Path with Uncertainty Visualization")
    print("=" * 60)

    # Load evaluation results
    with open(eval_results_path, 'rb') as f:
        eval_results = pickle.load(f)

    # Load satellite map
    terrain_window = TerrainWindow()
    satellite_map = terrain_window.stitched_map

    # Get data in pixels
    true_coords = eval_results['trajectory_coords'] * 7500
    predictions = eval_results['predictions'] * 7500
    uncertainties_m = eval_results['uncertainties']
    errors_m = eval_results['errors']

    # Convert uncertainties to pixels
    uncertainties_px = uncertainties_m / 10  # meters to pixels (10m/pixel)

    # Calculate bounds for zooming
    all_points = np.vstack([true_coords, predictions])
    min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
    min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()

    # Add padding
    x_range = max_x - min_x
    y_range = max_y - min_y
    padding_x = x_range * 0.2
    padding_y = y_range * 0.2

    zoom_min_x = max(0, min_x - padding_x)
    zoom_max_x = min(7500, max_x + padding_x)
    zoom_min_y = max(0, min_y - padding_y)
    zoom_max_y = min(7500, max_y + padding_y)

    # Adjust for 16:9 aspect ratio
    current_width = zoom_max_x - zoom_min_x
    current_height = zoom_max_y - zoom_min_y
    current_aspect = current_width / current_height
    target_aspect = 16.0 / 9.0

    if current_aspect > target_aspect:
        new_height = current_width / target_aspect
        height_expansion = (new_height - current_height) / 2
        zoom_min_y = max(0, zoom_min_y - height_expansion)
        zoom_max_y = min(7500, zoom_max_y + height_expansion)
    else:
        new_width = current_height * target_aspect
        width_expansion = (new_width - current_width) / 2
        zoom_min_x = max(0, zoom_min_x - width_expansion)
        zoom_max_x = min(7500, zoom_max_x + width_expansion)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Show satellite background
    left, top = int(zoom_min_x), int(zoom_min_y)
    right, bottom = int(zoom_max_x), int(zoom_max_y)
    cropped_map = satellite_map[top:bottom, left:right]
    ax.imshow(cropped_map, extent=[zoom_min_x, zoom_max_x, zoom_max_y, zoom_min_y], alpha=0.9)

    # Draw uncertainty circles FIRST (so they're behind other elements)
    for i, (pred, unc_px) in enumerate(zip(predictions, uncertainties_px)):
        # Uncertainty circle (1 standard deviation)
        circle = Circle(
            pred,
            radius=unc_px,
            facecolor='blue',
            edgecolor='darkblue',
            alpha=0.15,
            linewidth=1.5,
            linestyle='--'
        )
        ax.add_patch(circle)

    # Draw error lines
    for true_pt, pred_pt in zip(true_coords, predictions):
        ax.plot([true_pt[0], pred_pt[0]], [true_pt[1], pred_pt[1]],
                color='gray', linewidth=1.5, alpha=0.6, zorder=5)

    # Draw ground truth path
    ax.plot(true_coords[:, 0], true_coords[:, 1], 'white', linewidth=8, alpha=0.9, zorder=6)
    ax.plot(true_coords[:, 0], true_coords[:, 1], 'green', linewidth=6, alpha=1.0, zorder=7)

    # Draw ground truth points
    ax.scatter(true_coords[:, 0], true_coords[:, 1],
              c='lime', s=150, marker='o',
              edgecolors='darkgreen', linewidths=2,
              alpha=0.9, zorder=10, label='Ground Truth')

    # Draw predictions
    ax.scatter(predictions[:, 0], predictions[:, 1],
              c='red', s=150, marker='x',
              linewidths=3,
              alpha=0.9, zorder=11, label='Predictions')

    # Add start/end markers
    ax.scatter(true_coords[0, 0], true_coords[0, 1],
              c='lime', s=400, marker='o',
              edgecolors='black', linewidths=3, zorder=15)
    ax.scatter(true_coords[-1, 0], true_coords[-1, 1],
              c='red', s=400, marker='s',
              edgecolors='black', linewidths=3, zorder=15)

    # Set bounds
    ax.set_xlim(zoom_min_x, zoom_max_x)
    ax.set_ylim(zoom_max_y, zoom_min_y)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=4, label='Flight Path'),
        plt.Line2D([0], [0], marker='o', color='lime', markerfacecolor='lime',
                  markersize=10, markeredgecolor='darkgreen', markeredgewidth=2,
                  linestyle='None', label='Ground Truth'),
        plt.Line2D([0], [0], marker='x', color='red', markerfacecolor='red',
                  markersize=10, linewidth=3,
                  linestyle='None', label='Predictions'),
        Patch(facecolor='blue', edgecolor='darkblue', alpha=0.3,
              linestyle='--', label='Uncertainty (1σ)'),
        plt.Line2D([0], [0], color='gray', lw=2, alpha=0.6, label='Error Lines')
    ]

    legend = ax.legend(handles=legend_elements, loc='upper right',
                      bbox_to_anchor=(0.98, 0.98), fontsize=11,
                      framealpha=0.9, facecolor='black', edgecolor='white')
    plt.setp(legend.get_texts(), color='white')

    # Title with statistics
    mean_error = np.mean(errors_m)
    mean_uncertainty = np.mean(uncertainties_m)
    within_1sigma = np.mean(errors_m < uncertainties_m)

    title_text = (f'Flight Path Navigation with Uncertainty\n'
                 f'Mean Error: {mean_error:.0f}m  |  '
                 f'Mean Uncertainty: {mean_uncertainty:.0f}m  |  '
                 f'{within_1sigma:.0%} within 1σ  |  '
                 f'{len(predictions)} predictions')

    ax.text(0.5, 0.04, title_text,
            transform=ax.transAxes, fontsize=13, fontweight='bold',
            ha='center', va='bottom', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='white'))

    # Calibration indicator
    if 0.60 <= within_1sigma <= 0.75:
        calibration_text = "✅ Well Calibrated"
        cal_color = 'lightgreen'
    elif 0.50 <= within_1sigma < 0.60 or 0.75 < within_1sigma <= 0.85:
        calibration_text = "⚠️ Moderately Calibrated"
        cal_color = 'yellow'
    else:
        calibration_text = "❌ Poorly Calibrated"
        cal_color = 'lightcoral'

    ax.text(0.02, 0.15, f'Calibration: {calibration_text}\n(Expect ~68% within 1σ)',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=cal_color, alpha=0.7, edgecolor='white'))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()

    print(f"✅ Saved: {save_path}")
    print(f"  Mean error: {mean_error:.1f}m")
    print(f"  Mean uncertainty: {mean_uncertainty:.1f}m")
    print(f"  Calibration: {within_1sigma:.1%} within 1σ")


if __name__ == "__main__":
    create_trajectory_with_uncertainty_circles()
