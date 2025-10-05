"""
Generate side-by-side comparison visualizations between baseline and improved models.
Shows the improvement achieved through architectural enhancements.
"""
import sys
sys.path.append('../../..')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from matplotlib.patches import Circle
import matplotlib.patches as patches

def load_model_results():
    """Load results from both models."""
    # Load baseline results
    baseline_path = '../artifacts/flight_evaluation_results.pkl'
    # Try relative path first, then absolute
    if not os.path.exists(baseline_path):
        baseline_path = 'artifacts/flight_evaluation_results.pkl'
    if not os.path.exists(baseline_path):
        print(f"❌ Baseline results not found: {baseline_path}")
        return None, None

    with open(baseline_path, 'rb') as f:
        baseline_results = pickle.load(f)

    # Load improved results
    improved_path = '../artifacts/improved_model_results.pkl'
    if not os.path.exists(improved_path):
        improved_path = 'artifacts/improved_model_results.pkl'
    if not os.path.exists(improved_path):
        print(f"❌ Improved results not found: {improved_path}")
        return None, None

    with open(improved_path, 'rb') as f:
        improved_results = pickle.load(f)

    return baseline_results, improved_results

def create_trajectory_comparison():
    """Create side-by-side trajectory comparison visualization."""
    print("--- Creating Trajectory Comparison ---")

    baseline_results, improved_results = load_model_results()
    if baseline_results is None or improved_results is None:
        return

    # Get trajectory data
    gt_coords = baseline_results['ground_truth']  # Same for both
    baseline_pred = baseline_results['predictions']
    baseline_errors = baseline_results['errors']
    improved_pred = improved_results['predictions']
    improved_errors = improved_results['errors']

    # Configuration
    map_path = '../../../data/boneyard/davis_monthan_stitched_map.jpg'
    if not os.path.exists(map_path):
        map_path = '../../data/boneyard/davis_monthan_stitched_map.jpg'
    full_map = Image.open(map_path).convert('RGB')
    map_width, map_height = full_map.size

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 9))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.08)

    # Define zoom bounds (same for both)
    buffer = 800
    min_x = min(np.min(gt_coords[:, 0]), np.min(baseline_pred[:, 0]), np.min(improved_pred[:, 0])) - buffer
    max_x = max(np.max(gt_coords[:, 0]), np.max(baseline_pred[:, 0]), np.max(improved_pred[:, 0])) + buffer
    min_y = min(np.min(gt_coords[:, 1]), np.max(baseline_pred[:, 1]), np.min(improved_pred[:, 1])) - buffer
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
        (ax1, baseline_pred, baseline_errors, 'Baseline CorridorCNN', 'Baseline'),
        (ax2, improved_pred, improved_errors, 'Improved CoordConvPoseNet', 'Improved')
    ]:
        # Crop and display map
        cropped_map = full_map.crop(crop_bounds)
        ax.imshow(cropped_map, extent=[min_x, max_x, max_y, min_y], alpha=0.8)

        # Plot ground truth path
        ax.plot(gt_coords[:, 0], gt_coords[:, 1], 'g-', linewidth=4,
                label='Ground Truth Path', alpha=0.9)

        # Color scheme for errors
        colors = plt.cm.RdYlGn_r(errors / np.max([np.max(baseline_errors), np.max(improved_errors)]))

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
        max_error = np.max(errors)

        stats_text = f"""PERFORMANCE METRICS

Mean Error: {mean_error:.1f} px
Median Error: {median_error:.1f} px
Max Error: {max_error:.1f} px

Predictions < 100px: {np.sum(errors < 100)}/150
Accuracy Rate: {(np.sum(errors < 100)/150)*100:.1f}%"""

        color = 'lightgreen' if mean_error < 100 else 'lightyellow' if mean_error < 150 else 'lightcoral'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.95, edgecolor='black'))

        # Add title
        ax.text(0.5, 0.04, title,
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                ha='center', va='bottom', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8, edgecolor='white'))

    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                              norm=plt.Normalize(vmin=0, vmax=np.max([np.max(baseline_errors), np.max(improved_errors)])))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal',
                       fraction=0.04, pad=0.02, aspect=50)
    cbar.set_label('Prediction Error (pixels)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add main title
    improvement_pct = ((np.mean(baseline_errors) - np.mean(improved_errors)) / np.mean(baseline_errors)) * 100
    fig.suptitle(f'Model Architecture Comparison: {improvement_pct:.1f}% Error Reduction',
                fontsize=16, fontweight='bold', y=0.96)

    # Save comparison
    output_path = '../images/model_trajectory_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ Trajectory comparison saved to {output_path}")
    plt.close()

    return improvement_pct

def create_error_distribution_comparison():
    """Create error distribution comparison chart."""
    print("--- Creating Error Distribution Comparison ---")

    baseline_results, improved_results = load_model_results()
    if baseline_results is None or improved_results is None:
        return

    baseline_errors = baseline_results['errors']
    improved_errors = improved_results['errors']

    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.12, wspace=0.25)

    # Histogram comparison
    bins = np.linspace(0, max(np.max(baseline_errors), np.max(improved_errors)), 30)

    ax1.hist(baseline_errors, bins=bins, alpha=0.7, color='lightcoral',
             label=f'Baseline (Mean: {np.mean(baseline_errors):.1f}px)', edgecolor='black')
    ax1.hist(improved_errors, bins=bins, alpha=0.7, color='lightgreen',
             label=f'Improved (Mean: {np.mean(improved_errors):.1f}px)', edgecolor='black')

    ax1.axvline(np.mean(baseline_errors), color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.axvline(np.mean(improved_errors), color='green', linestyle='--', linewidth=2, alpha=0.8)

    ax1.set_xlabel('Prediction Error (pixels)', fontweight='bold')
    ax1.set_ylabel('Number of Predictions', fontweight='bold')
    ax1.set_title('Error Distribution Comparison', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Box plot comparison
    data = [baseline_errors, improved_errors]
    box_plot = ax2.boxplot(data, labels=['Baseline\nCorridorCNN', 'Improved\nCoordConvPoseNet'],
                          patch_artist=True, widths=0.6)

    # Color boxes
    colors = ['lightcoral', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Prediction Error (pixels)', fontweight='bold')
    ax2.set_title('Error Distribution Statistics', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add improvement percentage
    improvement_pct = ((np.mean(baseline_errors) - np.mean(improved_errors)) / np.mean(baseline_errors)) * 100
    fig.suptitle(f'Navigation Accuracy Improvement: {improvement_pct:.1f}% Error Reduction',
                fontsize=16, fontweight='bold')

    # Save
    output_path = '../images/model_error_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ Error distribution comparison saved to {output_path}")
    plt.close()

    return improvement_pct

def create_performance_summary():
    """Create a comprehensive performance summary table."""
    print("--- Creating Performance Summary ---")

    baseline_results, improved_results = load_model_results()
    if baseline_results is None or improved_results is None:
        return

    baseline_errors = baseline_results['errors']
    improved_errors = improved_results['errors']

    # Calculate metrics
    metrics = {
        'Mean Error (px)': [np.mean(baseline_errors), np.mean(improved_errors)],
        'Median Error (px)': [np.median(baseline_errors), np.median(improved_errors)],
        'Max Error (px)': [np.max(baseline_errors), np.max(improved_errors)],
        'Std Dev (px)': [np.std(baseline_errors), np.std(improved_errors)],
        'Predictions < 50px': [np.sum(baseline_errors < 50), np.sum(improved_errors < 50)],
        'Predictions < 100px': [np.sum(baseline_errors < 100), np.sum(improved_errors < 100)],
        'Accuracy Rate (%)': [(np.sum(baseline_errors < 100)/150)*100, (np.sum(improved_errors < 100)/150)*100]
    }

    # Create comparison table visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data for table
    headers = ['Metric', 'Baseline CorridorCNN', 'Improved CoordConvPoseNet', 'Improvement']
    table_data = []

    for metric, values in metrics.items():
        baseline_val = values[0]
        improved_val = values[1]

        if 'Rate' in metric or 'Predictions' in metric:
            if 'Rate' in metric:
                improvement = f"{improved_val - baseline_val:.1f}%"
                baseline_str = f"{baseline_val:.1f}%"
                improved_str = f"{improved_val:.1f}%"
            else:
                improvement = f"+{improved_val - baseline_val}"
                baseline_str = f"{int(baseline_val)}"
                improved_str = f"{int(improved_val)}"
        else:
            improvement_pct = ((baseline_val - improved_val) / baseline_val) * 100
            improvement = f"{improvement_pct:.1f}% better"
            baseline_str = f"{baseline_val:.1f}"
            improved_str = f"{improved_val:.1f}"

        table_data.append([metric, baseline_str, improved_str, improvement])

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code improvements
    for i in range(1, len(table_data) + 1):
        if 'better' in table_data[i-1][3] or '+' in table_data[i-1][3]:
            table[(i, 3)].set_facecolor('lightgreen')
        table[(i, 3)].set_text_props(weight='bold')

    ax.set_title('Navigation Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)

    # Save
    output_path = '../images/model_performance_summary.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"✅ Performance summary saved to {output_path}")
    plt.close()

def main():
    """Generate all comparison visualizations."""
    print("=== Generating Model Comparison Visualizations ===")

    # Create all comparison visualizations
    improvement_pct = create_trajectory_comparison()
    create_error_distribution_comparison()
    create_performance_summary()

    print(f"\n✅ All comparison visualizations generated!")
    print(f"Overall improvement: {improvement_pct:.1f}% error reduction")
    print("=== Model Comparison Complete ===")

if __name__ == "__main__":
    main()