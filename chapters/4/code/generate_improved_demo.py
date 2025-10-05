"""
Generate CNN demonstration visualization for the improved model using existing results.
This creates the same style visualizations as the baseline model for comparison.
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

def load_improved_results():
    """Load the improved model evaluation results."""
    results_path = '../artifacts/improved_model_results.pkl'
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        print("Please run train_improved_model.py first")
        return None

    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results

def create_improved_cnn_demonstration():
    """Create educational CNN demonstration using improved model results."""
    print("--- Creating Improved CNN Demonstration ---")

    # Load improved model results
    results = load_improved_results()
    if results is None:
        return

    # Handle different result formats
    if 'ground_truth' in results:
        gt_coords = results['ground_truth']
        pred_coords = results['predictions']
        errors = results['errors']
    else:
        # Convert from different format (targets are normalized)
        map_path = '../../../data/boneyard/davis_monthan_stitched_map.jpg'
        from PIL import Image
        full_map = Image.open(map_path).convert('RGB')
        map_width, map_height = full_map.size

        targets_norm = results['targets']
        pred_norm = results['predictions']

        # Denormalize to pixel coordinates
        gt_coords = targets_norm * np.array([map_width, map_height])
        pred_coords = pred_norm * np.array([map_width, map_height])
        errors = results['errors']

    # Configuration (same as original)
    map_path = '../../../data/boneyard/davis_monthan_stitched_map.jpg'
    tile_size = (1200, 675)
    zoom_factor = 4

    # Use the SAME example as baseline model for fair comparison
    example_idx = 58  # Same frame as baseline model demo

    gt_pos = gt_coords[example_idx]
    pred_pos = pred_coords[example_idx]
    error = errors[example_idx]

    print(f"Using frame {example_idx} with error: {error:.1f} pixels")

    # Create the input image that went into the CNN
    full_map = Image.open(map_path).convert('RGB')
    map_width, map_height = full_map.size

    # Calculate crop area (same as in evaluation)
    crop_width = tile_size[0] // zoom_factor
    crop_height = tile_size[1] // zoom_factor

    left = int(gt_pos[0] - crop_width / 2)
    top = int(gt_pos[1] - crop_height / 2)
    right = left + crop_width
    bottom = top + crop_height

    # Crop and resize to model input size
    cropped = full_map.crop((left, top, right, bottom))
    input_image = cropped.resize(tile_size, Image.LANCZOS)

    # Calculate normalized coordinates for display
    pred_norm_x = pred_pos[0] / map_width
    pred_norm_y = pred_pos[1] / map_height

    # Calculate input image aspect ratio
    input_aspect = tile_size[0] / tile_size[1]  # 1200/675 = 1.78 (close to 16:9)

    # 1. IMPROVED INPUT IMAGE VISUALIZATION (same style as original)
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 9))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax1.imshow(input_image)

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # Add prediction output as overlay in bottom right
    output_text = f"""IMPROVED MODEL OUTPUT

Frame: {example_idx}/150

Predicted Position:
x = {pred_norm_x:.6f}
y = {pred_norm_y:.6f}

Map Coordinates:
({pred_pos[0]:.0f}, {pred_pos[1]:.0f})

Classification: {'EXCELLENT' if error < 50 else 'GOOD' if error < 100 else 'FAIR' if error < 200 else 'POOR'}"""

    color = 'lightgreen' if error < 50 else 'lightyellow' if error < 100 else 'lightcoral' if error < 200 else 'lightpink'
    ax1.text(0.98, 0.02, output_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='bottom', horizontalalignment='right', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.95, edgecolor='black'))

    # Add title with CONSISTENT size (fontsize=14)
    ax1.text(0.5, 0.04, 'Improved CNN Input: What the CoordConv Model Sees',
            transform=ax1.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='bottom', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8, edgecolor='white'))

    # Save input visualization
    input_output_path = '../images/improved_cnn_input_demo.png'
    plt.savefig(input_output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"✅ Improved CNN input demo saved to {input_output_path}")
    plt.close()

    # 2. IMPROVED ERROR ANALYSIS VISUALIZATION (same style as original)
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 9))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Show same tight zoom as original
    context_size = 400  # Same as original
    context_left = max(0, int(gt_pos[0] - context_size/2))
    context_top = max(0, int(gt_pos[1] - context_size/2))
    context_right = min(map_width, context_left + context_size)
    context_bottom = min(map_height, context_top + context_size)

    # Force 16:9 aspect ratio by adjusting the crop area
    current_width = context_right - context_left
    current_height = context_bottom - context_top
    current_aspect = current_width / current_height
    target_aspect = 16.0 / 9.0

    if current_aspect > target_aspect:
        # Too wide - reduce height
        new_height = current_width / target_aspect
        height_reduction = (new_height - current_height) / 2
        context_top = max(0, context_top - height_reduction)
        context_bottom = min(map_height, context_bottom + height_reduction)
    else:
        # Too tall - reduce width
        new_width = current_height * target_aspect
        width_reduction = (new_width - current_width) / 2
        context_left = max(0, context_left - width_reduction)
        context_right = min(map_width, context_right + width_reduction)

    context_map = full_map.crop((int(context_left), int(context_top), int(context_right), int(context_bottom)))
    ax2.imshow(context_map, extent=[context_left, context_right, context_bottom, context_top])

    # Show the input crop area
    rect = patches.Rectangle((left, top), right-left, bottom-top,
                           linewidth=4, edgecolor='blue', facecolor='none',
                           linestyle='--', alpha=0.8)
    ax2.add_patch(rect)

    # Show ground truth and prediction
    ax2.plot(gt_pos[0], gt_pos[1], 'o', color='lime', markersize=25,
             markeredgecolor='black', markeredgewidth=4)
    ax2.plot(pred_pos[0], pred_pos[1], 's', color='red', markersize=25,
             markeredgecolor='black', markeredgewidth=4)

    # Draw error circle
    circle = Circle((pred_pos[0], pred_pos[1]), error, fill=False,
                   edgecolor='red', linewidth=4, linestyle='--', alpha=0.9)
    ax2.add_patch(circle)

    ax2.set_xlim(context_left, context_right)
    ax2.set_ylim(context_bottom, context_top)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='lime', markerfacecolor='lime',
                  markersize=12, markeredgecolor='black', markeredgewidth=2,
                  linestyle='None', label='Ground Truth'),
        plt.Line2D([0], [0], marker='s', color='red', markerfacecolor='red',
                  markersize=12, markeredgecolor='black', markeredgewidth=2,
                  linestyle='None', label='CoordConv Prediction'),
        patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='blue',
                         facecolor='none', linestyle='--', label='CNN Input Area'),
        patches.Circle((0, 0), 1, fill=False, edgecolor='red',
                      linewidth=2, linestyle='--', label='Error Radius')
    ]

    legend = ax2.legend(handles=legend_elements, loc='upper right',
                      bbox_to_anchor=(0.98, 0.98), fontsize=12,
                      framealpha=0.9, facecolor='black', edgecolor='white')
    plt.setp(legend.get_texts(), color='white')

    # Add error analysis overlay in TOP LEFT
    analysis_text = f"""IMPROVED ERROR ANALYSIS

Position Error: {error:.1f} pixels
Relative Error: {(error/map_width)*100:.3f}% of map width
Classification: {'EXCELLENT' if error < 50 else 'GOOD' if error < 100 else 'FAIR' if error < 200 else 'POOR'} prediction

CoordConv model {'achieved high precision' if error < 75 else 'successfully localized' if error < 150 else 'attempted localization'}
with spatial coordinate awareness"""

    ax2.text(0.02, 0.98, analysis_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='left', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.95, edgecolor='black'))

    # Add title with CONSISTENT size (fontsize=14)
    ax2.text(0.5, 0.04, 'Improved Model Accuracy: CoordConv vs Ground Truth',
            transform=ax2.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='bottom', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8, edgecolor='white'))

    # Save context visualization
    context_output_path = '../images/improved_cnn_context_demo.png'
    plt.savefig(context_output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"✅ Improved CNN context demo saved to {context_output_path}")
    plt.close()

    # Show statistics comparison
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    print(f"\nImproved Model Performance Summary:")
    print(f"Selected example: Frame {example_idx} with {error:.1f}px error")
    print(f"Overall performance: Mean {mean_error:.1f}px, Median {median_error:.1f}px")
    print(f"This example represents: {'Excellent' if error < 50 else 'Good' if error < 100 else 'Fair' if error < 200 else 'Poor'} prediction")

    return example_idx, error

if __name__ == "__main__":
    create_improved_cnn_demonstration()