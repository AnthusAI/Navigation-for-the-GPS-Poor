#!/usr/bin/env python3
"""
Generate CNN demonstration visualization using existing trained model.
This creates the educational visualization without needing to retrain.
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

def load_existing_results():
    """Load the existing evaluation results."""
    results_path = '../artifacts/flight_evaluation_results.pkl'
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results

def create_cnn_demonstration_standalone():
    """Create educational CNN demonstration using existing results."""
    print("--- Creating CNN Demonstration from Existing Results ---")

    # Load existing evaluation results
    results = load_existing_results()
    gt_coords = results['ground_truth']
    pred_coords = results['predictions']
    errors = results['errors']

    # Configuration (same as training script)
    map_path = '../../../data/boneyard/davis_monthan_stitched_map.jpg'
    tile_size = (1200, 675)
    zoom_factor = 4

    # Select an interesting example (medium error around middle of flight)
    sorted_indices = np.argsort(errors)
    middle_range_start = len(errors) // 3
    middle_range_end = 2 * len(errors) // 3
    example_idx = sorted_indices[middle_range_start + (middle_range_end - middle_range_start) // 2]

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

    # Create two focused visualizations

    # Calculate normalized coordinates for display
    pred_norm_x = pred_pos[0] / map_width
    pred_norm_y = pred_pos[1] / map_height

    # Calculate input image aspect ratio
    input_aspect = tile_size[0] / tile_size[1]  # 1200/675 = 1.78 (close to 16:9)

    # 1. INPUT IMAGE VISUALIZATION (input image aspect ratio with overlay)
    # Use the actual input image aspect ratio (1200x675 = 1.78:1)
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 9))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax1.imshow(input_image)

    # Remove crosshairs - doesn't make sense to show actual location on model input
    # Model only sees the terrain, not where it actually is

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # Add prediction output as overlay in bottom right
    output_text = f"""MODEL OUTPUT

Frame: {example_idx}/150

Predicted Position:
x = {pred_norm_x:.6f}
y = {pred_norm_y:.6f}

Map Coordinates:
({pred_pos[0]:.0f}, {pred_pos[1]:.0f})

Classification: {'GOOD' if error < 100 else 'FAIR' if error < 200 else 'POOR'}"""

    color = 'lightgreen' if error < 100 else 'lightyellow' if error < 200 else 'lightcoral'
    ax1.text(0.98, 0.02, output_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='bottom', horizontalalignment='right', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.95, edgecolor='black'))

    # Add title with CONSISTENT size (fontsize=14 like trajectory plot)
    ax1.text(0.5, 0.04, 'CNN Input: What the Model Sees',
            transform=ax1.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='bottom', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8, edgecolor='white'))

    # Save input visualization
    input_output_path = '../images/cnn_input_demo.png'
    plt.savefig(input_output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"✅ CNN input demo saved to {input_output_path}")
    plt.close()

    # 2. ERROR ANALYSIS VISUALIZATION (16:9 aspect ratio, tighter zoom)
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 9))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Show much tighter zoom around the prediction for maximum detail
    context_size = 400  # 2x tighter zoom (was 800, now 400)
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
                  linestyle='None', label='Model Prediction'),
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
    analysis_text = f"""ERROR ANALYSIS

Position Error: {error:.1f} pixels
Relative Error: {(error/map_width)*100:.3f}% of map width
Classification: {'GOOD' if error < 100 else 'FAIR' if error < 200 else 'POOR'} prediction

Model successfully {'localized' if error < 150 else 'attempted to localize'}
aircraft position from terrain features"""

    ax2.text(0.02, 0.98, analysis_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='left', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.95, edgecolor='black'))

    # Add title with CONSISTENT size (fontsize=14 like trajectory plot)
    ax2.text(0.5, 0.04, 'Prediction Accuracy: Model vs Ground Truth',
            transform=ax2.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='bottom', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8, edgecolor='white'))

    # Save context visualization
    context_output_path = '../images/cnn_context_demo.png'
    plt.savefig(context_output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"✅ CNN context demo saved to {context_output_path}")
    plt.close()

    # Show statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    print(f"\nModel Performance Summary:")
    print(f"Selected example: Frame {example_idx} with {error:.1f}px error")
    print(f"Overall performance: Mean {mean_error:.1f}px, Median {median_error:.1f}px")
    print(f"This example represents: {'Good' if error < 100 else 'Fair' if error < 200 else 'Poor'} prediction")

    return example_idx, error

if __name__ == "__main__":
    create_cnn_demonstration_standalone()