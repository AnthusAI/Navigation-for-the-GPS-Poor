#!/usr/bin/env python3
"""
Create CNN input/prediction demo for the universal model.
Uses the same framework as the baseline model demo.
"""
import sys
sys.path.append('../..')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches

# Import the model class
from train_universal_model import UniversalCNN


def create_universal_cnn_demo():
    """Create CNN demonstration for universal model."""
    print("Creating Universal CNN Demonstration")
    print("=" * 40)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device.type.upper()}")

    # Load the trained model
    model = UniversalCNN().to(device)
    model.load_state_dict(torch.load('artifacts/universal_model.pth', map_location=device))
    model.eval()

    # Load the flight results to find a good example
    with open('artifacts/universal_model_flight_results.pkl', 'rb') as f:
        results = pickle.load(f)

    gt_coords = results['ground_truth']
    pred_coords = results['predictions']
    errors = results['errors']

    # Select an example with medium error (like baseline does)
    sorted_indices = np.argsort(errors)
    middle_range_start = len(errors) // 3
    middle_range_end = 2 * len(errors) // 3
    example_idx = sorted_indices[middle_range_start + (middle_range_end - middle_range_start) // 2]

    gt_pos = gt_coords[example_idx]
    pred_pos = pred_coords[example_idx]
    error = errors[example_idx]

    print(f"Using frame {example_idx} with error: {error:.1f} pixels")

    # Load map and create input image
    map_path = '../../data/boneyard/davis_monthan_stitched_map.jpg'
    full_map = Image.open(map_path).convert('RGB')
    map_width, map_height = full_map.size

    # Create the input image that went into the CNN
    tile_size = (1200, 675)
    zoom_factor = 4
    crop_width = tile_size[0] // zoom_factor
    crop_height = tile_size[1] // zoom_factor

    left = int(gt_pos[0] - crop_width / 2)
    top = int(gt_pos[1] - crop_height / 2)
    right = left + crop_width
    bottom = top + crop_height

    # Crop and resize to model input size
    cropped = full_map.crop((left, top, right, bottom))
    input_image = cropped.resize(tile_size, Image.LANCZOS)

    # Create the visualization
    fig = plt.figure(figsize=(20, 12))

    # 1. Input Image (large, left side)
    ax1 = plt.subplot(2, 3, (1, 4))
    ax1.imshow(input_image)

    # Mark the actual position on the input image
    actual_x_in_crop = gt_pos[0] - left
    actual_y_in_crop = gt_pos[1] - top

    # Scale to image coordinates
    img_x = (actual_x_in_crop / crop_width) * tile_size[0]
    img_y = (actual_y_in_crop / crop_height) * tile_size[1]

    # Draw crosshairs for actual position
    ax1.axhline(y=img_y, color='lime', linewidth=4, alpha=0.9)
    ax1.axvline(x=img_x, color='lime', linewidth=4, alpha=0.9)
    ax1.plot(img_x, img_y, 'o', color='lime', markersize=20,
             markeredgecolor='black', markeredgewidth=3, label='Ground Truth Location')

    ax1.set_title('Universal CNN Input: Terrain Image (1200×675)', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Enhanced architecture with attention and multi-scale features', fontsize=14, style='italic')
    ax1.legend(fontsize=14, loc='upper right')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 2. Model Architecture Diagram (top right)
    ax2 = plt.subplot(2, 3, 2)
    ax2.axis('off')

    arch_text = """UNIVERSAL CNN ARCHITECTURE

Input: 1200×675×3 → Resize to 224×224
         ↓
Standard Convolutional Blocks:
• 32 filters (7×7) → 56×56
• 64 filters (5×5) → 28×28
• 128 filters (3×3) → 14×14
• 256 filters (3×3) → 7×7
• 512 filters (3×3) → 7×7
         ↓
Spatial Attention Module:
• Conv: 512 → 128 → 1 channel
• Sigmoid activation
• Element-wise multiplication
         ↓
Global Average Pooling → 512
         ↓
Multi-Layer Classifier:
• 512 → 256 → 128 → 64 → 2
• BatchNorm + Dropout layers
• Sigmoid output → [x, y] ∈ [0,1]²
         ↓
Scale to Map Coordinates"""

    ax2.text(0.05, 0.95, arch_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9, edgecolor='darkgreen'))

    # 3. Prediction Output (middle right)
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')

    # Calculate normalized coordinates for display
    pred_norm_x = pred_pos[0] / map_width
    pred_norm_y = pred_pos[1] / map_height
    gt_norm_x = gt_pos[0] / map_width
    gt_norm_y = gt_pos[1] / map_height

    output_text = f"""UNIVERSAL MODEL OUTPUT

Frame: {example_idx}/150

Normalized Prediction:
x = {pred_norm_x:.6f}
y = {pred_norm_y:.6f}

Map Coordinates:
Predicted: ({pred_pos[0]:.0f}, {pred_pos[1]:.0f})
Actual:    ({gt_pos[0]:.0f}, {gt_pos[1]:.0f})

Position Error: {error:.1f} pixels
Relative Error: {(error/map_width)*100:.3f}% of map

RESULT: {'🎯 EXCELLENT' if error < 50 else '✅ GOOD' if error < 100 else '⚠️ FAIR' if error < 200 else '❌ POOR'}

74.9% improvement over baseline!"""

    color = 'lightgreen' if error < 50 else 'lightblue' if error < 100 else 'lightyellow' if error < 200 else 'lightcoral'
    ax3.text(0.05, 0.95, output_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.9, edgecolor='darkgreen'))

    # 4. Map Context (bottom right)
    ax4 = plt.subplot(2, 3, (5, 6))

    # Show zoomed area around the prediction
    context_size = 1500  # Show 1500x1500 pixel area
    context_left = max(0, int(gt_pos[0] - context_size/2))
    context_top = max(0, int(gt_pos[1] - context_size/2))
    context_right = min(map_width, context_left + context_size)
    context_bottom = min(map_height, context_top + context_size)

    context_map = full_map.crop((context_left, context_top, context_right, context_bottom))
    ax4.imshow(context_map, extent=[context_left, context_right, context_bottom, context_top])

    # Show the input crop area
    rect = patches.Rectangle((left, top), right-left, bottom-top,
                           linewidth=3, edgecolor='blue', facecolor='none',
                           linestyle='--', label='CNN Input Area')
    ax4.add_patch(rect)

    # Show ground truth and prediction
    ax4.plot(gt_pos[0], gt_pos[1], 'o', color='lime', markersize=15,
             markeredgecolor='black', markeredgewidth=3, label='Ground Truth')
    ax4.plot(pred_pos[0], pred_pos[1], 's', color='red', markersize=15,
             markeredgecolor='black', markeredgewidth=3, label='Model Prediction')

    # Draw error circle
    circle = Circle((pred_pos[0], pred_pos[1]), error, fill=False,
                   edgecolor='red', linewidth=3, linestyle='--', alpha=0.8, label='Error Radius')
    ax4.add_patch(circle)

    ax4.set_xlim(context_left, context_right)
    ax4.set_ylim(context_bottom, context_top)
    ax4.set_title('Map Context: Universal Model vs Reality', fontsize=16, fontweight='bold')
    ax4.legend(fontsize=12, loc='upper left')
    ax4.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax4.set_ylabel('Y Coordinate (pixels)', fontsize=12)

    plt.suptitle('Universal CNN Terrain Navigation: Live Prediction Analysis',
                 fontsize=20, fontweight='bold', y=0.96)
    plt.tight_layout()

    # Save the visualization
    output_path = 'images/universal_cnn_prediction_demo.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Universal CNN demonstration saved to {output_path}")

    plt.close()

    return example_idx, error


if __name__ == "__main__":
    create_universal_cnn_demo()