#!/usr/bin/env python3
"""
Create a CNN input/output demonstration visualization showing exactly what
the model sees and predicts. This educational visual shows the complete
prediction process from terrain image to position output.
"""
import sys
sys.path.append('../../..')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

def load_evaluation_data():
    """Load the evaluation results."""
    results_path = '../artifacts/flight_evaluation_results.pkl'
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results

def create_input_image(map_path, center_coord, tile_size, zoom_factor):
    """Create the exact input image that goes into the CNN."""
    full_map = Image.open(map_path).convert('RGB')

    # Calculate crop area (same as in training script)
    crop_width = tile_size[0] // zoom_factor
    crop_height = tile_size[1] // zoom_factor

    left = int(center_coord[0] - crop_width / 2)
    top = int(center_coord[1] - crop_height / 2)
    right = left + crop_width
    bottom = top + crop_height

    # Crop and resize to model input size
    cropped = full_map.crop((left, top, right, bottom))
    input_image = cropped.resize(tile_size, Image.LANCZOS)

    return input_image, (left, top, right, bottom)

def run_prediction(model, input_image, device):
    """Run the model prediction on the input image."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.squeeze().cpu().numpy()

        # Get raw logits for confidence estimation
        features = model.features(input_tensor)
        flattened = torch.flatten(features, 1)
        logits = model.regressor[:-1](flattened)  # Before sigmoid
        confidence = torch.sigmoid(logits).squeeze().cpu().numpy()

    return prediction, confidence

def create_cnn_demo_visualization():
    """Create the complete CNN demonstration visualization."""
    print("Creating CNN input/output demonstration...")

    # Configuration
    map_path = '../../../data/boneyard/davis_monthan_stitched_map.jpg'
    tile_size = (1200, 675)
    zoom_factor = 4
    map_width, map_height = 7500, 7500

    # Load model and data
    model, results, device = load_model_and_data()
    gt_coords = results['ground_truth']
    pred_coords = results['predictions']
    errors = results['errors']

    # Select an interesting example (medium error, around frame 75)
    example_idx = 75
    gt_pos = gt_coords[example_idx]
    pred_pos = pred_coords[example_idx]
    error = errors[example_idx]

    # Create the input image
    input_image, crop_bounds = create_input_image(map_path, gt_pos, tile_size, zoom_factor)

    # Run prediction
    prediction_norm, confidence = run_prediction(model, input_image, device)

    # Denormalize prediction
    pred_x = prediction_norm[0] * map_width
    pred_y = prediction_norm[1] * map_height

    # Create the visualization
    fig = plt.figure(figsize=(20, 12))

    # 1. Input Image (large, left side)
    ax1 = plt.subplot(2, 3, (1, 4))
    ax1.imshow(input_image)

    # Mark the actual position on the input image if visible
    actual_x_in_crop = gt_pos[0] - crop_bounds[0]
    actual_y_in_crop = gt_pos[1] - crop_bounds[1]

    # Scale to image coordinates
    img_x = (actual_x_in_crop / (crop_bounds[2] - crop_bounds[0])) * tile_size[0]
    img_y = (actual_y_in_crop / (crop_bounds[3] - crop_bounds[1])) * tile_size[1]

    # Draw crosshairs for actual position
    ax1.axhline(y=img_y, color='lime', linewidth=3, alpha=0.8)
    ax1.axvline(x=img_x, color='lime', linewidth=3, alpha=0.8)
    ax1.plot(img_x, img_y, 'o', color='lime', markersize=15,
             markeredgecolor='black', markeredgewidth=2, label='Ground Truth')

    ax1.set_title('CNN Input: Terrain Image (1200×675)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Model sees this exact terrain view', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 2. Model Architecture Diagram (top right)
    ax2 = plt.subplot(2, 3, 2)
    ax2.axis('off')

    # Simple model architecture visualization
    arch_text = """
    CNN ARCHITECTURE

    Input: 1200×675×3 RGB
           ↓
    Conv Layers + Pooling
    32 → 64 → 128 → 256 filters
           ↓
    Global Average Pool
           ↓
    Dense Layers
    256 → 128 → 2
           ↓
    Sigmoid Activation
           ↓
    Output: [x, y] ∈ [0,1]²
    """

    ax2.text(0.1, 0.9, arch_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 3. Prediction Output (middle right)
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')

    output_text = f"""
    MODEL OUTPUT

    Raw Prediction:
    x = {prediction_norm[0]:.6f}
    y = {prediction_norm[1]:.6f}

    Denormalized:
    X = {pred_x:.1f} pixels
    Y = {pred_y:.1f} pixels

    Ground Truth:
    X = {gt_pos[0]:.1f} pixels
    Y = {gt_pos[1]:.1f} pixels

    Error: {error:.1f} pixels

    Confidence Proxy:
    X-coord: {confidence[0]:.3f}
    Y-coord: {confidence[1]:.3f}
    """

    ax3.text(0.1, 0.9, output_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # 4. Map Context (bottom right)
    ax4 = plt.subplot(2, 3, (5, 6))

    # Load full map for context
    full_map = Image.open(map_path).convert('RGB')

    # Show zoomed area around the prediction
    context_size = 2000  # Show 2000x2000 pixel area
    context_left = max(0, int(gt_pos[0] - context_size/2))
    context_top = max(0, int(gt_pos[1] - context_size/2))
    context_right = min(map_width, context_left + context_size)
    context_bottom = min(map_height, context_top + context_size)

    context_map = full_map.crop((context_left, context_top, context_right, context_bottom))
    ax4.imshow(context_map, extent=[context_left, context_right, context_bottom, context_top])

    # Show the input crop area
    ax4.add_patch(plt.Rectangle((crop_bounds[0], crop_bounds[1]),
                               crop_bounds[2]-crop_bounds[0], crop_bounds[3]-crop_bounds[1],
                               fill=False, edgecolor='blue', linewidth=3, label='CNN Input Area'))

    # Show ground truth and prediction
    ax4.plot(gt_pos[0], gt_pos[1], 'o', color='lime', markersize=12,
             markeredgecolor='black', markeredgewidth=2, label='Ground Truth')
    ax4.plot(pred_x, pred_y, 's', color='red', markersize=12,
             markeredgecolor='black', markeredgewidth=2, label='Prediction')

    # Draw error circle
    circle = plt.Circle((pred_x, pred_y), error, fill=False,
                       edgecolor='red', linewidth=2, linestyle='--', alpha=0.7, label='Error Radius')
    ax4.add_patch(circle)

    ax4.set_xlim(context_left, context_right)
    ax4.set_ylim(context_bottom, context_top)
    ax4.set_title('Map Context: Prediction vs Ground Truth', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.set_xlabel('X Coordinate (pixels)', fontsize=10)
    ax4.set_ylabel('Y Coordinate (pixels)', fontsize=10)

    plt.suptitle('CNN Terrain Navigation: Complete Prediction Process',
                 fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()

    # Save the visualization
    output_path = '../images/cnn_prediction_demo.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"CNN demonstration saved to {output_path}")

    plt.show()

if __name__ == "__main__":
    create_cnn_demo_visualization()