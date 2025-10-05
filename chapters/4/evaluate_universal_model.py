#!/usr/bin/env python3
"""
Evaluate the universal model on the flight path and generate results
compatible with the existing visualization scripts.
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
from tqdm import tqdm

# Import the model class
from train_universal_model import UniversalCNN


def evaluate_universal_model():
    """Evaluate the universal model on the flight path."""
    print("Evaluating Universal Model on Flight Path")
    print("=" * 45)

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

    print("✅ Universal model loaded successfully")

    # Flight path configuration (same as baseline)
    map_path = '../../data/boneyard/davis_monthan_stitched_map.jpg'
    start_coord = (5500, 4500)  # Desert start
    end_coord = (4167, 4167)    # Boneyard end
    tile_size = (1200, 675)     # Model input size
    zoom_factor = 4
    num_frames = 150

    # Load map
    full_map = Image.open(map_path).convert('RGB')
    map_width, map_height = full_map.size

    print(f"Map size: {map_width} x {map_height}")
    print(f"Flight path: {start_coord} → {end_coord}")

    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Universal model uses 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Generate flight path
    path_x = np.linspace(start_coord[0], end_coord[0], num_frames)
    path_y = np.linspace(start_coord[1], end_coord[1], num_frames)

    crop_width = tile_size[0] // zoom_factor
    crop_height = tile_size[1] // zoom_factor

    gt_coords = []
    pred_coords = []

    print("Running inference on flight path...")
    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Processing frames"):
            cam_x, cam_y = path_x[i], path_y[i]

            # Crop and resize (same as baseline evaluation)
            left = int(cam_x - crop_width / 2)
            top = int(cam_y - crop_height / 2)
            right = left + crop_width
            bottom = top + crop_height

            frame = full_map.crop((left, top, right, bottom))
            frame = frame.resize(tile_size, Image.LANCZOS)

            # Run inference
            frame_tensor = transform(frame).unsqueeze(0).to(device)
            pred = model(frame_tensor).squeeze().cpu().numpy()

            # Denormalize (convert from [0,1] to pixel coordinates)
            pred_x = pred[0] * map_width
            pred_y = pred[1] * map_height

            gt_coords.append([cam_x, cam_y])
            pred_coords.append([pred_x, pred_y])

    gt_coords = np.array(gt_coords)
    pred_coords = np.array(pred_coords)

    # Calculate errors
    errors = np.sqrt(np.sum((gt_coords - pred_coords)**2, axis=1))
    mean_error = np.mean(errors)
    median_error = np.median(errors)

    print(f"\n🎯 FLIGHT PATH EVALUATION RESULTS:")
    print(f"   Mean Error:    {mean_error:.1f} pixels")
    print(f"   Median Error:  {median_error:.1f} pixels")
    print(f"   Min Error:     {errors.min():.1f} pixels")
    print(f"   Max Error:     {errors.max():.1f} pixels")
    print(f"   Std Dev:       {np.std(errors):.1f} pixels")

    # Compare with baseline
    baseline_error = 153.9
    improvement = ((baseline_error - mean_error) / baseline_error) * 100
    print(f"\n📊 COMPARISON:")
    print(f"   Baseline:      {baseline_error:.1f} pixels")
    print(f"   Universal:     {mean_error:.1f} pixels")
    print(f"   Improvement:   {improvement:.1f}% better!")

    # Save results in the format expected by visualization scripts
    results = {
        'ground_truth': gt_coords,
        'predictions': pred_coords,
        'errors': errors,
        'mean_error': mean_error,
        'median_error': median_error,
        'demo_frame': len(errors) // 2,  # Middle frame for demo
        'demo_error': errors[len(errors) // 2]
    }

    # Save in the expected location for visualization scripts
    results_path = 'artifacts/universal_model_flight_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✅ Results saved to {results_path}")
    print("Ready for visualization generation!")

    return results


if __name__ == "__main__":
    evaluate_universal_model()