#!/usr/bin/env python3
"""
Simple command to evaluate models on flight path.
Usage: python evaluate_model.py --model model_20251006_173043.pth --arch deep
"""
import argparse
import torch
import numpy as np
import sys
from pathlib import Path
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent))

from navigation.extractor import TerrainExtractor
from navigation.visualizer import PredictionVisualizer
from navigation.flight_config import FlightPathConfig
from train_model import SimpleDeepModel, BasicModel

def load_model(model_path, arch):
    """Load trained model."""
    if arch == "deep":
        model = SimpleDeepModel()
    elif arch == "basic":
        model = BasicModel()
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, device

def evaluate_flight_path(model, device):
    """Evaluate model on flight path."""
    print("Evaluating on flight path...")

    # Setup
    extractor = TerrainExtractor()
    extractor.load_satellite_map("../../data/boneyard/davis_monthan_stitched_map.jpg")

    flight_path = FlightPathConfig.get_default_flight_path()
    flight_coords = FlightPathConfig.create_pixel_coordinates(flight_path)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Evaluate on 20 points
    step = max(1, len(flight_coords) // 20)
    indices = range(0, len(flight_coords), step)

    actual_coords = []
    predicted_coords = []
    terrain_tiles = []
    errors = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            x, y = flight_coords[idx]

            # Extract terrain
            tile = extractor.extract_tile(int(x), int(y), 224)
            terrain_tiles.append(tile)

            # Get prediction
            input_tensor = transform(tile).unsqueeze(0).to(device)
            output = model(input_tensor)
            pred_norm = output.cpu().numpy()[0]

            # Convert to pixels
            pred_x = pred_norm[0] * 7500
            pred_y = pred_norm[1] * 7500

            actual_coords.append([x, y])
            predicted_coords.append([pred_x, pred_y])

            # Calculate error in meters
            error = np.sqrt((x - pred_x)**2 + (y - pred_y)**2) * 2.0
            errors.append(error)

    actual_coords = np.array(actual_coords)
    predicted_coords = np.array(predicted_coords)
    errors = np.array(errors)

    print(f"  Mean error: {np.mean(errors):.0f} meters")
    print(f"  Median error: {np.median(errors):.0f} meters")
    print(f"  Max error: {np.max(errors):.0f} meters")
    print(f"  Min error: {np.min(errors):.0f} meters")

    return {
        'ground_truth': actual_coords,
        'predictions': predicted_coords,
        'errors': errors,
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'terrain_images': terrain_tiles
    }

def generate_visualization(results):
    """Generate flight path visualization."""
    print("Generating visualization...")

    visualizer = PredictionVisualizer()
    visualizer.load_satellite_map("../../data/boneyard/davis_monthan_stitched_map.jpg")

    visualizer.create_flight_path_viz(
        results,
        save_path="images/navigation_flight_trajectory.png"
    )

    print("✅ Visualization saved: images/navigation_flight_trajectory.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model (.pth)")
    parser.add_argument("--arch", choices=["basic", "deep"], required=True, help="Model architecture")
    args = parser.parse_args()

    print(f"🔬 Evaluating Model")
    print(f"  Model: {args.model}")
    print(f"  Architecture: {args.arch}")

    # Load model
    model, device = load_model(args.model, args.arch)

    # Evaluate
    results = evaluate_flight_path(model, device)

    # Generate visualization
    generate_visualization(results)

    print(f"\n📊 Final Results:")
    print(f"  Mean Error: {results['mean_error']:.0f} meters")

    # Assessment
    if results['mean_error'] < 600:
        print(f"  🎯 EXCELLENT: Navigation-grade precision!")
    elif results['mean_error'] < 800:
        print(f"  ✅ GOOD: Strong performance")
    elif results['mean_error'] < 1200:
        print(f"  🟡 IMPROVED: Getting better")
    else:
        print(f"  🔄 PROGRESS: Try more data or different approach")

if __name__ == "__main__":
    main()