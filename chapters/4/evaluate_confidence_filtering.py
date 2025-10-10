#!/usr/bin/env python3
"""
Evaluate confidence-based filtering for navigation.

Tests the hypothesis: Can we improve navigation accuracy by only using
the most confident predictions?
"""

import sys
import torch
import numpy as np
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from train_model import BasicModel
from navigation.terrain_window import TerrainWindow
from navigation.flight_config import FlightPathConfig
from navigation.utils import get_device
from navigation.visualizer import PredictionVisualizer


def evaluate_confidence_filtering(model_path: str,
                                  calibration_path: str,
                                  flight_name: str = "main_evaluation",
                                  total_points: int = 100,
                                  keep_top_n: int = 20):
    """
    Evaluate model with confidence-based filtering.

    Args:
        model_path: Path to trained model
        calibration_path: Path to calibration data
        flight_name: Flight path to evaluate
        total_points: Total points to evaluate
        keep_top_n: Number of most confident predictions to keep

    Returns:
        Dictionary with results
    """
    print(f"🎯 Confidence-Based Navigation Filtering")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Calibration: {calibration_path}")
    print(f"  Total evaluation points: {total_points}")
    print(f"  Keeping top {keep_top_n} most confident")

    device = get_device()

    # Load model
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model = BasicModel(predict_uncertainty=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load calibration
    with open(calibration_path, 'rb') as f:
        calibration_data = pickle.load(f)
        calibration_factor = calibration_data['calibration_factor']
    print(f"  ✅ Loaded calibration factor: {calibration_factor:.4f}")

    # Initialize terrain window
    terrain_window = TerrainWindow()

    # Get flight path
    flight_path = FlightPathConfig.get_flight_path(flight_name)
    flight_coords = FlightPathConfig.create_flight_coordinates(flight_path)
    print(f"  Flight path loaded: {len(flight_coords)} total points")

    # Sample points evenly along flight path
    indices = np.linspace(0, len(flight_coords) - 1, total_points).astype(int)

    # Run predictions on all points
    print(f"\n  Evaluating {total_points} points...")
    predictions = []
    uncertainties = []
    errors = []
    coords_list = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            coord = flight_coords[idx]
            pixel_x = coord[0] * 7500
            pixel_y = coord[1] * 7500

            # Calculate heading (same as evaluate_augmented_model_live.py)
            if idx < len(flight_coords) - 1:
                current = coord
                next_point = flight_coords[idx + 1]
                dx = next_point[0] - current[0]
                dy = next_point[1] - current[1]
                # Calculate heading: 0° = North (up), 90° = East (right)
                heading = np.degrees(np.arctan2(dx, -dy)) % 360
            else:
                # Use heading from previous point
                if idx > 0:
                    current = flight_coords[idx - 1]
                    next_point = coord
                    dx = next_point[0] - current[0]
                    dy = next_point[1] - current[1]
                    heading = np.degrees(np.arctan2(dx, -dy)) % 360
                else:
                    heading = 296  # Default heading for this flight

            # Extract terrain with aircraft perspective rotation
            environmental_effects = {
                'brightness': 1.0 + np.random.uniform(-0.1, 0.1),
                'contrast': 1.0,
                'fog_intensity': np.random.uniform(0.0, 0.2),
                'noise_std': np.random.uniform(1.0, 4.0)
            }

            terrain_tile = terrain_window.extract_model_input(
                pixel_x, pixel_y,
                aircraft_heading=heading,
                environmental_effects=environmental_effects
            )

            # Normalize
            tile_tensor = torch.from_numpy(terrain_tile).float().permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            tile_tensor = (tile_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
            tile_tensor = tile_tensor.unsqueeze(0).to(device)

            # Predict
            pred_coord, pred_log_var = model(tile_tensor)
            pred_coord = pred_coord.cpu().numpy()[0]
            log_var = pred_log_var.cpu().numpy()[0, 0]

            # Calculate calibrated uncertainty
            std_dev = np.sqrt(np.exp(log_var))
            uncertainty_m = std_dev * calibration_factor * 7500 * 10

            # Calculate error
            error_normalized = np.linalg.norm(pred_coord - coord)
            error_meters = error_normalized * 7500 * 10

            predictions.append(pred_coord)
            uncertainties.append(uncertainty_m)
            errors.append(error_meters)
            coords_list.append(coord)

            if (i + 1) % 25 == 0:
                print(f"    Progress: {i+1}/{total_points}")

    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    errors = np.array(errors)
    coords_list = np.array(coords_list)

    # Calculate baseline metrics (all points)
    print(f"\n📊 Baseline (All {total_points} Points):")
    print(f"  Mean error: {np.mean(errors):.1f}m")
    print(f"  Median error: {np.median(errors):.1f}m")
    print(f"  Std error: {np.std(errors):.1f}m")
    print(f"  Min error: {np.min(errors):.1f}m")
    print(f"  Max error: {np.max(errors):.1f}m")

    # Sort by uncertainty (ascending = most confident first)
    sorted_indices = np.argsort(uncertainties)
    top_confident_indices = sorted_indices[:keep_top_n]

    # Get statistics for most confident predictions
    confident_errors = errors[top_confident_indices]
    confident_uncertainties = uncertainties[top_confident_indices]
    confident_predictions = predictions[top_confident_indices]
    confident_coords = coords_list[top_confident_indices]

    print(f"\n🎯 Confidence-Filtered (Top {keep_top_n} Most Confident):")
    print(f"  Mean error: {np.mean(confident_errors):.1f}m")
    print(f"  Median error: {np.median(confident_errors):.1f}m")
    print(f"  Std error: {np.std(confident_errors):.1f}m")
    print(f"  Min error: {np.min(confident_errors):.1f}m")
    print(f"  Max error: {np.max(confident_errors):.1f}m")
    print(f"\n  Uncertainty range: {np.min(confident_uncertainties):.1f}m to {np.max(confident_uncertainties):.1f}m")

    # Calculate improvement
    baseline_mean = np.mean(errors)
    filtered_mean = np.mean(confident_errors)
    improvement = baseline_mean - filtered_mean
    improvement_pct = 100 * improvement / baseline_mean

    print(f"\n✅ Improvement Analysis:")
    print(f"  Baseline mean error: {baseline_mean:.1f}m")
    print(f"  Filtered mean error: {filtered_mean:.1f}m")
    print(f"  Absolute improvement: {improvement:.1f}m")
    print(f"  Relative improvement: {improvement_pct:.1f}%")

    if improvement > 0:
        print(f"\n  🎉 SUCCESS! Confidence filtering improves accuracy by {improvement:.1f}m ({improvement_pct:.1f}%)")
        print(f"     The uncertainty signal is useful for selecting better predictions!")
    elif improvement < -20:
        print(f"\n  ⚠️ WORSE: Confidence filtering degrades accuracy by {abs(improvement):.1f}m")
        print(f"     The uncertainty signal may be anti-correlated with actual error")
    else:
        print(f"\n  ➡️ NEUTRAL: Confidence filtering has minimal effect ({abs(improvement):.1f}m)")
        print(f"     The uncertainty signal is not discriminative enough")

    # Save results
    results = {
        'model_path': model_path,
        'calibration_path': calibration_path,
        'flight_name': flight_name,
        'total_points': total_points,
        'keep_top_n': keep_top_n,
        'baseline': {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'all_errors': errors,
            'all_uncertainties': uncertainties
        },
        'filtered': {
            'mean_error': float(np.mean(confident_errors)),
            'median_error': float(np.median(confident_errors)),
            'std_error': float(np.std(confident_errors)),
            'errors': confident_errors,
            'uncertainties': confident_uncertainties,
            'indices': top_confident_indices
        },
        'improvement': {
            'absolute': float(improvement),
            'relative_pct': float(improvement_pct)
        }
    }

    output_path = 'artifacts/confidence_filtering_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n💾 Results saved: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--calibration", required=True, help="Path to calibration data")
    parser.add_argument("--flight", default="main_evaluation", help="Flight path name")
    parser.add_argument("--total", type=int, default=100, help="Total points to evaluate")
    parser.add_argument("--keep", type=int, default=20, help="Number of most confident to keep")

    args = parser.parse_args()

    evaluate_confidence_filtering(
        args.model,
        args.calibration,
        args.flight,
        args.total,
        args.keep
    )
