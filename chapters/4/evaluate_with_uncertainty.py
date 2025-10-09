#!/usr/bin/env python3
"""
Comprehensive evaluation of uncertainty-enabled navigation model.
Generates flight path predictions with confidence estimates and visualizations.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from PIL import Image, ImageDraw, ImageFont
import pickle
from pathlib import Path
from datetime import datetime
import argparse

sys.path.append(str(Path(__file__).parent))
from train_model import BasicModel
from navigation.terrain_window import TerrainWindow
from navigation.flight_config import FlightPathConfig
from navigation.utils import get_device
from navigation.visualizer import PredictionVisualizer


def evaluate_with_uncertainty(model_path: str, flight_name: str = "main_evaluation",
                              num_eval_points: int = 20) -> dict:
    """
    Evaluate model on flight path with uncertainty estimates.

    Returns:
        Dict with predictions, uncertainties, errors, and calibration metrics
    """
    print(f"🎯 Evaluating Uncertainty Model on Flight Path")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Flight: {flight_name}")
    print(f"  Evaluation points: {num_eval_points}")

    device = get_device()

    # Load model with uncertainty
    model = BasicModel(predict_uncertainty=True)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # Initialize terrain window
    terrain_window = TerrainWindow()

    # Get flight path
    flight_path = FlightPathConfig.get_flight_path(flight_name)
    flight_coords = FlightPathConfig.create_flight_coordinates(flight_path)

    print(f"  Flight path loaded: {len(flight_coords)} total points")

    # Sample points evenly
    indices = np.linspace(0, len(flight_coords) - 1, num_eval_points).astype(int)
    sampled_coords = flight_coords[indices]

    # Run predictions with uncertainty
    predictions = []
    uncertainties = []
    errors = []
    terrain_tiles = []
    headings = []

    print(f"\n  Generating predictions with uncertainty...")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            coord = flight_coords[idx]
            pixel_x = coord[0] * 7500
            pixel_y = coord[1] * 7500

            # Calculate aircraft heading
            if idx < len(flight_coords) - 1:
                dx = flight_coords[idx + 1][0] - coord[0]
                dy = flight_coords[idx + 1][1] - coord[1]
                aircraft_heading = np.degrees(np.arctan2(dy, dx))
            else:
                aircraft_heading = 0
            headings.append(aircraft_heading)

            # Extract terrain with aircraft perspective
            environmental_effects = {
                'brightness': 1.0 + np.random.uniform(-0.1, 0.1),
                'contrast': 1.0,
                'fog_intensity': np.random.uniform(0.0, 0.2),
                'noise_std': np.random.uniform(1.0, 4.0)
            }

            terrain_tile = terrain_window.extract_model_input(
                pixel_x, pixel_y,
                aircraft_heading=aircraft_heading,
                environmental_effects=environmental_effects
            )
            terrain_tiles.append(terrain_tile)

            # Normalize and predict
            tile_tensor = torch.from_numpy(terrain_tile).float().permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            tile_tensor = (tile_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
            tile_tensor = tile_tensor.unsqueeze(0).to(device)

            # Get prediction and uncertainty
            pred_coord, pred_log_var = model(tile_tensor)
            pred_coord = pred_coord.cpu().numpy()[0]
            log_var = pred_log_var.cpu().numpy()[0, 0]

            predictions.append(pred_coord)

            # Calculate uncertainty in meters
            std_dev = np.sqrt(np.exp(log_var))
            uncertainty_meters = std_dev * 7500 * 10
            uncertainties.append(uncertainty_meters)

            # Calculate error in meters
            error_normalized = np.linalg.norm(pred_coord - coord)
            error_meters = error_normalized * 7500 * 10
            errors.append(error_meters)

            if (i + 1) % 5 == 0 or (i + 1) == num_eval_points:
                confidence = "HIGH" if uncertainty_meters < 500 else "MEDIUM" if uncertainty_meters < 1000 else "LOW"
                print(f"    {i+1}/{num_eval_points} | Error: {error_meters:4.0f}m | Uncertainty: {uncertainty_meters:4.0f}m | {confidence}")

    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    errors = np.array(errors)

    # Statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    mean_uncertainty = np.mean(uncertainties)

    # Calibration: check if errors fall within uncertainty bounds
    within_1_sigma = np.mean(errors < uncertainties)
    within_2_sigma = np.mean(errors < 2 * uncertainties)

    # Correlation between errors and uncertainties
    correlation = np.corrcoef(errors, uncertainties)[0, 1] if len(errors) > 1 else 0.0

    print(f"\n✅ Evaluation Complete!")
    print(f"  Mean error: {mean_error:.1f}m")
    print(f"  Median error: {median_error:.1f}m")
    print(f"  Mean uncertainty: {mean_uncertainty:.1f}m")
    print(f"\n  Calibration:")
    print(f"    Within 1σ: {within_1_sigma:.1%} (expect ~68%)")
    print(f"    Within 2σ: {within_2_sigma:.1%} (expect ~95%)")
    print(f"  Error-Uncertainty correlation: {correlation:.3f}")

    return {
        'model_path': model_path,
        'flight_name': flight_name,
        'trajectory_coords': sampled_coords,
        'predictions': predictions,
        'uncertainties': uncertainties,
        'errors': errors,
        'terrain_tiles': terrain_tiles,
        'headings': headings,
        'statistics': {
            'mean_error_m': mean_error,
            'median_error_m': median_error,
            'std_error_m': std_error,
            'min_error_m': np.min(errors),
            'max_error_m': np.max(errors),
            'mean_uncertainty_m': mean_uncertainty,
            'within_1_sigma': within_1_sigma,
            'within_2_sigma': within_2_sigma,
            'correlation': correlation,
            'num_points': num_eval_points
        }
    }


def create_uncertainty_scatter_plot(eval_results: dict, save_path: str = "images/uncertainty_vs_error.png"):
    """Create scatter plot showing relationship between error and uncertainty."""
    print(f"\n📊 Creating Error vs Uncertainty Plot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    errors = eval_results['errors']
    uncertainties = eval_results['uncertainties']
    correlation = eval_results['statistics']['correlation']

    # Left: Scatter plot
    ax1.scatter(uncertainties, errors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    ax1.plot([0, max(uncertainties)], [0, max(uncertainties)], 'r--', alpha=0.5, label='Perfect calibration')

    ax1.set_xlabel('Predicted Uncertainty (m)', fontsize=12)
    ax1.set_ylabel('Actual Error (m)', fontsize=12)
    ax1.set_title(f'Error vs Uncertainty\nCorrelation: {correlation:.3f}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Distribution comparison
    ax2.hist(errors, bins=15, alpha=0.5, label='Actual Errors', color='blue', edgecolor='black')
    ax2.hist(uncertainties, bins=15, alpha=0.5, label='Predicted Uncertainties', color='red', edgecolor='black')
    ax2.axvline(np.mean(errors), color='blue', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(errors):.0f}m')
    ax2.axvline(np.mean(uncertainties), color='red', linestyle='--', linewidth=2, label=f'Mean Uncertainty: {np.mean(uncertainties):.0f}m')

    ax2.set_xlabel('Distance (meters)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {save_path}")


def create_trajectory_with_uncertainty(eval_results: dict, save_path: str = "images/navigation_flight_trajectory.png"):
    """Create trajectory visualization with uncertainty ellipses."""
    print(f"\n🎨 Creating Trajectory with Uncertainty Visualization")

    visualizer = PredictionVisualizer()
    visualizer.load_satellite_map("../../data/boneyard/davis_monthan_stitched_map.jpg")

    # Convert to pixels
    true_coords_pixels = eval_results['trajectory_coords'] * 7500
    pred_coords_pixels = eval_results['predictions'] * 7500
    uncertainties_pixels = eval_results['uncertainties'] / 10  # Convert meters to pixels

    # Create flight_results dict
    flight_results = {
        'ground_truth': true_coords_pixels,
        'predictions': pred_coords_pixels,
        'errors': eval_results['errors'],
        'uncertainties': uncertainties_pixels,
        'mean_error': eval_results['statistics']['mean_error_m']
    }

    visualizer.create_flight_path_viz(
        flight_results,
        save_path=save_path
    )

    print(f"✅ Saved: {save_path}")


def create_confidence_weighted_demo(eval_results: dict, save_path: str = "images/confidence_weighted_demo.png"):
    """Demonstrate confidence-weighted position averaging."""
    print(f"\n🎯 Creating Confidence-Weighted Demo")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Load map
    terrain_window = TerrainWindow()
    map_img = terrain_window.stitched_map

    true_coords = eval_results['trajectory_coords'] * 7500
    predictions = eval_results['predictions'] * 7500
    uncertainties = eval_results['uncertainties']
    errors = eval_results['errors']

    # Calculate inverse variance weights (precision)
    variances = (uncertainties / (7500 * 10)) ** 2  # Normalized variance
    precisions = 1.0 / (variances + 1e-6)
    weights = precisions / np.sum(precisions)

    # Weighted average position
    weighted_position = np.sum(predictions * weights[:, np.newaxis], axis=0)
    true_center = np.mean(true_coords, axis=0)

    # Left: Show all predictions with uncertainty
    zoom_size = 1500
    center = true_center
    x_min, x_max = int(center[0] - zoom_size), int(center[0] + zoom_size)
    y_min, y_max = int(center[1] - zoom_size), int(center[1] + zoom_size)

    zoomed_map = map_img[y_min:y_max, x_min:x_max]
    ax1.imshow(zoomed_map, extent=[x_min, x_max, y_max, y_min])

    # Plot predictions with uncertainty circles
    for i, (pred, unc, err) in enumerate(zip(predictions, uncertainties, errors)):
        alpha = weights[i] * 5  # Scale for visibility
        color = plt.cm.RdYlGn_r(err / np.max(errors))

        # Uncertainty circle
        unc_pixels = unc / 10
        circle = Circle(pred, unc_pixels, fill=False, edgecolor=color, linewidth=2, alpha=0.5)
        ax1.add_patch(circle)

        # Prediction point
        ax1.scatter(pred[0], pred[1], c=[color], s=100*alpha, alpha=0.7, edgecolors='black', linewidth=1)

    # Weighted average
    ax1.scatter(weighted_position[0], weighted_position[1], c='blue', s=300, marker='*',
               edgecolors='white', linewidth=2, label='Weighted Average', zorder=10)

    # True center
    ax1.scatter(true_center[0], true_center[1], c='lime', s=200, marker='o',
               edgecolors='black', linewidth=2, label='True Center', zorder=10)

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_max, y_min)
    ax1.set_title('Confidence-Weighted Navigation\n(circle size = uncertainty)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.axis('off')

    # Right: Weight distribution
    ax2.bar(range(len(weights)), weights, color=plt.cm.viridis(weights / np.max(weights)))
    ax2.set_xlabel('Prediction Index', fontsize=12)
    ax2.set_ylabel('Weight (confidence)', fontsize=12)
    ax2.set_title(f'Confidence Weights\nHigher weight = lower uncertainty', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add summary text
    weighted_error = np.linalg.norm(weighted_position - true_center) * 10  # to meters
    simple_avg = np.mean(predictions, axis=0)
    simple_error = np.linalg.norm(simple_avg - true_center) * 10

    summary_text = f"Weighted avg error: {weighted_error:.1f}m\nSimple avg error: {simple_error:.1f}m\nImprovement: {simple_error - weighted_error:.1f}m"
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {save_path}")
    print(f"  Weighted error: {weighted_error:.1f}m vs Simple average: {simple_error:.1f}m")


def main():
    parser = argparse.ArgumentParser(description="Evaluate uncertainty model comprehensively")
    parser.add_argument("--model", default="artifacts/model_20251008_160225.pth",
                       help="Path to trained model with uncertainty")
    parser.add_argument("--flight", default="main_evaluation",
                       help="Flight path name")
    parser.add_argument("--points", type=int, default=20,
                       help="Number of evaluation points")
    args = parser.parse_args()

    print(f"📊 Comprehensive Uncertainty Model Evaluation")
    print("=" * 60)

    # Run evaluation
    eval_results = evaluate_with_uncertainty(args.model, args.flight, args.points)

    # Create visualizations
    create_uncertainty_scatter_plot(eval_results, "images/uncertainty_vs_error.png")
    create_trajectory_with_uncertainty(eval_results, "images/navigation_flight_trajectory.png")
    create_confidence_weighted_demo(eval_results, "images/confidence_weighted_demo.png")

    # Save results
    results_path = "artifacts/uncertainty_evaluation_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(eval_results, f)

    print(f"\n" + "=" * 60)
    print(f"✅ Complete Uncertainty Evaluation Finished!")
    print(f"  Results: {results_path}")
    print(f"  Visualizations:")
    print(f"    - images/uncertainty_vs_error.png")
    print(f"    - images/navigation_flight_trajectory.png")
    print(f"    - images/confidence_weighted_demo.png")
    print(f"\n  Model Performance:")
    print(f"    Mean error: {eval_results['statistics']['mean_error_m']:.1f}m")
    print(f"    Calibration: {eval_results['statistics']['within_1_sigma']:.1%} within 1σ")
    print(f"    Correlation: {eval_results['statistics']['correlation']:.3f}")


if __name__ == "__main__":
    main()
