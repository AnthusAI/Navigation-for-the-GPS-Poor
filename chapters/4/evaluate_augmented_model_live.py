#!/usr/bin/env python3
"""
Evaluate the augmented navigation model with live image generation.

This script generates evaluation images on-the-fly with proper aircraft perspective
rotation to match the training data generation approach.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pickle
from pathlib import Path
from datetime import datetime
import argparse

sys.path.append(str(Path(__file__).parent))
from navigation.terrain_window import TerrainWindow
from navigation.flight_config import FlightPathConfig
from navigation.utils import get_device
from navigation.visualizer import PredictionVisualizer


class BasicModel(nn.Module):
    """Ultra-simple model for small datasets with optional uncertainty estimation."""
    def __init__(self, predict_uncertainty=False):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights
        self.predict_uncertainty = predict_uncertainty
        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        if predict_uncertainty:
            # Shared features
            self.shared = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(1024, 128),
                nn.ReLU()
            )
            # Coordinate head
            self.coord_head = nn.Linear(128, 2)
            # Uncertainty head (predicts log variance)
            self.uncertainty_head = nn.Linear(128, 1)
        else:
            # Minimal classifier to prevent overfitting
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )

    def forward(self, x):
        if self.predict_uncertainty:
            # Extract features from DenseNet backbone
            features = self.backbone.features(x)
            features = torch.nn.functional.relu(features, inplace=True)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)

            # Pass through shared layers
            shared_features = self.shared(features)

            # Get coordinates and uncertainty
            coords = self.coord_head(shared_features)
            log_var = self.uncertainty_head(shared_features)

            return coords, log_var
        else:
            return self.backbone(x)


def calculate_aircraft_heading(flight_coords, index):
    """Calculate aircraft heading at a given point on the flight path."""
    if index < len(flight_coords) - 1:
        current = flight_coords[index]
        next_point = flight_coords[index + 1]
        dx = next_point[0] - current[0]
        dy = next_point[1] - current[1]
        # Calculate heading: 0° = North (up), 90° = East (right)
        heading = np.degrees(np.arctan2(dx, -dy)) % 360
        return heading
    else:
        # Use previous heading for last point
        return calculate_aircraft_heading(flight_coords, index - 1)


def evaluate_model_with_live_generation(model_path: str,
                                        flight_name: str = "main_evaluation",
                                        num_eval_points: int = 100,
                                        viz_points: int = 20,
                                        calibration_path: str = None) -> dict:
    """
    Evaluate model on flight path with live image generation using aircraft perspective.

    Args:
        model_path: Path to trained model
        flight_name: Flight path to evaluate
        num_eval_points: Number of points to evaluate along flight path
        viz_points: Number of points to show in visualization (sampled from eval points)
        calibration_path: Path to calibration data (optional)

    Returns:
        Dictionary with evaluation results
    """
    print(f"🎯 Evaluating Model on Flight Path (Live Generation)")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Flight: {flight_name}")
    print(f"  Evaluation points: {num_eval_points}")
    print(f"  Visualization points: {viz_points}")

    device = get_device()

    # Try to detect if model has uncertainty by checking state dict keys
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    has_uncertainty = 'uncertainty_head.weight' in state_dict or 'coord_head.weight' in state_dict

    # Load calibration if provided
    calibration_factor = 1.0
    if calibration_path and Path(calibration_path).exists():
        with open(calibration_path, 'rb') as f:
            calibration_data = pickle.load(f)
            calibration_factor = calibration_data['calibration_factor']
        print(f"  ✅ Loaded calibration factor: {calibration_factor:.4f}")

    # Load model
    model = BasicModel(predict_uncertainty=has_uncertainty)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if has_uncertainty:
        print(f"  ✅ Detected uncertainty-enabled model")
        if calibration_factor != 1.0:
            print(f"  ✅ Using calibrated uncertainties")

    # Initialize terrain window for live extraction
    terrain_window = TerrainWindow()

    # Get flight path
    flight_path = FlightPathConfig.get_flight_path(flight_name)
    flight_coords = FlightPathConfig.create_flight_coordinates(flight_path)

    print(f"  Flight path loaded: {len(flight_coords)} total points")

    # Sample points evenly along flight path
    indices = np.linspace(0, len(flight_coords) - 1, num_eval_points).astype(int)
    sampled_coords = flight_coords[indices]

    # Run predictions with live image generation
    predictions = []
    errors = []
    uncertainties = []
    terrain_tiles = []
    headings = []

    print(f"\n  Generating images and running predictions...")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            coord = flight_coords[idx]

            # Convert normalized coordinate to pixel coordinate
            pixel_x = coord[0] * 7500
            pixel_y = coord[1] * 7500

            # Calculate aircraft heading at this point
            aircraft_heading = calculate_aircraft_heading(flight_coords, idx)
            headings.append(aircraft_heading)

            # Extract terrain with aircraft perspective rotation
            # Use environmental presets (same as training)
            from navigation.environmental_presets import EnvironmentalPresets
            if np.random.random() < 0.7:  # 70% probability like training
                environmental_effects = EnvironmentalPresets.get_random_preset()
            else:
                environmental_effects = None

            terrain_tile = terrain_window.extract_model_input(
                pixel_x, pixel_y,
                aircraft_heading=aircraft_heading,
                environmental_effects=environmental_effects
            )

            terrain_tiles.append(terrain_tile)

            # Apply ColorJitter augmentation (same as training)
            from torchvision import transforms
            from PIL import Image
            color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
            tile_pil = Image.fromarray(terrain_tile.astype('uint8'))
            tile_pil = color_jitter(tile_pil)
            terrain_tile_augmented = np.array(tile_pil)

            # Convert to tensor and normalize
            tile_tensor = torch.from_numpy(terrain_tile_augmented).float().permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            tile_tensor = (tile_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
            tile_tensor = tile_tensor.unsqueeze(0).to(device)

            # Predict
            if has_uncertainty:
                pred_coord, pred_log_var = model(tile_tensor)
                pred_coord = pred_coord.cpu().numpy()[0]
                log_var = pred_log_var.cpu().numpy()[0, 0]

                # Convert log variance to uncertainty in meters
                std_dev = np.sqrt(np.exp(log_var))
                # Apply calibration to convert to expected error
                uncertainty_m = std_dev * calibration_factor * 7500 * 10  # Convert to meters
                uncertainties.append(uncertainty_m)
            else:
                pred_coord = model(tile_tensor).cpu().numpy()[0]

            predictions.append(pred_coord)

            # Calculate error in meters
            error_normalized = np.linalg.norm(pred_coord - coord)
            error_pixels = error_normalized * 7500
            error_meters = error_pixels * 10  # ~10m per pixel
            errors.append(error_meters)

            if (i + 1) % 5 == 0 or (i + 1) == num_eval_points:
                if has_uncertainty:
                    print(f"    Progress: {i+1}/{num_eval_points} | "
                          f"Heading: {aircraft_heading:.0f}° | Error: {error_meters:.0f}m | "
                          f"Uncertainty: {uncertainty_m:.0f}m")
                else:
                    print(f"    Progress: {i+1}/{num_eval_points} | "
                          f"Heading: {aircraft_heading:.0f}° | Error: {error_meters:.0f}m")

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)

    print(f"\n✅ Evaluation Complete!")
    print(f"  Mean error: {mean_error:.1f}m")
    print(f"  Median error: {median_error:.1f}m")
    print(f"  Std error: {std_error:.1f}m")
    print(f"  Min error: {np.min(errors):.1f}m")
    print(f"  Max error: {np.max(errors):.1f}m")

    if has_uncertainty:
        uncertainties = np.array(uncertainties)
        mean_uncertainty = np.mean(uncertainties)
        uncertainty_cov = 100 * np.std(uncertainties) / mean_uncertainty if mean_uncertainty > 0 else 0
        print(f"\n  Uncertainty Statistics:")
        print(f"    Mean: {mean_uncertainty:.1f}m")
        print(f"    Range: {np.min(uncertainties):.1f}m to {np.max(uncertainties):.1f}m")
        print(f"    CoV: {uncertainty_cov:.1f}%")

    results = {
        'model_path': model_path,
        'flight_name': flight_name,
        'flight_coords': flight_coords,
        'trajectory_coords': sampled_coords,
        'predictions': np.array(predictions),
        'errors': np.array(errors),
        'terrain_tiles': terrain_tiles,
        'headings': headings,
        'has_uncertainty': has_uncertainty,
        'viz_points': viz_points,  # How many points to show in visualization
        'statistics': {
            'mean_error_m': mean_error,
            'median_error_m': median_error,
            'std_error_m': std_error,
            'min_error_m': np.min(errors),
            'max_error_m': np.max(errors),
            'num_points': num_eval_points
        },
        'evaluation_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    if has_uncertainty:
        results['uncertainties'] = uncertainties
        results['statistics']['mean_uncertainty_m'] = mean_uncertainty
        results['statistics']['uncertainty_cov'] = uncertainty_cov

    return results


def create_evaluation_animation(eval_results: dict,
                                save_path: str = "images/evaluation_flight_path.gif",
                                fps: int = 2):
    """Create animated GIF showing the evaluation images with aircraft perspective."""
    print(f"\n🎬 Creating Evaluation Animation")

    terrain_tiles = eval_results['terrain_tiles']
    trajectory_coords = eval_results['trajectory_coords']
    predictions = eval_results['predictions']
    errors = eval_results['errors']
    headings = eval_results['headings']

    frames = []

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()
        small_font = font

    for i, (tile, coord, pred, error, heading) in enumerate(zip(
        terrain_tiles, trajectory_coords, predictions, errors, headings
    )):
        # Create PIL image from tile
        img = Image.fromarray(tile.astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # Add direction indicator at center
        center_x, center_y = 112, 112
        arrow_length = 30

        # Draw aircraft forward arrow (always pointing up in aircraft perspective)
        draw.polygon([
            (center_x, center_y - arrow_length),
            (center_x - 8, center_y - arrow_length + 15),
            (center_x + 8, center_y - arrow_length + 15)
        ], fill='yellow', outline='black')

        # Add text overlay with info
        pixel_coord = coord * 7500
        pred_pixel = pred * 7500

        info_text = [
            f"Frame {i+1}/{len(terrain_tiles)}",
            f"Position: ({pixel_coord[0]:.0f}, {pixel_coord[1]:.0f})",
            f"Heading: {heading:.0f}°",
            f"Prediction: ({pred_pixel[0]:.0f}, {pred_pixel[1]:.0f})",
            f"Error: {error:.0f}m"
        ]

        y_offset = 10
        for line in info_text:
            # Draw text background
            bbox = draw.textbbox((10, y_offset), line, font=small_font)
            draw.rectangle(bbox, fill='black')
            draw.text((10, y_offset), line, fill='yellow', font=small_font)
            y_offset += 15

        # Add aircraft forward label
        draw.text((center_x - 20, center_y - arrow_length - 20),
                 "FWD", fill='yellow', font=font, stroke_width=1, stroke_fill='black')

        frames.append(img)

    # Save as animated GIF
    frame_duration = int(1000 / fps)
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0
    )

    print(f"✅ Evaluation animation saved: {save_path}")
    print(f"   Frames: {len(frames)}, Duration: {frame_duration}ms/frame ({fps} fps)")


def create_trajectory_visualization(eval_results: dict,
                                   save_path: str = "images/navigation_flight_trajectory.png"):
    """Create trajectory visualization using standard format."""
    print(f"\n🎨 Creating Trajectory Visualization")

    # Use PredictionVisualizer for consistent format
    visualizer = PredictionVisualizer()
    visualizer.load_satellite_map("../../data/boneyard/davis_monthan_stitched_map.jpg")

    # Subsample points for visualization if requested
    viz_points = eval_results.get('viz_points', len(eval_results['predictions']))
    total_points = len(eval_results['predictions'])

    if viz_points < total_points:
        # Sample evenly across all evaluated points
        viz_indices = np.linspace(0, total_points - 1, viz_points).astype(int)
        print(f"  Showing {viz_points} of {total_points} evaluated points")
    else:
        viz_indices = np.arange(total_points)

    # Convert coordinates to pixel coordinates (subsampled for visualization)
    true_coords_pixels = eval_results['trajectory_coords'][viz_indices] * 7500
    pred_coords_pixels = eval_results['predictions'][viz_indices] * 7500
    viz_errors = eval_results['errors'][viz_indices]

    # Create flight_results dict
    flight_results = {
        'ground_truth': true_coords_pixels,
        'predictions': pred_coords_pixels,
        'errors': viz_errors,
        'mean_error': eval_results['statistics']['mean_error_m']  # Keep full statistics
    }

    # Add uncertainties if available (also subsampled)
    if eval_results.get('has_uncertainty') and 'uncertainties' in eval_results:
        flight_results['uncertainties'] = eval_results['uncertainties'][viz_indices]

    # Use the standard visualization format
    visualizer.create_flight_path_viz(
        flight_results,
        save_path=save_path
    )

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate augmented model with live generation")
    parser.add_argument("--model", default="artifacts/model_20251008_121628.pth",
                       help="Path to trained model")
    parser.add_argument("--flight", default="main_evaluation",
                       help="Flight path name")
    parser.add_argument("--points", type=int, default=100,
                       help="Number of evaluation points")
    parser.add_argument("--viz-points", type=int, default=20,
                       help="Number of points to show in visualization")
    parser.add_argument("--fps", type=int, default=2,
                       help="Animation frame rate")
    parser.add_argument("--calibration", default=None,
                       help="Path to calibration data (optional)")
    args = parser.parse_args()

    print(f"📊 Augmented Model Evaluation (Live Generation)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Flight: {args.flight}")
    print(f"Evaluation points: {args.points}")
    print(f"Visualization points: {args.viz_points}")

    # Evaluate model with live generation
    eval_results = evaluate_model_with_live_generation(
        args.model, args.flight, args.points, args.viz_points, args.calibration
    )

    # Create evaluation animation
    create_evaluation_animation(
        eval_results,
        "images/evaluation_flight_path.gif",
        fps=args.fps
    )

    # Create trajectory visualization
    trajectory_viz_path = create_trajectory_visualization(
        eval_results, "images/navigation_flight_trajectory.png"
    )

    # Save evaluation results
    results_path = f"artifacts/live_evaluation_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(eval_results, f)

    print(f"\n✅ Evaluation Complete!")
    print(f"  Results: {results_path}")
    print(f"  Trajectory visualization: {trajectory_viz_path}")
    print(f"  Animation: images/evaluation_flight_path.gif")
    print(f"  Mean error: {eval_results['statistics']['mean_error_m']:.1f}m")

    return eval_results


if __name__ == "__main__":
    results = main()
