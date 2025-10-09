#!/usr/bin/env python3
"""
Evaluate terrain difficulty uncertainty model.
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from improved_uncertainty_models import BasicModelWithTerrainDifficulty
from navigation.terrain_window import TerrainWindow
from navigation.flight_config import FlightPathConfig
from navigation.utils import get_device


def evaluate_terrain_difficulty(model_path: str, num_points: int = 20):
    """Evaluate terrain difficulty uncertainty model."""
    print(f"🎯 Evaluating Terrain Difficulty Uncertainty Model")
    print("=" * 60)

    device = get_device()
    model = BasicModelWithTerrainDifficulty()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # Get flight path
    terrain_window = TerrainWindow()
    flight_path = FlightPathConfig.get_flight_path("main_evaluation")
    flight_coords = FlightPathConfig.create_flight_coordinates(flight_path)

    indices = np.linspace(0, len(flight_coords) - 1, num_points).astype(int)

    predictions = []
    uncertainties = []
    difficulties = []
    errors = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            coord = flight_coords[idx]
            pixel_x = coord[0] * 7500
            pixel_y = coord[1] * 7500

            # Calculate heading
            if idx < len(flight_coords) - 1:
                dx = flight_coords[idx + 1][0] - coord[0]
                dy = flight_coords[idx + 1][1] - coord[1]
                heading = np.degrees(np.arctan2(dy, dx))
            else:
                heading = 0

            # Extract terrain
            terrain_tile = terrain_window.extract_model_input(
                pixel_x, pixel_y, aircraft_heading=heading
            )

            # Normalize
            tile_tensor = torch.from_numpy(terrain_tile).float().permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            tile_tensor = (tile_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
            tile_tensor = tile_tensor.unsqueeze(0).to(device)

            # Predict
            pred_coord, pred_log_var, pred_difficulty = model(tile_tensor)
            pred_coord = pred_coord.cpu().numpy()[0]
            log_var = pred_log_var.cpu().numpy()[0, 0]
            difficulty = pred_difficulty.cpu().numpy()[0, 0]

            predictions.append(pred_coord)

            # Calculate uncertainty
            std_dev = np.sqrt(np.exp(log_var))
            unc_m = std_dev * 7500 * 10
            uncertainties.append(unc_m)
            difficulties.append(difficulty)

            # Error
            error_normalized = np.linalg.norm(pred_coord - coord)
            error_meters = error_normalized * 7500 * 10
            errors.append(error_meters)

            if (i + 1) % 5 == 0:
                conf_level = "HIGH" if unc_m < 3000 else "MEDIUM" if unc_m < 5000 else "LOW"
                diff_level = "EASY" if difficulty < 0.4 else "MEDIUM" if difficulty < 0.6 else "HARD"
                print(f"  {i+1}/{num_points} | Error: {error_meters:5.0f}m | "
                      f"Unc: {unc_m:5.0f}m ({conf_level}) | "
                      f"Difficulty: {difficulty:.2f} ({diff_level})")

    uncertainties = np.array(uncertainties)
    difficulties = np.array(difficulties)
    errors = np.array(errors)

    print(f"\n✅ Evaluation Complete!")
    print(f"\n  Error Statistics:")
    print(f"    Mean: {np.mean(errors):.1f}m")
    print(f"    Range: {np.min(errors):.1f}m to {np.max(errors):.1f}m")

    print(f"\n  Uncertainty:")
    print(f"    Mean: {np.mean(uncertainties):.1f}m")
    print(f"    Range: {np.min(uncertainties):.1f}m to {np.max(uncertainties):.1f}m")
    print(f"    CoV: {100*np.std(uncertainties)/np.mean(uncertainties):.1f}%")

    print(f"\n  Terrain Difficulty:")
    print(f"    Mean: {np.mean(difficulties):.3f}")
    print(f"    Range: {np.min(difficulties):.3f} to {np.max(difficulties):.3f}")
    print(f"    CoV: {100*np.std(difficulties)/np.mean(difficulties):.1f}%")

    print(f"\n  Comparison to Best Scalar Model (12.3% CoV, 48m error):")
    unc_cov = 100*np.std(uncertainties)/np.mean(uncertainties)
    if unc_cov > 20:
        print(f"    ✅ MAJOR IMPROVEMENT - {unc_cov:.1f}% variation!")
    elif unc_cov > 15:
        print(f"    ✅ GOOD IMPROVEMENT - {unc_cov:.1f}% variation")
    elif unc_cov > 12.3:
        print(f"    ⚠️ SLIGHT IMPROVEMENT - {unc_cov:.1f}% variation")
    else:
        print(f"    ❌ NO IMPROVEMENT - {unc_cov:.1f}% variation")
        print(f"    Coordinate accuracy also worse: 176m vs 48m")

    # Correlations
    unc_corr = np.corrcoef(errors, uncertainties)[0, 1]
    diff_corr = np.corrcoef(errors, difficulties)[0, 1]

    print(f"\n  Error-Uncertainty Correlation: {unc_corr:.3f}")
    print(f"  Error-Difficulty Correlation: {diff_corr:.3f}")

    if diff_corr > 0.3:
        print(f"    ✅ Difficulty head learned well (predicts harder terrain = more error)")
    elif diff_corr > 0:
        print(f"    ⚠️ Difficulty head learned weakly")
    else:
        print(f"    ❌ Difficulty head learned backwards")

    return {
        'uncertainties': uncertainties,
        'difficulties': difficulties,
        'errors': errors,
        'predictions': np.array(predictions)
    }


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/model_20251008_224818.pth"
    evaluate_terrain_difficulty(model_path, num_points=20)
