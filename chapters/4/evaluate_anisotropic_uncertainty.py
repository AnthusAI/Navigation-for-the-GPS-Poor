#!/usr/bin/env python3
"""
Evaluate anisotropic uncertainty model (separate x/y uncertainties).
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from improved_uncertainty_models import BasicModelWithAnisotropicUncertainty
from navigation.terrain_window import TerrainWindow
from navigation.flight_config import FlightPathConfig
from navigation.utils import get_device


def evaluate_anisotropic(model_path: str, num_points: int = 20):
    """Evaluate anisotropic uncertainty model."""
    print(f"🎯 Evaluating Anisotropic Uncertainty Model")
    print("=" * 60)

    device = get_device()
    model = BasicModelWithAnisotropicUncertainty()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # Get flight path
    terrain_window = TerrainWindow()
    flight_path = FlightPathConfig.get_flight_path("main_evaluation")
    flight_coords = FlightPathConfig.create_flight_coordinates(flight_path)

    indices = np.linspace(0, len(flight_coords) - 1, num_points).astype(int)

    predictions = []
    x_uncertainties = []
    y_uncertainties = []
    total_uncertainties = []
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
            pred_coord, pred_log_vars = model(tile_tensor)
            pred_coord = pred_coord.cpu().numpy()[0]
            log_var_x = pred_log_vars.cpu().numpy()[0, 0]
            log_var_y = pred_log_vars.cpu().numpy()[0, 1]

            predictions.append(pred_coord)

            # Calculate uncertainties
            std_x = np.sqrt(np.exp(log_var_x))
            std_y = np.sqrt(np.exp(log_var_y))
            unc_x_m = std_x * 7500 * 10
            unc_y_m = std_y * 7500 * 10
            total_unc_m = np.sqrt(unc_x_m**2 + unc_y_m**2)

            x_uncertainties.append(unc_x_m)
            y_uncertainties.append(unc_y_m)
            total_uncertainties.append(total_unc_m)

            # Error
            error_normalized = np.linalg.norm(pred_coord - coord)
            error_meters = error_normalized * 7500 * 10
            errors.append(error_meters)

            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{num_points} | Error: {error_meters:4.0f}m | "
                      f"Unc(x): {unc_x_m:4.0f}m | Unc(y): {unc_y_m:4.0f}m | "
                      f"Total: {total_unc_m:4.0f}m")

    x_uncertainties = np.array(x_uncertainties)
    y_uncertainties = np.array(y_uncertainties)
    total_uncertainties = np.array(total_uncertainties)
    errors = np.array(errors)

    print(f"\n✅ Evaluation Complete!")
    print(f"\n  Error Statistics:")
    print(f"    Mean: {np.mean(errors):.1f}m")
    print(f"    Range: {np.min(errors):.1f}m to {np.max(errors):.1f}m")

    print(f"\n  X-Uncertainty:")
    print(f"    Mean: {np.mean(x_uncertainties):.1f}m")
    print(f"    Range: {np.min(x_uncertainties):.1f}m to {np.max(x_uncertainties):.1f}m")
    print(f"    CoV: {100*np.std(x_uncertainties)/np.mean(x_uncertainties):.1f}%")

    print(f"\n  Y-Uncertainty:")
    print(f"    Mean: {np.mean(y_uncertainties):.1f}m")
    print(f"    Range: {np.min(y_uncertainties):.1f}m to {np.max(y_uncertainties):.1f}m")
    print(f"    CoV: {100*np.std(y_uncertainties)/np.mean(y_uncertainties):.1f}%")

    print(f"\n  Total Uncertainty:")
    print(f"    Mean: {np.mean(total_uncertainties):.1f}m")
    print(f"    Range: {np.min(total_uncertainties):.1f}m to {np.max(total_uncertainties):.1f}m")
    print(f"    CoV: {100*np.std(total_uncertainties)/np.mean(total_uncertainties):.1f}%")

    print(f"\n  Comparison to Scalar Model (12.3% CoV):")
    total_cov = 100*np.std(total_uncertainties)/np.mean(total_uncertainties)
    if total_cov > 20:
        print(f"    ✅ MAJOR IMPROVEMENT - {total_cov:.1f}% variation!")
    elif total_cov > 15:
        print(f"    ✅ GOOD IMPROVEMENT - {total_cov:.1f}% variation")
    elif total_cov > 12.3:
        print(f"    ⚠️ SLIGHT IMPROVEMENT - {total_cov:.1f}% variation")
    else:
        print(f"    ❌ NO IMPROVEMENT - {total_cov:.1f}% variation (worse or same)")

    # Correlation
    correlation = np.corrcoef(errors, total_uncertainties)[0, 1]
    print(f"\n  Error-Uncertainty Correlation: {correlation:.3f}")
    if correlation > 0.3:
        print(f"    ✅ EXCELLENT - Model predicts higher uncertainty for larger errors!")
    elif correlation > 0.1:
        print(f"    ⚠️ MODERATE - Positive correlation but weak")
    elif correlation > -0.1:
        print(f"    ⚠️ MINIMAL - Near zero correlation")
    else:
        print(f"    ❌ BACKWARDS - Negative correlation (wrong direction)")

    return {
        'x_uncertainties': x_uncertainties,
        'y_uncertainties': y_uncertainties,
        'total_uncertainties': total_uncertainties,
        'errors': errors,
        'predictions': np.array(predictions)
    }


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/model_20251008_214521.pth"
    evaluate_anisotropic(model_path, num_points=20)
