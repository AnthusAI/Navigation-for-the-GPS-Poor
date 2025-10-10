#!/usr/bin/env python3
"""
Calibrate uncertainty predictions using validation set.

Uses the held-aside validation data to map predicted uncertainties
to actual error magnitudes.
"""

import sys
import torch
import numpy as np
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from train_model import BasicModel
from navigation.utils import get_device


def calibrate_uncertainty_model(model_path: str, dataset_path: str):
    """
    Calibrate uncertainty predictions using validation set.

    Args:
        model_path: Path to trained model
        dataset_path: Path to training dataset (will use 20% validation split)

    Returns:
        Dictionary with calibration parameters
    """
    print(f"🎯 Calibrating Uncertainty Model")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Dataset: {dataset_path}")

    device = get_device()

    # Load model
    model = BasicModel(predict_uncertainty=True)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    tiles = dataset['tiles']
    coordinates = dataset['coordinates']

    # Use same 20% validation split as training (with same seed)
    from torch.utils.data import random_split
    val_size = int(0.2 * len(tiles))
    train_size = len(tiles) - val_size

    indices = list(range(len(tiles)))
    torch.manual_seed(42)
    train_indices, val_indices = random_split(
        indices, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_indices = list(val_indices)
    print(f"\n  Validation samples: {len(val_indices)}")

    # Normalize tiles
    from torchvision import transforms
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Run predictions on validation set
    predicted_uncertainties = []
    actual_errors = []

    print(f"\n  Running predictions on validation set...")

    with torch.no_grad():
        for i, idx in enumerate(val_indices):
            tile = tiles[idx]
            coord = coordinates[idx]

            # Normalize
            tile_tensor = normalize(tile).unsqueeze(0).to(device)

            # Predict
            pred_coord, pred_log_var = model(tile_tensor)
            pred_coord = pred_coord.cpu().numpy()[0]
            log_var = pred_log_var.cpu().numpy()[0, 0]

            # Calculate predicted uncertainty (in normalized coordinates)
            std_dev = np.sqrt(np.exp(log_var))
            predicted_uncertainties.append(std_dev)

            # Calculate actual error (in normalized coordinates)
            error = np.linalg.norm(pred_coord - coord)
            actual_errors.append(error)

            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{len(val_indices)}")

    predicted_uncertainties = np.array(predicted_uncertainties)
    actual_errors = np.array(actual_errors)

    # Calculate calibration metrics
    print(f"\n📊 Calibration Analysis:")
    print(f"  Predicted uncertainty (std):")
    print(f"    Mean: {np.mean(predicted_uncertainties):.6f}")
    print(f"    Range: {np.min(predicted_uncertainties):.6f} to {np.max(predicted_uncertainties):.6f}")

    print(f"\n  Actual errors:")
    print(f"    Mean: {np.mean(actual_errors):.6f} ({np.mean(actual_errors) * 7500 * 10:.1f}m)")
    print(f"    Range: {np.min(actual_errors):.6f} to {np.max(actual_errors):.6f}")
    print(f"           ({np.min(actual_errors) * 7500 * 10:.1f}m to {np.max(actual_errors) * 7500 * 10:.1f}m)")

    # Calculate calibration factor
    # We want: predicted_uncertainty * calibration_factor ≈ actual_error
    calibration_factor = np.mean(actual_errors) / np.mean(predicted_uncertainties)

    print(f"\n  📏 Calibration Factor: {calibration_factor:.4f}")
    print(f"     (multiply predicted uncertainty by this to get expected error)")

    # Check calibration quality
    calibrated_uncertainties = predicted_uncertainties * calibration_factor

    # How many predictions fall within 1-sigma?
    within_1sigma = np.sum(actual_errors <= calibrated_uncertainties)
    within_1sigma_pct = 100 * within_1sigma / len(actual_errors)

    # How many within 2-sigma?
    within_2sigma = np.sum(actual_errors <= 2 * calibrated_uncertainties)
    within_2sigma_pct = 100 * within_2sigma / len(actual_errors)

    print(f"\n  📈 Calibration Quality:")
    print(f"    Within 1σ: {within_1sigma}/{len(actual_errors)} ({within_1sigma_pct:.1f}%) [expected ~68%]")
    print(f"    Within 2σ: {within_2sigma}/{len(actual_errors)} ({within_2sigma_pct:.1f}%) [expected ~95%]")

    # Correlation between predicted uncertainty and actual error
    correlation = np.corrcoef(predicted_uncertainties, actual_errors)[0, 1]
    print(f"    Correlation: {correlation:.3f}")

    # Save calibration results
    calibration_data = {
        'model_path': model_path,
        'dataset_path': dataset_path,
        'calibration_factor': calibration_factor,
        'validation_size': len(val_indices),
        'mean_predicted_uncertainty': float(np.mean(predicted_uncertainties)),
        'mean_actual_error': float(np.mean(actual_errors)),
        'within_1sigma_pct': float(within_1sigma_pct),
        'within_2sigma_pct': float(within_2sigma_pct),
        'correlation': float(correlation),
        'predicted_uncertainties': predicted_uncertainties,
        'actual_errors': actual_errors
    }

    output_path = 'artifacts/uncertainty_calibration.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(calibration_data, f)

    print(f"\n✅ Calibration data saved: {output_path}")
    print(f"\n💡 Usage:")
    print(f"   To convert model predictions to expected error:")
    print(f"   expected_error = sqrt(exp(log_var)) * {calibration_factor:.4f} * 7500 * 10  # meters")

    return calibration_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--dataset", required=True, help="Path to training dataset")

    args = parser.parse_args()

    calibrate_uncertainty_model(args.model, args.dataset)
