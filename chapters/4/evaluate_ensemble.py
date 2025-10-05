#!/usr/bin/env python3
"""
Evaluate ensemble of models for ultra-high precision navigation.
Combines predictions from multiple diverse models.
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

# Import model architectures
from train_ensemble_models import (
    EnsembleModel1_EfficientNet, EnsembleModel2_ResNet50,
    EnsembleModel3_MultiScale, EnsembleModel4_Attention, EnsembleModel5_Deep
)


class EnsemblePredictor:
    """Combines predictions from multiple trained models."""

    def __init__(self, device):
        self.device = device
        self.models = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_models(self):
        """Load all available trained ensemble models."""
        model_configs = [
            (EnsembleModel1_EfficientNet, "efficientnet_b3", "EfficientNet B3"),
            (EnsembleModel2_ResNet50, "resnet50_attention", "ResNet50 Attention"),
            (EnsembleModel3_MultiScale, "multiscale_cnn", "MultiScale CNN"),
            (EnsembleModel4_Attention, "heavy_attention", "Heavy Attention"),
            (EnsembleModel5_Deep, "densenet_deep", "DenseNet Deep"),
        ]

        loaded_count = 0
        for model_class, model_key, model_name in model_configs:
            model_path = f'artifacts/ensemble_{model_key}_model.pth'
            if os.path.exists(model_path):
                try:
                    model = model_class().to(self.device)
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.eval()
                    self.models[model_name] = model
                    loaded_count += 1
                    print(f"✅ Loaded {model_name}")
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
            else:
                print(f"⚠️ Model file not found: {model_path}")

        print(f"\\nLoaded {loaded_count} ensemble models")
        return loaded_count > 0

    def predict_single_image(self, image):
        """Get ensemble prediction for a single image."""
        if not self.models:
            raise ValueError("No models loaded!")

        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        predictions = []
        with torch.no_grad():
            for model_name, model in self.models.items():
                pred = model(image_tensor).cpu().numpy()[0]
                predictions.append(pred)

        # Ensemble strategies
        predictions = np.array(predictions)

        # Simple average
        ensemble_mean = np.mean(predictions, axis=0)

        # Weighted average (could be improved with model confidence)
        # For now, equal weights
        weights = np.ones(len(predictions)) / len(predictions)
        ensemble_weighted = np.average(predictions, axis=0, weights=weights)

        # Median (robust to outliers)
        ensemble_median = np.median(predictions, axis=0)

        return {
            'mean': ensemble_mean,
            'weighted': ensemble_weighted,
            'median': ensemble_median,
            'individual': predictions,
            'model_names': list(self.models.keys())
        }


def evaluate_ensemble_on_flight_path():
    """Evaluate ensemble on the flight path."""
    print("ENSEMBLE EVALUATION ON FLIGHT PATH")
    print("=" * 50)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device.type.upper()}")

    # Create ensemble predictor
    ensemble = EnsemblePredictor(device)
    if not ensemble.load_models():
        print("❌ No ensemble models found! Run train_ensemble_models.py first.")
        return

    # Flight path configuration
    map_path = '../../data/boneyard/davis_monthan_stitched_map.jpg'
    start_coord = (5500, 4500)
    end_coord = (4167, 4167)
    tile_size = (1200, 675)
    zoom_factor = 4
    num_frames = 150

    # Load map
    full_map = Image.open(map_path).convert('RGB')
    map_width, map_height = full_map.size

    print(f"Flight path: {start_coord} → {end_coord}")
    print(f"Processing {num_frames} frames...")

    # Generate flight path
    path_x = np.linspace(start_coord[0], end_coord[0], num_frames)
    path_y = np.linspace(start_coord[1], end_coord[1], num_frames)

    crop_width = tile_size[0] // zoom_factor
    crop_height = tile_size[1] // zoom_factor

    # Storage for results
    gt_coords = []
    ensemble_predictions = {
        'mean': [],
        'weighted': [],
        'median': []
    }
    individual_predictions = {name: [] for name in ensemble.models.keys()}

    print("\\nRunning ensemble inference...")
    for i in tqdm(range(num_frames), desc="Processing frames"):
        cam_x, cam_y = path_x[i], path_y[i]

        # Create input image
        left = int(cam_x - crop_width / 2)
        top = int(cam_y - crop_height / 2)
        right = left + crop_width
        bottom = top + crop_height

        frame = full_map.crop((left, top, right, bottom))
        frame = frame.resize(tile_size, Image.LANCZOS)

        # Get ensemble prediction
        result = ensemble.predict_single_image(frame)

        # Store ground truth
        gt_coords.append([cam_x, cam_y])

        # Store ensemble predictions
        for method in ['mean', 'weighted', 'median']:
            pred_norm = result[method]
            pred_x = pred_norm[0] * map_width
            pred_y = pred_norm[1] * map_height
            ensemble_predictions[method].append([pred_x, pred_y])

        # Store individual predictions
        for j, model_name in enumerate(result['model_names']):
            pred_norm = result['individual'][j]
            pred_x = pred_norm[0] * map_width
            pred_y = pred_norm[1] * map_height
            individual_predictions[model_name].append([pred_x, pred_y])

    # Convert to numpy arrays
    gt_coords = np.array(gt_coords)
    for method in ensemble_predictions:
        ensemble_predictions[method] = np.array(ensemble_predictions[method])
    for model_name in individual_predictions:
        individual_predictions[model_name] = np.array(individual_predictions[model_name])

    # Calculate errors
    def calculate_errors(predictions, ground_truth):
        return np.sqrt(np.sum((predictions - ground_truth)**2, axis=1))

    print("\\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)

    # Individual model performance
    print("\\nIndividual Model Performance:")
    individual_errors = {}
    for model_name, predictions in individual_predictions.items():
        errors = calculate_errors(predictions, gt_coords)
        mean_error = np.mean(errors)
        individual_errors[model_name] = mean_error
        print(f"   {model_name:<20}: {mean_error:6.1f}px")

    # Ensemble performance
    print("\\nEnsemble Performance:")
    ensemble_errors = {}
    for method, predictions in ensemble_predictions.items():
        errors = calculate_errors(predictions, gt_coords)
        mean_error = np.mean(errors)
        ensemble_errors[method] = mean_error

        status = "🏆 CHAMPION" if mean_error < 30 else "🥇 EXCELLENT" if mean_error < 35 else "✅ GOOD"
        print(f"   {method.capitalize():<12}: {mean_error:6.1f}px {status}")

    # Comparison with previous best
    universal_cnn_error = 38.6
    baseline_error = 153.9
    best_ensemble_error = min(ensemble_errors.values())

    print(f"\\nComparison:")
    print(f"   Original Baseline:  {baseline_error:.1f}px")
    print(f"   Universal CNN:      {universal_cnn_error:.1f}px")
    print(f"   Best Ensemble:      {best_ensemble_error:.1f}px")

    if best_ensemble_error < universal_cnn_error:
        improvement = ((universal_cnn_error - best_ensemble_error) / universal_cnn_error) * 100
        total_improvement = ((baseline_error - best_ensemble_error) / baseline_error) * 100
        print(f"   🎉 Additional improvement: {improvement:.1f}%")
        print(f"   🎯 Total improvement: {total_improvement:.1f}%")

    # Find best ensemble method
    best_method = min(ensemble_errors.items(), key=lambda x: x[1])
    print(f"\\n🏆 Best ensemble method: {best_method[0]} ({best_method[1]:.1f}px)")

    # Save results
    results = {
        'ground_truth': gt_coords,
        'ensemble_predictions': ensemble_predictions,
        'individual_predictions': individual_predictions,
        'ensemble_errors': ensemble_errors,
        'individual_errors': individual_errors,
        'best_method': best_method[0],
        'best_error': best_method[1]
    }

    with open('artifacts/ensemble_flight_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\\n✅ Results saved to artifacts/ensemble_flight_results.pkl")

    # Detailed statistics for best method
    best_predictions = ensemble_predictions[best_method[0]]
    best_errors = calculate_errors(best_predictions, gt_coords)

    print(f"\\nDetailed statistics for {best_method[0]} ensemble:")
    print(f"   Mean error:    {np.mean(best_errors):.1f}px")
    print(f"   Median error:  {np.median(best_errors):.1f}px")
    print(f"   Std dev:       {np.std(best_errors):.1f}px")
    print(f"   Min error:     {np.min(best_errors):.1f}px")
    print(f"   Max error:     {np.max(best_errors):.1f}px")
    print(f"   95th percentile: {np.percentile(best_errors, 95):.1f}px")

    return results


if __name__ == "__main__":
    evaluate_ensemble_on_flight_path()