"""
NavigationPredictor: Main prediction interface for terrain navigation
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union, List
import pickle

from .utils import (
    get_device, setup_model_device, batch_predict,
    normalize_coordinates, denormalize_coordinates,
    calculate_pixel_error, CoordinateSystem
)


# Import the new architecture from training script
import sys
from pathlib import Path
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from train_corridor_navigation_model import NavigationDenseNet
except ImportError:
    # Define the trained corridor model architecture locally if import fails
    class NavigationDenseNet(nn.Module):
        def __init__(self, pretrained=True):
            super(NavigationDenseNet, self).__init__()
            densenet = models.densenet121(pretrained=pretrained)
            self.features = densenet.features

            # Simplified classifier (the EXACT architecture that was trained: 1024->64->2)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Dropout(0.8), nn.Linear(1024, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                nn.Dropout(0.5), nn.Linear(64, 2), nn.Sigmoid()
            )

        def forward(self, x):
            features = self.features(x)
            return self.classifier(features)

class EnsembleModel5_Deep(nn.Module):
    """
    DenseNet-based champion model for terrain navigation.
    This reproduces the champion architecture from train_ensemble_models.py
    """
    def __init__(self):
        super(EnsembleModel5_Deep, self).__init__()

        # DenseNet121 backbone with pre-trained weights
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features

        # Match the actual saved model architecture
        # Based on the state dict analysis, the classifier has these layers:
        # 3, 4 (BN), 7, 10, 13, 16
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),         # classifier.3
            nn.BatchNorm1d(512),          # classifier.4
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),          # classifier.7
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),          # classifier.10
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),           # classifier.13
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),             # classifier.16
            nn.Sigmoid()  # Output normalized coordinates
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class NavigationPredictor:
    """
    Main interface for terrain navigation predictions using deep learning.

    This class provides a clean, DRY interface for loading trained models
    and performing single or batch predictions on terrain images.
    """

    def __init__(self, model_path: Optional[str] = None, map_size: Tuple[int, int] = (7500, 7500)):
        """
        Initialize the NavigationPredictor.

        Args:
            model_path: Path to trained model file (.pth)
            map_size: Size of the full satellite map (width, height)
        """
        self.map_size = map_size
        self.coord_system = CoordinateSystem(map_size)
        self.model = None
        self.device = None
        self.transform = None

        # Setup image preprocessing transforms
        self._setup_transforms()

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: str, model_class: Optional[nn.Module] = None) -> None:
        """
        Load a trained navigation model.

        Args:
            model_path: Path to the model file (.pth)
            model_class: Model class to use (defaults to EnsembleModel5_Deep)
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Use default champion model if no class specified
        if model_class is None:
            model_class = EnsembleModel5_Deep

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Extract state dict (handle both direct state_dict and checkpoints)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Auto-detect architecture based on model path and structure
        if model_class == EnsembleModel5_Deep:
            # Check if this is the corridor model based on filename
            if 'corridor' in str(model_path):
                print("   Detected corridor NavigationDenseNet architecture")
                model_class = NavigationDenseNet
            else:
                print("   Using EnsembleModel5_Deep architecture")

        # Initialize and load model
        self.model = model_class()
        self.model.load_state_dict(state_dict)

        # Setup device
        self.model, self.device = setup_model_device(self.model)

        print(f"✅ Model loaded successfully: {Path(model_path).name}")
        print(f"   Device: {self.device}")

    def predict(self, terrain_image: Union[np.ndarray, Image.Image, str],
                return_confidence: bool = False) -> Union[Tuple[float, float], Tuple[Tuple[float, float], float]]:
        """
        Predict coordinates for a single terrain image.

        Args:
            terrain_image: Input terrain image (array, PIL Image, or file path)
            return_confidence: Whether to return prediction confidence

        Returns:
            Predicted (x, y) coordinates in pixels, optionally with confidence
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Preprocess input
        if isinstance(terrain_image, str):
            terrain_image = Image.open(terrain_image)
        elif isinstance(terrain_image, np.ndarray):
            terrain_image = Image.fromarray(terrain_image.astype(np.uint8))

        # Transform and predict
        input_tensor = self.transform(terrain_image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            normalized_pred = self.model(input_tensor).cpu().numpy()[0]

        # Check if model outputs are already in pixel space (corridor models) or normalized space
        if np.any(normalized_pred > 1.0):
            # Model outputs pixel coordinates directly (corridor models)
            pixel_coords = normalized_pred
        else:
            # Model outputs normalized coordinates (legacy models)
            pixel_coords = self.coord_system.denormalize(normalized_pred.reshape(1, -1))[0]

        if return_confidence:
            # Simple confidence based on prediction certainty
            # Higher values closer to 0.5 indicate lower confidence
            confidence = 1.0 - np.mean(np.abs(normalized_pred - 0.5))
            return tuple(pixel_coords), confidence
        else:
            return tuple(pixel_coords)

    def predict_batch(self, terrain_images: List[Union[np.ndarray, Image.Image]],
                     batch_size: int = 32) -> np.ndarray:
        """
        Predict coordinates for multiple terrain images.

        Args:
            terrain_images: List of terrain images
            batch_size: Batch size for processing

        Returns:
            Array of predicted coordinates in pixels
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Convert all images to numpy arrays
        image_arrays = []
        for img in terrain_images:
            if isinstance(img, Image.Image):
                img = np.array(img)
            elif isinstance(img, str):
                img = np.array(Image.open(img))
            image_arrays.append(img)

        # Use batch prediction utility
        normalized_preds = batch_predict(self.model, np.array(image_arrays),
                                       self.device, batch_size)

        # Check if model outputs are already in pixel space (corridor models) or normalized space
        if np.any(normalized_preds > 1.0):
            # Model outputs pixel coordinates directly (corridor models)
            pixel_coords = normalized_preds
        else:
            # Model outputs normalized coordinates (legacy models)
            pixel_coords = self.coord_system.denormalize(normalized_preds)

        return pixel_coords

    def predict_flight_path(self, flight_coordinates: np.ndarray,
                          terrain_extractor, batch_size: int = 32) -> dict:
        """
        Predict positions along an entire flight path.

        Args:
            flight_coordinates: Array of flight path coordinates (normalized)
            terrain_extractor: TerrainExtractor instance for getting images
            batch_size: Batch size for processing

        Returns:
            Dictionary containing predictions, ground truth, and errors
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Extract terrain images for each flight point
        terrain_images = []
        for coord in flight_coordinates:
            pixel_coord = self.coord_system.denormalize(coord.reshape(1, -1))[0]
            terrain_img = terrain_extractor.extract_tile(
                int(pixel_coord[0]), int(pixel_coord[1]), size=224
            )
            terrain_images.append(terrain_img)

        # Batch predict all positions
        predictions = self.predict_batch(terrain_images, batch_size)

        # Convert ground truth to pixel coordinates
        ground_truth = self.coord_system.denormalize(flight_coordinates)

        # Calculate errors
        errors = self.coord_system.calculate_error(predictions, ground_truth)

        return {
            'predictions': predictions,
            'ground_truth': ground_truth,
            'errors': errors,
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'flight_path': flight_coordinates,
            'terrain_images': terrain_images
        }

    def evaluate_accuracy(self, test_coordinates: np.ndarray,
                         terrain_extractor, batch_size: int = 32) -> dict:
        """
        Evaluate model accuracy on test coordinates.

        Args:
            test_coordinates: Array of test coordinates (normalized)
            terrain_extractor: TerrainExtractor for getting ground truth images
            batch_size: Batch size for processing

        Returns:
            Evaluation results dictionary
        """
        results = self.predict_flight_path(test_coordinates, terrain_extractor, batch_size)

        # Add additional statistics
        results.update({
            'std_error': np.std(results['errors']),
            'min_error': np.min(results['errors']),
            'max_error': np.max(results['errors']),
            'percentile_90': np.percentile(results['errors'], 90),
            'percentile_95': np.percentile(results['errors'], 95)
        })

        return results

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {"status": "No model loaded"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "status": "Model loaded",
            "architecture": self.model.__class__.__name__,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "map_size": self.map_size
        }

    @staticmethod
    def load_champion_model(artifacts_dir: str = "artifacts") -> 'NavigationPredictor':
        """
        Convenience method to load the champion DenseNet model.

        Args:
            artifacts_dir: Directory containing model artifacts

        Returns:
            NavigationPredictor with champion model loaded
        """
        # Always try corridor model first (most recent training)
        alternative_paths = [
            Path(artifacts_dir) / "corridor_navigation_model_best.pth",
            Path(artifacts_dir) / "ensemble_densenet_deep_model.pth",
            Path(artifacts_dir) / "champion_refined_model.pth",
            Path(artifacts_dir) / "densenet_deep_model.pth"
        ]

        model_path = None
        for alt_path in alternative_paths:
            if alt_path.exists():
                model_path = alt_path
                break

            raise FileNotFoundError(f"No navigation model found in {artifacts_dir}")

        predictor = NavigationPredictor()
        predictor.load_model(str(model_path))
        return predictor