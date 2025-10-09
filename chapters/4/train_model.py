#!/usr/bin/env python3
"""
Simple command to train navigation models.
Usage: python train_model.py --data training_data.pkl --arch deep --epochs 20
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import json
from pathlib import Path
import sys
from datetime import datetime
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent))
from navigation.utils import get_device
from navigation.augmented_dataset import RobustNavigationDataset
from improved_uncertainty_models import (
    BasicModelWithAnisotropicUncertainty,
    BasicModelWithDeepUncertaintyHead,
    BasicModelWithTerrainDifficulty,
    anisotropic_uncertainty_loss,
    terrain_difficulty_loss
)

class SimpleDeepModel(nn.Module):
    """Lightweight model optimized for small diverse datasets with uncertainty."""

    def __init__(self, predict_uncertainty=False):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights

        self.predict_uncertainty = predict_uncertainty
        # Load DenseNet backbone
        backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = backbone.features

        if predict_uncertainty:
            # Shared features
            self.shared = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            # Coordinate head
            self.coord_head = nn.Linear(256, 2)
            # Uncertainty head
            self.uncertainty_head = nn.Linear(256, 1)
            # Initialize uncertainty head to predict moderate uncertainty
            nn.init.constant_(self.uncertainty_head.bias, -2.0)
            nn.init.normal_(self.uncertainty_head.weight, mean=0.0, std=0.01)
        else:
            # Original architecture
            self.regressor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2)
            )

    def forward(self, x):
        features = self.features(x)
        if self.predict_uncertainty:
            shared_features = self.shared(features)
            coords = self.coord_head(shared_features)
            log_var = self.uncertainty_head(shared_features)
            return coords, log_var
        else:
            return self.regressor(features)

class BasicModel(nn.Module):
    """Ultra-simple model for small datasets with uncertainty estimation."""

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
            # Initialize uncertainty head to predict moderate uncertainty
            # Start with log_var ≈ -2 (variance ≈ 0.135, std ≈ 0.37 in normalized coords)
            nn.init.constant_(self.uncertainty_head.bias, -2.0)
            nn.init.normal_(self.uncertainty_head.weight, mean=0.0, std=0.01)
        else:
            # Original architecture without uncertainty
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

class TerrainDataset(Dataset):
    def __init__(self, tiles, coordinates):
        self.tiles = tiles
        self.coordinates = coordinates
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.transform(self.tiles[idx])
        coord = torch.FloatTensor(self.coordinates[idx])
        return tile, coord

def load_data(data_path):
    """Load training dataset."""
    print(f"Loading data from {data_path}...")

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    tiles = dataset['tiles']
    coordinates = dataset['coordinates']

    print(f"  Samples: {len(tiles)}")
    print(f"  Coord range: {coordinates.min():.3f} to {coordinates.max():.3f}")

    return tiles, coordinates

def create_model(arch, predict_uncertainty=False, uncertainty_arch="scalar"):
    """Create model based on architecture choice."""
    if predict_uncertainty:
        # Uncertainty-enabled architectures
        if uncertainty_arch == "anisotropic":
            return BasicModelWithAnisotropicUncertainty()
        elif uncertainty_arch == "deep_head":
            return BasicModelWithDeepUncertaintyHead()
        elif uncertainty_arch == "terrain_difficulty":
            return BasicModelWithTerrainDifficulty()
        elif uncertainty_arch == "scalar":
            # Original scalar uncertainty
            if arch == "deep":
                return SimpleDeepModel(predict_uncertainty=True)
            else:
                return BasicModel(predict_uncertainty=True)
        else:
            raise ValueError(f"Unknown uncertainty architecture: {uncertainty_arch}")
    else:
        # Standard models without uncertainty
        if arch == "deep":
            return SimpleDeepModel(predict_uncertainty=False)
        elif arch == "basic":
            return BasicModel(predict_uncertainty=False)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

def multi_task_loss_with_uncertainty(pred_coords, pred_log_var, true_coords, coord_weight=1.0, uncertainty_weight=1.0):
    """
    Multi-task loss: prioritize coordinate accuracy, add uncertainty as auxiliary.

    Args:
        pred_coords: Predicted coordinates
        pred_log_var: Predicted log variance (uncertainty)
        true_coords: Ground truth coordinates
        coord_weight: Weight for coordinate MSE (default 1.0)
        uncertainty_weight: Weight for uncertainty term (default 0.01)

    Returns:
        total_loss, coord_loss, uncertainty_loss (for monitoring)
    """
    # Check for NaN in inputs
    if torch.isnan(pred_coords).any():
        print("ERROR: NaN in pred_coords!")
        return torch.tensor(1e6, device=pred_coords.device), torch.tensor(1e6, device=pred_coords.device), torch.tensor(0.0, device=pred_coords.device)

    if torch.isnan(pred_log_var).any():
        print("ERROR: NaN in pred_log_var!")
        return torch.tensor(1e6, device=pred_coords.device), torch.tensor(1e6, device=pred_coords.device), torch.tensor(0.0, device=pred_coords.device)

    # Main task: Coordinate prediction accuracy (MSE)
    coord_loss = torch.mean((pred_coords - true_coords) ** 2)

    # Safety check
    if torch.isnan(coord_loss):
        print("ERROR: NaN in coord_loss computation!")
        return torch.tensor(1e6, device=pred_coords.device), torch.tensor(1e6, device=pred_coords.device), torch.tensor(0.0, device=pred_coords.device)

    # Clamp log variance for stability
    pred_log_var = torch.clamp(pred_log_var, min=-5, max=5)

    # Auxiliary task: Uncertainty estimation
    # Detach pred_coords so uncertainty doesn't affect coordinate gradients
    squared_error = (pred_coords - true_coords) ** 2  # Removed detach to allow learning
    mse_per_sample = torch.mean(squared_error, dim=1, keepdim=True)

    # Uncertainty loss: encourage high uncertainty when errors are large
    precision = torch.exp(-pred_log_var)
    precision = torch.clamp(precision, min=0.01, max=100.0)

    # Loss that encourages uncertainty to match actual errors
    uncertainty_loss = torch.mean(precision * mse_per_sample + 0.5 * pred_log_var)

    # Combined loss with weighting
    total_loss = coord_weight * coord_loss + uncertainty_weight * uncertainty_loss

    return total_loss, coord_loss, uncertainty_loss

def train_model(model, train_loader, val_loader, epochs, lr, predict_uncertainty=False,
                uncertainty_arch="scalar", using_terrain_difficulty=False):
    """Train the model with robust early stopping."""
    device = get_device()
    model = model.to(device)

    print(f"Training on {device} for up to {epochs} epochs (with early stopping)")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Uncertainty estimation: {predict_uncertainty}")

    if not predict_uncertainty:
        criterion = nn.MSELoss()
    # Revert to original weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing for smoother convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    history = {'train_losses': [], 'val_losses': [], 'learning_rates': []}
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 8  # Revert to original aggressive stopping

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_coord_loss = 0.0  # Track coordinate loss separately
        for batch_tiles, batch_coords in train_loader:
            batch_tiles, batch_coords = batch_tiles.to(device), batch_coords.to(device)

            optimizer.zero_grad()

            if predict_uncertainty:
                model_outputs = model(batch_tiles)

                # Handle different output formats
                if using_terrain_difficulty:
                    pred_coords, pred_log_var, pred_difficulty = model_outputs
                    loss, coord_loss, unc_loss, diff_loss = terrain_difficulty_loss(
                        pred_coords, pred_log_var, pred_difficulty, batch_coords,
                        coord_weight=1.0, uncertainty_weight=1.0, difficulty_weight=0.5
                    )
                elif uncertainty_arch == "anisotropic":
                    pred_coords, pred_log_vars = model_outputs
                    loss, coord_loss, unc_loss = anisotropic_uncertainty_loss(
                        pred_coords, pred_log_vars, batch_coords,
                        coord_weight=1.0, uncertainty_weight=1.0
                    )
                else:
                    pred_coords, pred_log_var = model_outputs
                    loss, coord_loss, unc_loss = multi_task_loss_with_uncertainty(
                        pred_coords, pred_log_var, batch_coords,
                        coord_weight=1.0,
                        uncertainty_weight=1.0  # Maximum weight: uncertainty as important as coordinates
                    )
                train_coord_loss += coord_loss.item()
            else:
                outputs = model(batch_tiles)
                loss = criterion(outputs, batch_coords)
                coord_loss = loss

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        if predict_uncertainty:
            train_coord_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_coord_loss = 0.0  # Track coordinate loss separately
        with torch.no_grad():
            for batch_tiles, batch_coords in val_loader:
                batch_tiles, batch_coords = batch_tiles.to(device), batch_coords.to(device)

                if predict_uncertainty:
                    model_outputs = model(batch_tiles)

                    # Handle different output formats
                    if using_terrain_difficulty:
                        pred_coords, pred_log_var, pred_difficulty = model_outputs
                        loss, coord_loss, unc_loss, diff_loss = terrain_difficulty_loss(
                            pred_coords, pred_log_var, pred_difficulty, batch_coords,
                            coord_weight=1.0, uncertainty_weight=1.0, difficulty_weight=0.5
                        )
                    elif uncertainty_arch == "anisotropic":
                        pred_coords, pred_log_vars = model_outputs
                        loss, coord_loss, unc_loss = anisotropic_uncertainty_loss(
                            pred_coords, pred_log_vars, batch_coords,
                            coord_weight=1.0, uncertainty_weight=1.0
                        )
                    else:
                        pred_coords, pred_log_var = model_outputs
                        loss, coord_loss, unc_loss = multi_task_loss_with_uncertainty(
                            pred_coords, pred_log_var, batch_coords,
                            coord_weight=1.0,
                            uncertainty_weight=1.0
                        )
                    val_coord_loss += coord_loss.item()
                else:
                    outputs = model(batch_tiles)
                    loss = criterion(outputs, batch_coords)
                    coord_loss = loss

                val_loss += loss.item()

        val_loss /= len(val_loader)
        if predict_uncertainty:
            val_coord_loss /= len(val_loader)

        # Update scheduler
        scheduler.step()
        lr_current = optimizer.param_groups[0]['lr']

        # Store history
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['learning_rates'].append(lr_current)

        # Convert to meters for display
        # For uncertainty models, use the coordinate loss component (pure MSE)
        if predict_uncertainty:
            # Debug: check for issues
            if train_coord_loss < 0 or val_coord_loss < 0:
                print(f"WARNING: Negative coord loss! train={train_coord_loss:.6f}, val={val_coord_loss:.6f}")
            if np.isnan(train_coord_loss) or np.isnan(val_coord_loss):
                print(f"WARNING: NaN in coord loss! train={train_coord_loss:.6f}, val={val_coord_loss:.6f}")

            # Ensure non-negative before sqrt
            train_m = np.sqrt(max(0, train_coord_loss)) * 7500 * 2.0
            val_m = np.sqrt(max(0, val_coord_loss)) * 7500 * 2.0
        else:
            train_m = np.sqrt(max(0, train_loss)) * 7500 * 2.0
            val_m = np.sqrt(max(0, val_loss)) * 7500 * 2.0

        print(f"  Epoch {epoch+1:2d}/{epochs} | Train: {train_m:4.0f}m | Val: {val_m:4.0f}m | LR: {lr_current:.1e}")

        # Early stopping logic with model saving
        # Use coordinate loss for early stopping (what we actually care about)
        comparison_loss = val_coord_loss if predict_uncertainty else val_loss

        if comparison_loss < best_val_loss:
            best_val_loss = comparison_loss
            patience_counter = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"artifacts/model_{timestamp}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"    → Saved best model: {model_path} ({val_m:.0f}m)")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"    Early stopping after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)")
                break

        # Stop if learning rate gets too low
        if lr_current < 1e-6:
            print(f"    Stopping: Learning rate too low ({lr_current:.2e})")
            break

    print(f"\n✅ Training completed!")
    # For uncertainty models, we tracked coord loss separately
    # For regular models, best_val_loss is already coord loss
    best_error_m = np.sqrt(best_val_loss) * 7500 * 2.0
    print(f"  Best validation error: {best_error_m:.0f} meters")
    if predict_uncertainty:
        print(f"  (with uncertainty estimation enabled)")
    print(f"  Total epochs: {epoch+1}")

    return model, history, model_path, best_val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training data (.pkl)")
    parser.add_argument("--arch", choices=["basic", "deep"], default="basic", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (revert to working value)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (revert to working value)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (revert to working value)")

    # Augmentation options
    parser.add_argument("--flight-name", default="main_evaluation", help="Flight path name for heading calculation")
    parser.add_argument("--enable-augmentation", action="store_true", help="Enable robust augmentation pipeline")
    parser.add_argument("--disable-rotation", action="store_true", help="Disable rotation augmentation")
    parser.add_argument("--disable-scale", action="store_true", help="Disable scale augmentation")
    parser.add_argument("--disable-noise", action="store_true", help="Disable environmental noise")
    parser.add_argument("--noise-prob", type=float, default=0.7, help="Probability of applying noise effects")

    # Uncertainty estimation
    parser.add_argument("--predict-uncertainty", action="store_true", help="Enable uncertainty estimation head")
    parser.add_argument("--uncertainty-arch", choices=["scalar", "anisotropic", "deep_head", "terrain_difficulty"],
                       default="scalar", help="Uncertainty architecture type")

    args = parser.parse_args()

    print(f"🎯 Training Navigation Model")
    print(f"  Data: {args.data}")
    print(f"  Architecture: {args.arch}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Augmentation enabled: {args.enable_augmentation}")
    print(f"  Uncertainty estimation: {args.predict_uncertainty}")
    if args.predict_uncertainty:
        print(f"  Uncertainty architecture: {args.uncertainty_arch}")

    # Load data
    tiles, coordinates = load_data(args.data)

    # Create datasets based on augmentation settings
    if args.enable_augmentation:
        print(f"  Using robust augmentation pipeline:")
        print(f"    Flight path: {args.flight_name}")
        print(f"    Rotation: {not args.disable_rotation}")
        print(f"    Scale: {not args.disable_scale}")
        print(f"    Environmental noise: {not args.disable_noise}")

        dataset = RobustNavigationDataset(
            tiles=tiles,
            coordinates=coordinates,
            flight_name=args.flight_name,
            enable_rotation=not args.disable_rotation,
            enable_scale=not args.disable_scale,
            enable_environmental_noise=not args.disable_noise,
            noise_probability=args.noise_prob
        )
    else:
        print(f"  Using basic dataset (legacy mode)")
        dataset = TerrainDataset(tiles, coordinates)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    # Create and train model
    model = create_model(args.arch, predict_uncertainty=args.predict_uncertainty,
                        uncertainty_arch=args.uncertainty_arch if args.predict_uncertainty else "scalar")

    # Check if using terrain difficulty model (has 3 outputs instead of 2)
    using_terrain_difficulty = (args.predict_uncertainty and args.uncertainty_arch == "terrain_difficulty")

    model, history, model_path, best_val_loss = train_model(
        model, train_loader, val_loader, args.epochs, args.lr,
        predict_uncertainty=args.predict_uncertainty,
        uncertainty_arch=args.uncertainty_arch if args.predict_uncertainty else "scalar",
        using_terrain_difficulty=using_terrain_difficulty
    )

    # Save results
    # For uncertainty models, we need to track coord loss separately for proper error metric
    # The history contains combined loss, but best_val_loss is the coord loss
    results = {
        'args': vars(args),
        'history': history,
        'model_path': model_path,
        'best_val_error_meters': np.sqrt(best_val_loss) * 7500 * 2.0,  # Use best_val_loss which is coord loss
        'uncertainty_model': args.predict_uncertainty
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"artifacts/training_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"  Best validation error: {results['best_val_error_meters']:.0f} meters")
    if args.predict_uncertainty:
        print(f"  Uncertainty estimation: enabled")
    print(f"  Model: {model_path}")
    print(f"  Results: {results_path}")

if __name__ == "__main__":
    main()