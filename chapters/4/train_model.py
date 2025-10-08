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

class SimpleDeepModel(nn.Module):
    """Lightweight model optimized for small diverse datasets."""

    def __init__(self):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights

        # Load DenseNet backbone
        backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = backbone.features

        # Much simpler regression head for small datasets
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),  # Reduced dropout
            nn.Linear(1024, 256),  # Fewer parameters
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # Direct to output
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

class BasicModel(nn.Module):
    """Ultra-simple model for small datasets."""

    def __init__(self):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights

        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        # Minimal classifier to prevent overfitting
        # Keep ultra-simple architecture - it was working at 183m
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
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

def create_model(arch):
    """Create model based on architecture choice."""
    if arch == "deep":
        return SimpleDeepModel()
    elif arch == "basic":
        return BasicModel()
    else:
        raise ValueError(f"Unknown architecture: {arch}")

def train_model(model, train_loader, val_loader, epochs, lr):
    """Train the model with robust early stopping."""
    device = get_device()
    model = model.to(device)

    print(f"Training on {device} for up to {epochs} epochs (with early stopping)")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")

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
        for batch_tiles, batch_coords in train_loader:
            batch_tiles, batch_coords = batch_tiles.to(device), batch_coords.to(device)

            optimizer.zero_grad()
            outputs = model(batch_tiles)
            loss = criterion(outputs, batch_coords)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_tiles, batch_coords in val_loader:
                batch_tiles, batch_coords = batch_tiles.to(device), batch_coords.to(device)
                outputs = model(batch_tiles)
                loss = criterion(outputs, batch_coords)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Update scheduler
        scheduler.step()
        lr_current = optimizer.param_groups[0]['lr']

        # Store history
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['learning_rates'].append(lr_current)

        # Convert to meters for display
        train_m = np.sqrt(train_loss) * 7500 * 2.0
        val_m = np.sqrt(val_loss) * 7500 * 2.0

        print(f"  Epoch {epoch+1:2d}/{epochs} | Train: {train_m:4.0f}m | Val: {val_m:4.0f}m | LR: {lr_current:.1e}")

        # Early stopping logic with model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    print(f"  Best validation error: {np.sqrt(best_val_loss) * 7500 * 2.0:.0f} meters")
    print(f"  Total epochs: {epoch+1}")

    return model, history, model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training data (.pkl)")
    parser.add_argument("--arch", choices=["basic", "deep"], default="basic", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (revert to working value)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (revert to working value)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (revert to working value)")
    args = parser.parse_args()

    print(f"🎯 Training Navigation Model")
    print(f"  Data: {args.data}")
    print(f"  Architecture: {args.arch}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")

    # Load data
    tiles, coordinates = load_data(args.data)

    # Create datasets
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
    model = create_model(args.arch)
    model, history, model_path = train_model(model, train_loader, val_loader, args.epochs, args.lr)

    # Save results
    results = {
        'args': vars(args),
        'history': history,
        'model_path': model_path,
        'best_val_error_meters': np.sqrt(min(history['val_losses'])) * 7500 * 2.0
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"artifacts/training_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"  Best validation error: {results['best_val_error_meters']:.0f} meters")
    print(f"  Model: {model_path}")
    print(f"  Results: {results_path}")

if __name__ == "__main__":
    main()