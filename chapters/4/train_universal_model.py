#!/usr/bin/env python3
"""
Universal improved model designed to beat the 153.9px baseline.
This version is architecture-agnostic and will work on:
- CUDA GPUs (NVIDIA)
- MPS (Apple Silicon)
- CPU (any architecture)

Key improvements:
1. Portable architecture with proven techniques
2. Smart device detection and optimization
3. Efficient feature extraction that works everywhere
4. Conservative memory usage
"""
import sys
sys.path.append('../..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import pickle
from tqdm import tqdm


class UniversalCNN(nn.Module):
    """
    Universal CNN that works on any device and beats the baseline.
    Uses standard operations that work everywhere.
    """
    def __init__(self):
        super(UniversalCNN, self).__init__()

        # Standard convolutional layers - work on all devices
        self.features = nn.Sequential(
            # Block 1: Initial feature extraction
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 56

            # Block 2: More detailed features
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            # Block 3: Higher-level features
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14

            # Block 4: Complex features
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14 -> 7

            # Block 5: Final feature refinement
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Simple attention mechanism using standard operations
        self.attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Use standard global average pooling (works everywhere)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier with proper regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features
        features = self.features(x)  # [B, 512, 7, 7]

        # Apply attention
        attention_weights = self.attention(features)  # [B, 1, 7, 7]
        attended_features = features * attention_weights

        # Global pooling (works on all devices)
        pooled = self.global_pool(attended_features)  # [B, 512, 1, 1]

        # Flatten and classify
        flattened = pooled.view(pooled.size(0), -1)  # [B, 512]
        output = self.classifier(flattened)

        return output


class UniversalDataset(Dataset):
    """Universal dataset with portable transforms."""
    def __init__(self, tiles, coordinates, transform=None, is_training=False):
        self.tiles = tiles
        self.coordinates = coordinates
        self.is_training = is_training

        # Conservative transforms that work everywhere
        if is_training and transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomRotation(2),  # Very conservative
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = Image.fromarray(self.tiles[idx])
        coords = torch.tensor(self.coordinates[idx], dtype=torch.float32)

        if self.transform:
            tile = self.transform(tile)

        return tile, coords


def get_optimal_device_config():
    """Get optimal configuration for current device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        batch_size = 24
        num_workers = 4
        pin_memory = True
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        batch_size = 16
        num_workers = 2
        pin_memory = False  # MPS doesn't support pin_memory
        print("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        batch_size = 8
        num_workers = 2
        pin_memory = False
        print("Using CPU")

    return device, batch_size, num_workers, pin_memory


def train_universal_model():
    """Train universal model with device-specific optimizations."""
    print("Training Universal Improved CNN")
    print("=" * 40)

    # Get optimal configuration for current device
    device, batch_size, num_workers, pin_memory = get_optimal_device_config()

    # Load dataset
    print("Loading corridor dataset...")
    dataset_path = 'artifacts/corridor_dataset.pkl'

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    tiles = data['tiles']
    coordinates = data['coordinates']

    # Convert to numpy arrays if needed
    if isinstance(tiles, list):
        tiles = np.array(tiles)
        coordinates = np.array(coordinates)

    print(f"Dataset: {len(tiles)} samples")

    # Memory management - use reasonable subset
    max_samples = 40000
    if len(tiles) > max_samples:
        print(f"Sampling {max_samples} examples for efficiency...")
        indices = np.random.choice(len(tiles), max_samples, replace=False)
        tiles = tiles[indices]
        coordinates = coordinates[indices]

    # Split data
    split_idx = int(0.85 * len(tiles))
    train_tiles, val_tiles = tiles[:split_idx], tiles[split_idx:]
    train_coords, val_coords = coordinates[:split_idx], coordinates[split_idx:]

    print(f"Split: {len(train_tiles)} train, {len(val_tiles)} validation")

    # Create datasets
    train_dataset = UniversalDataset(train_tiles, train_coords, is_training=True)
    val_dataset = UniversalDataset(val_tiles, val_coords, is_training=False)

    # Data loaders with device-specific settings
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

    # Create model
    model = UniversalCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Batch size: {batch_size}")

    # Training setup - adaptive based on device
    criterion = nn.MSELoss()

    # Adjust learning rate based on device capabilities
    base_lr = 1e-3
    if device.type == 'cpu':
        base_lr *= 0.5  # More conservative on CPU

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    # Training parameters
    num_epochs = 30
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0

    history = {'train_losses': [], 'val_losses': []}

    print(f"\\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:2d}/{num_epochs}')
        for frames_batch, coords_batch in pbar:
            frames_batch = frames_batch.to(device, non_blocking=pin_memory)
            coords_batch = coords_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(frames_batch)
            loss = criterion(outputs, coords_batch)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        train_loss /= train_batches
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for frames_batch, coords_batch in val_loader:
                frames_batch = frames_batch.to(device, non_blocking=pin_memory)
                coords_batch = coords_batch.to(device, non_blocking=pin_memory)

                outputs = model(frames_batch)
                loss = criterion(outputs, coords_batch)

                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)

        print(f'Epoch {epoch+1:2d}: Train: {train_loss:.6f}, Val: {val_loss:.6f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'artifacts/universal_model.pth')
            print(f'    → New best model saved (val_loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        # Periodic cleanup
        if epoch % 5 == 0 and device.type in ['cuda', 'mps']:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            # Note: MPS doesn't have equivalent of empty_cache yet

    # Load best model for evaluation
    print("\\nLoading best model for evaluation...")
    model.load_state_dict(torch.load('artifacts/universal_model.pth', map_location=device))

    # Evaluate on validation set
    model.eval()
    predictions = []
    targets = []

    print("Evaluating final performance...")
    with torch.no_grad():
        for frames_batch, coords_batch in tqdm(val_loader, desc="Evaluating"):
            frames_batch = frames_batch.to(device)
            coords_batch = coords_batch.to(device)

            outputs = model(frames_batch)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(coords_batch.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate pixel errors
    map_width, map_height = 7500, 7500
    pred_pixels = predictions * np.array([map_width, map_height])
    target_pixels = targets * np.array([map_width, map_height])

    errors = np.sqrt(np.sum((pred_pixels - target_pixels)**2, axis=1))
    mean_error = np.mean(errors)

    print(f"\\n🎯 UNIVERSAL MODEL RESULTS:")
    print(f"   Device: {device.type.upper()}")
    print(f"   Mean error:   {mean_error:.1f} pixels")
    print(f"   Median error: {np.median(errors):.1f} pixels")
    print(f"   Std dev:      {np.std(errors):.1f} pixels")
    print(f"   Min error:    {np.min(errors):.1f} pixels")
    print(f"   Max error:    {np.max(errors):.1f} pixels")

    # Compare with baseline
    baseline_error = 153.9
    print(f"\\n📊 COMPARISON WITH BASELINE:")
    print(f"   Baseline (simple):  {baseline_error:.1f} pixels")
    print(f"   Universal model:    {mean_error:.1f} pixels")

    if mean_error < baseline_error:
        improvement = ((baseline_error - mean_error) / baseline_error) * 100
        print(f"   🎉 IMPROVEMENT: {improvement:.1f}% better!")
        print(f"   🏆 SUCCESS: New best model achieved!")
        success = True
    else:
        regression = ((mean_error - baseline_error) / baseline_error) * 100
        print(f"   😞 REGRESSION: {regression:.1f}% worse")
        success = False

    # Save results
    results = {
        'predictions': predictions,
        'targets': targets,
        'errors': errors,
        'mean_error': mean_error,
        'median_error': np.median(errors),
        'history': history,
        'model_type': 'universal',
        'device_used': device.type,
        'success': success
    }

    with open('artifacts/universal_model_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\\n✅ Training complete!")
    print(f"   Model saved: artifacts/universal_model.pth")
    print(f"   Results saved: artifacts/universal_model_results.pkl")

    return model, results


if __name__ == "__main__":
    train_universal_model()