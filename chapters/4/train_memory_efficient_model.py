#!/usr/bin/env python3
"""
Memory-efficient improved model designed to beat the 153.9px baseline.
Focus on efficiency while maintaining performance improvements:
1. Lightweight architecture with proven techniques
2. Smart data loading to avoid memory issues
3. Gradient accumulation for effective larger batch sizes
4. Efficient feature extraction
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


class MemoryEfficientCNN(nn.Module):
    """
    Memory-efficient CNN designed to beat the baseline while using minimal memory.
    Key innovations:
    - Lightweight but deeper architecture
    - Depthwise separable convolutions for efficiency
    - Smart pooling strategy
    - Efficient attention mechanism
    """
    def __init__(self):
        super(MemoryEfficientCNN, self).__init__()

        # Efficient feature extraction using depthwise separable convolutions
        self.features = nn.Sequential(
            # Initial convolution - larger to capture terrain features
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 56

            # Depthwise separable conv block 1
            self._make_depthwise_block(32, 64, stride=2),  # 56 -> 28

            # Depthwise separable conv block 2
            self._make_depthwise_block(64, 128, stride=2),  # 28 -> 14

            # Depthwise separable conv block 3
            self._make_depthwise_block(128, 256, stride=2),  # 14 -> 7

            # Final feature refinement
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Lightweight spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Use regular pooling instead of adaptive pooling for MPS compatibility
        self.final_pool = nn.AvgPool2d(kernel_size=7)  # Assumes 7x7 feature maps

        # Efficient classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 * 1 * 1, 256),  # After global pooling: 512 x 1 x 1
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(128, 2),
            nn.Sigmoid()
        )

        # Initialize weights
        self._initialize_weights()

    def _make_depthwise_block(self, in_channels, out_channels, stride=1):
        """Create a depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

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
        attended_features = features * attention_weights  # Broadcast multiply

        # Global average pooling
        pooled = self.final_pool(attended_features)  # [B, 512, 1, 1]

        # Flatten and classify
        flattened = pooled.view(pooled.size(0), -1)  # [B, 512]
        output = self.classifier(flattened)

        return output


class MemoryEfficientDataset(Dataset):
    """Memory-efficient dataset that loads data on-demand."""
    def __init__(self, tiles, coordinates, transform=None, is_training=False):
        self.tiles = tiles
        self.coordinates = coordinates
        self.is_training = is_training

        # Efficient transforms
        if is_training and transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomRotation(3),  # Very conservative
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


def train_memory_efficient_model():
    """Train memory-efficient model with gradient accumulation."""
    print("Training Memory-Efficient Improved CNN")
    print("=" * 45)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset with memory management
    print("Loading corridor dataset...")
    dataset_path = 'artifacts/corridor_dataset.pkl'

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    tiles = data['tiles']
    coordinates = data['coordinates']

    print(f"Dataset: {len(tiles)} samples")

    # Convert to numpy arrays if they're lists
    if isinstance(tiles, list):
        tiles = np.array(tiles)
        coordinates = np.array(coordinates)

    print(f"Memory usage: ~{tiles.nbytes / 1024**3:.1f} GB")

    # Use subset of data to avoid memory issues, but sample strategically
    max_samples = 50000  # Reasonable limit
    if len(tiles) > max_samples:
        print(f"Sampling {max_samples} examples for memory efficiency...")
        indices = np.random.choice(len(tiles), max_samples, replace=False)
        tiles = tiles[indices]
        coordinates = coordinates[indices]

    # Split data
    split_idx = int(0.85 * len(tiles))
    train_tiles, val_tiles = tiles[:split_idx], tiles[split_idx:]
    train_coords, val_coords = coordinates[:split_idx], coordinates[split_idx:]

    print(f"Split: {len(train_tiles)} train, {len(val_tiles)} validation")

    # Create datasets
    train_dataset = MemoryEfficientDataset(train_tiles, train_coords, is_training=True)
    val_dataset = MemoryEfficientDataset(val_tiles, val_coords, is_training=False)

    # Conservative batch sizes
    batch_size = 16 if device.type == 'cuda' else 8 if device.type == 'mps' else 4
    accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps

    # Disable pin_memory for MPS compatibility
    use_pin_memory = device.type == 'cuda'

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=1, pin_memory=use_pin_memory)

    # Create model
    model = MemoryEfficientCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    # Training parameters
    num_epochs = 35
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    history = {'train_losses': [], 'val_losses': []}

    print(f"\\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase with gradient accumulation
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:2d}/{num_epochs}')
        for batch_idx, (frames_batch, coords_batch) in enumerate(pbar):
            frames_batch = frames_batch.to(device, non_blocking=True)
            coords_batch = coords_batch.to(device, non_blocking=True)

            # Forward pass
            outputs = model(frames_batch)
            loss = criterion(outputs, coords_batch) / accumulation_steps  # Scale loss

            # Backward pass
            loss.backward()

            train_loss += loss.item() * accumulation_steps  # Unscale for logging

            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.6f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # Handle remaining gradients
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for frames_batch, coords_batch in val_loader:
                frames_batch = frames_batch.to(device, non_blocking=True)
                coords_batch = coords_batch.to(device, non_blocking=True)

                outputs = model(frames_batch)
                loss = criterion(outputs, coords_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)

        print(f'Epoch {epoch+1:2d}: Train: {train_loss:.6f}, Val: {val_loss:.6f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'artifacts/memory_efficient_model.pth')
            print(f'    → New best model saved (val_loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        # Clear cache periodically
        if epoch % 5 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Load best model for evaluation
    print("\\nLoading best model for evaluation...")
    model.load_state_dict(torch.load('artifacts/memory_efficient_model.pth'))

    # Evaluate on validation set
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for frames_batch, coords_batch in val_loader:
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

    print(f"\\n🎯 MEMORY-EFFICIENT MODEL RESULTS:")
    print(f"   Mean error:   {mean_error:.1f} pixels")
    print(f"   Median error: {np.median(errors):.1f} pixels")
    print(f"   Std dev:      {np.std(errors):.1f} pixels")
    print(f"   Min error:    {np.min(errors):.1f} pixels")
    print(f"   Max error:    {np.max(errors):.1f} pixels")

    # Compare with baseline
    baseline_error = 153.9
    print(f"\\n📊 COMPARISON WITH BASELINE:")
    print(f"   Baseline (simple):     {baseline_error:.1f} pixels")
    print(f"   Memory-efficient:      {mean_error:.1f} pixels")

    if mean_error < baseline_error:
        improvement = ((baseline_error - mean_error) / baseline_error) * 100
        print(f"   🎉 IMPROVEMENT: {improvement:.1f}% better!")
        print(f"   🏆 SUCCESS: New best model achieved!")
    else:
        regression = ((mean_error - baseline_error) / baseline_error) * 100
        print(f"   😞 REGRESSION: {regression:.1f}% worse")

    # Save results
    results = {
        'predictions': predictions,
        'targets': targets,
        'errors': errors,
        'mean_error': mean_error,
        'median_error': np.median(errors),
        'history': history,
        'model_type': 'memory_efficient'
    }

    with open('artifacts/memory_efficient_model_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\\n✅ Training complete!")
    print(f"   Model saved: artifacts/memory_efficient_model.pth")
    print(f"   Results saved: artifacts/memory_efficient_model_results.pkl")

    return model, results


if __name__ == "__main__":
    train_memory_efficient_model()