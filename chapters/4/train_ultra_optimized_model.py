#!/usr/bin/env python3
"""
Ultra-optimized model designed to beat the 153.9px baseline.
This focuses on proven techniques that actually work:
1. Efficient architecture with proper regularization
2. Better data utilization with smart augmentation
3. Multi-scale feature extraction
4. Ensemble-like behavior within single model
5. Transfer learning from stronger backbone
"""
import sys
sys.path.append('../..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import os
import pickle
from tqdm import tqdm


class UltraOptimizedCNN(nn.Module):
    """
    Ultra-optimized CNN designed to beat the baseline.
    Key innovations:
    - ResNet18 backbone for proven feature extraction
    - Multi-scale processing for different terrain features
    - Attention mechanism for spatial focus
    - Ensemble-like dual-path processing
    """
    def __init__(self):
        super(UltraOptimizedCNN, self).__init__()

        # Use pre-trained ResNet18 as backbone (proven architecture)
        resnet = models.resnet18(pretrained=True)

        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze early layers to preserve low-level features
        for param in list(self.backbone.parameters())[:30]:
            param.requires_grad = False

        # Multi-scale feature extraction
        self.scale1_pool = nn.AdaptiveAvgPool2d((8, 8))    # Large scale features
        self.scale2_pool = nn.AdaptiveAvgPool2d((4, 4))    # Medium scale features
        self.scale3_pool = nn.AdaptiveAvgPool2d((2, 2))    # Fine scale features

        # Attention mechanism for spatial focus
        self.attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Dual path processing for robustness
        path1_features = 512 * (8*8 + 4*4 + 2*2)  # Multi-scale features
        path2_features = 512 * 1  # Attention-weighted global features

        self.path1 = nn.Sequential(
            nn.Linear(path1_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.path2 = nn.Sequential(
            nn.Linear(path2_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Final fusion and prediction
        self.final_predictor = nn.Sequential(
            nn.Linear(256, 128),  # 128 + 128 from dual paths
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        # Initialize new layers properly
        self._initialize_new_layers()

    def _initialize_new_layers(self):
        """Initialize only the new layers we added."""
        for module in [self.attention, self.path1, self.path2, self.final_predictor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features using ResNet backbone
        features = self.backbone(x)  # [B, 512, H, W]

        # Multi-scale feature extraction
        scale1 = self.scale1_pool(features).flatten(1)  # [B, 512*64]
        scale2 = self.scale2_pool(features).flatten(1)  # [B, 512*16]
        scale3 = self.scale3_pool(features).flatten(1)  # [B, 512*4]

        # Combine multi-scale features for path 1
        multiscale_features = torch.cat([scale1, scale2, scale3], dim=1)
        path1_out = self.path1(multiscale_features)

        # Attention-weighted global features for path 2
        attention_weights = self.attention(features)
        weighted_features = features * attention_weights
        global_features = torch.mean(weighted_features, dim=[2, 3])  # Global average pooling
        path2_out = self.path2(global_features)

        # Fuse both paths
        combined = torch.cat([path1_out, path2_out], dim=1)
        output = self.final_predictor(combined)

        return output


class CorridorDatasetOptimized(Dataset):
    """Optimized dataset with better augmentation and data utilization."""
    def __init__(self, tiles, coordinates, transform=None, is_training=False):
        self.tiles = tiles
        self.coordinates = coordinates
        self.is_training = is_training

        # Enhanced transforms for training
        if is_training and transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Slightly larger for crops
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomRotation(5),  # Conservative rotation
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                transforms.RandomHorizontalFlip(p=0.3),
                # Add some blur occasionally to improve generalization
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.1),
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


def cosine_annealing_with_warmup(epoch, max_epochs, warmup_epochs=5, min_lr=1e-6, max_lr=1e-3):
    """Custom learning rate schedule with warmup."""
    if epoch < warmup_epochs:
        return min_lr + (max_lr - min_lr) * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return min_lr + (max_lr - min_lr) * (1 + np.cos(np.pi * progress)) / 2


def train_ultra_optimized_model():
    """Train the ultra-optimized model with advanced techniques."""
    print("Training Ultra-Optimized CNN Model")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the large corridor dataset
    print("Loading corridor dataset...")
    dataset_path = 'artifacts/corridor_dataset.pkl'
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    tiles = data['tiles']
    coordinates = data['coordinates']

    print(f"Dataset: {len(tiles)} samples")
    print(f"Tile shape: {tiles[0].shape}")
    print(f"Coordinate range: [{coordinates.min():.3f}, {coordinates.max():.3f}]")

    # Strategic train/val split - use more data for training
    split_idx = int(0.85 * len(tiles))  # Use 85% for training
    train_tiles, val_tiles = tiles[:split_idx], tiles[split_idx:]
    train_coords, val_coords = coordinates[:split_idx], coordinates[split_idx:]

    print(f"Split: {len(train_tiles)} train, {len(val_tiles)} validation")

    # Create optimized datasets
    train_dataset = CorridorDatasetOptimized(train_tiles, train_coords, is_training=True)
    val_dataset = CorridorDatasetOptimized(val_tiles, val_coords, is_training=False)

    # Optimized data loaders
    batch_size = 48 if device.type == 'cuda' else 32 if device.type == 'mps' else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)

    # Create model
    model = UltraOptimizedCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Advanced training setup
    criterion = nn.MSELoss()

    # Use different learning rates for backbone vs new layers
    backbone_params = []
    new_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                new_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4, 'weight_decay': 1e-5},  # Lower LR for backbone
        {'params': new_params, 'lr': 1e-3, 'weight_decay': 1e-4}       # Higher LR for new layers
    ])

    # Training parameters
    num_epochs = 40
    best_val_loss = float('inf')
    patience = 12
    patience_counter = 0

    history = {'train_losses': [], 'val_losses': [], 'learning_rates': []}

    print(f"\\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Custom learning rate schedule
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:  # Backbone
                param_group['lr'] = cosine_annealing_with_warmup(epoch, num_epochs,
                                                               warmup_epochs=3, min_lr=5e-6, max_lr=1e-4)
            else:  # New layers
                param_group['lr'] = cosine_annealing_with_warmup(epoch, num_epochs,
                                                               warmup_epochs=3, min_lr=1e-5, max_lr=1e-3)

        current_lr = optimizer.param_groups[1]['lr']  # Track new layer LR
        history['learning_rates'].append(current_lr)

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:2d}/{num_epochs}')
        for frames_batch, coords_batch in pbar:
            frames_batch = frames_batch.to(device, non_blocking=True)
            coords_batch = coords_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(frames_batch)
            loss = criterion(outputs, coords_batch)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{current_lr:.2e}'})

        train_loss /= train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for frames_batch, coords_batch in val_loader:
                frames_batch = frames_batch.to(device, non_blocking=True)
                coords_batch = coords_batch.to(device, non_blocking=True)

                outputs = model(frames_batch)
                loss = criterion(outputs, coords_batch)

                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)

        print(f'Epoch {epoch+1:2d}: Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'artifacts/ultra_optimized_model.pth')
            print(f'    → New best model saved (val_loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

    # Load best model for evaluation
    print("\\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load('artifacts/ultra_optimized_model.pth'))

    # Final validation to confirm performance
    model.eval()
    final_predictions = []
    final_targets = []

    with torch.no_grad():
        for frames_batch, coords_batch in val_loader:
            frames_batch = frames_batch.to(device)
            coords_batch = coords_batch.to(device)

            outputs = model(frames_batch)

            final_predictions.extend(outputs.cpu().numpy())
            final_targets.extend(coords_batch.cpu().numpy())

    final_predictions = np.array(final_predictions)
    final_targets = np.array(final_targets)

    # Calculate pixel errors (convert normalized coordinates to pixels)
    map_width, map_height = 7500, 7500
    pred_pixels = final_predictions * np.array([map_width, map_height])
    target_pixels = final_targets * np.array([map_width, map_height])

    errors = np.sqrt(np.sum((pred_pixels - target_pixels)**2, axis=1))

    mean_error = np.mean(errors)
    median_error = np.median(errors)

    print(f"\\n🎯 ULTRA-OPTIMIZED MODEL RESULTS:")
    print(f"   Mean error:   {mean_error:.1f} pixels")
    print(f"   Median error: {median_error:.1f} pixels")
    print(f"   Std dev:      {np.std(errors):.1f} pixels")
    print(f"   Min error:    {np.min(errors):.1f} pixels")
    print(f"   Max error:    {np.max(errors):.1f} pixels")

    # Compare with baseline
    baseline_error = 153.9
    print(f"\\n📊 COMPARISON WITH BASELINE:")
    print(f"   Baseline (simple):     {baseline_error:.1f} pixels")
    print(f"   Ultra-optimized:       {mean_error:.1f} pixels")

    if mean_error < baseline_error:
        improvement = ((baseline_error - mean_error) / baseline_error) * 100
        print(f"   🎉 IMPROVEMENT: {improvement:.1f}% better!")
        print(f"   🏆 SUCCESS: New best model achieved!")
    else:
        regression = ((mean_error - baseline_error) / baseline_error) * 100
        print(f"   😞 REGRESSION: {regression:.1f}% worse")

    # Save detailed results
    results = {
        'predictions': final_predictions,
        'targets': final_targets,
        'errors': errors,
        'mean_error': mean_error,
        'median_error': median_error,
        'history': history,
        'model_type': 'ultra_optimized'
    }

    with open('artifacts/ultra_optimized_model_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\\n✅ Training complete!")
    print(f"   Model saved: artifacts/ultra_optimized_model.pth")
    print(f"   Results saved: artifacts/ultra_optimized_model_results.pkl")

    return model, results


if __name__ == "__main__":
    train_ultra_optimized_model()