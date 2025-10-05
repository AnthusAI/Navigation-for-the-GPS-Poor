#!/usr/bin/env python3
"""
Train ensemble of diverse models for ultra-high precision navigation.
5 different architectures designed to capture different aspects of terrain navigation.
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


# ===================== MODEL ARCHITECTURES =====================

class EnsembleModel1_EfficientNet(nn.Module):
    """EfficientNet-B3 based model - excellent accuracy/efficiency balance."""
    def __init__(self):
        super(EnsembleModel1_EfficientNet, self).__init__()

        # Use EfficientNet-B3 as backbone
        self.backbone = models.efficientnet_b3(pretrained=True)

        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)


class EnsembleModel2_ResNet50(nn.Module):
    """ResNet50 with enhanced classifier - deeper feature extraction."""
    def __init__(self):
        super(EnsembleModel2_ResNet50, self).__init__()

        # ResNet50 backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Enhanced classifier with attention
        self.attention = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)  # [B, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, 2048]

        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights

        return self.classifier(attended_features)


class EnsembleModel3_MultiScale(nn.Module):
    """Multi-scale CNN processing different resolutions simultaneously."""
    def __init__(self):
        super(EnsembleModel3_MultiScale, self).__init__()

        # Three different scale processors
        self.scale1_net = self._make_scale_net(64)   # Fine details
        self.scale2_net = self._make_scale_net(128)  # Medium features
        self.scale3_net = self._make_scale_net(256)  # Coarse features

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(64 + 128 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def _make_scale_net(self, output_dim):
        return nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 2, 2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, output_dim, 3, 1, 1), nn.BatchNorm2d(output_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )

    def forward(self, x):
        # Process at different scales
        scale1_out = self.scale1_net(x)
        scale2_out = self.scale2_net(x)
        scale3_out = self.scale3_net(x)

        # Fuse features
        fused = torch.cat([scale1_out, scale2_out, scale3_out], dim=1)
        return self.fusion(fused)


class EnsembleModel4_Attention(nn.Module):
    """Heavy attention-based model focusing on important regions."""
    def __init__(self):
        super(EnsembleModel4_Attention, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, 2, 2), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
        )

        # Multi-head spatial attention
        self.attention_heads = nn.ModuleList([
            self._make_attention_head(512) for _ in range(4)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2), nn.Sigmoid()
        )

    def _make_attention_head(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(channels//8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)

        # Apply multi-head attention
        attended_features = features
        for attention_head in self.attention_heads:
            attention_map = attention_head(features)
            attended_features = attended_features + (features * attention_map)

        return self.classifier(attended_features)


class EnsembleModel5_Deep(nn.Module):
    """Very deep model with skip connections for complex patterns."""
    def __init__(self):
        super(EnsembleModel5_Deep, self).__init__()

        # Use DenseNet backbone for deep feature extraction
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2), nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


# ===================== DATASET =====================

class EnsembleDataset(Dataset):
    """Enhanced dataset with model-specific augmentations."""
    def __init__(self, tiles, coordinates, model_type='standard', is_training=False):
        self.tiles = tiles
        self.coordinates = coordinates
        self.model_type = model_type
        self.is_training = is_training

        if is_training:
            self.transform = self._get_training_transform(model_type)
        else:
            self.transform = self._get_validation_transform()

    def _get_training_transform(self, model_type):
        """Different augmentation strategies for different models."""
        base_transforms = [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        ]

        if model_type == 'efficientnet':
            # More aggressive augmentation for EfficientNet
            augmentations = [
                transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomVerticalFlip(p=0.1),
            ]
        elif model_type == 'multiscale':
            # Conservative augmentation for multi-scale
            augmentations = [
                transforms.RandomRotation(3),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomHorizontalFlip(p=0.3),
            ]
        elif model_type == 'attention':
            # Focus-preserving augmentation for attention model
            augmentations = [
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.RandomHorizontalFlip(p=0.35),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.5))], p=0.2),
            ]
        else:
            # Standard augmentation for ResNet50 and Deep models
            augmentations = [
                transforms.RandomRotation(6),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                transforms.RandomHorizontalFlip(p=0.35),
            ]

        return transforms.Compose(base_transforms + augmentations + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_validation_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = Image.fromarray(self.tiles[idx])
        coords = torch.tensor(self.coordinates[idx], dtype=torch.float32)

        if self.transform:
            tile = self.transform(tile)

        return tile, coords


# ===================== TRAINING FUNCTIONS =====================

def get_device_config():
    """Get optimal configuration for current device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        batch_size = 24
        num_workers = 4
        pin_memory = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        batch_size = 16
        num_workers = 2
        pin_memory = False
    else:
        device = torch.device('cpu')
        batch_size = 8
        num_workers = 2
        pin_memory = False

    return device, batch_size, num_workers, pin_memory


def train_single_model(model_class, model_name, model_type, tiles, coordinates, device, batch_size, num_workers, pin_memory):
    """Train a single ensemble member."""
    print(f"\\n{'='*60}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*60}")

    # Split data
    split_idx = int(0.85 * len(tiles))
    train_tiles, val_tiles = tiles[:split_idx], tiles[split_idx:]
    train_coords, val_coords = coordinates[:split_idx], coordinates[split_idx:]

    # Create datasets with model-specific augmentation
    train_dataset = EnsembleDataset(train_tiles, train_coords, model_type=model_type, is_training=True)
    val_dataset = EnsembleDataset(val_tiles, val_coords, model_type=model_type, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

    # Create and setup model
    model = model_class().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    # Training loop
    num_epochs = 25
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0

    history = {'train_losses': [], 'val_losses': []}

    for epoch in range(num_epochs):
        # Training
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

        # Validation
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
            torch.save(model.state_dict(), f'artifacts/ensemble_{model_name.lower().replace(" ", "_")}_model.pth')
            print(f'    → New best model saved (val_loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

    # Final evaluation
    model.load_state_dict(torch.load(f'artifacts/ensemble_{model_name.lower().replace(" ", "_")}_model.pth', map_location=device))

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

    print(f"\\n🎯 {model_name} FINAL RESULTS:")
    print(f"   Mean error:   {mean_error:.1f} pixels")
    print(f"   Median error: {np.median(errors):.1f} pixels")
    print(f"   Min error:    {np.min(errors):.1f} pixels")
    print(f"   Max error:    {np.max(errors):.1f} pixels")

    # Save results
    results = {
        'predictions': predictions,
        'targets': targets,
        'errors': errors,
        'mean_error': mean_error,
        'history': history,
        'model_name': model_name
    }

    with open(f'artifacts/ensemble_{model_name.lower().replace(" ", "_")}_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return mean_error


def train_ensemble():
    """Train all ensemble members."""
    print("ENSEMBLE TRAINING: 5 Diverse Models for Ultra-High Precision")
    print("=" * 70)

    device, batch_size, num_workers, pin_memory = get_device_config()
    print(f"Device: {device.type.upper()}")
    print(f"Batch size: {batch_size}")

    # Load data
    dataset_path = 'artifacts/corridor_dataset.pkl'
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    tiles = data['tiles']
    coordinates = data['coordinates']

    if isinstance(tiles, list):
        tiles = np.array(tiles)
        coordinates = np.array(coordinates)

    print(f"Dataset: {len(tiles)} samples")

    # Memory management
    max_samples = 40000
    if len(tiles) > max_samples:
        print(f"Sampling {max_samples} examples for efficiency...")
        indices = np.random.choice(len(tiles), max_samples, replace=False)
        tiles = tiles[indices]
        coordinates = coordinates[indices]

    # Define ensemble models
    ensemble_models = [
        (EnsembleModel1_EfficientNet, "EfficientNet B3", "efficientnet"),
        (EnsembleModel2_ResNet50, "ResNet50 Attention", "resnet"),
        (EnsembleModel3_MultiScale, "MultiScale CNN", "multiscale"),
        (EnsembleModel4_Attention, "Heavy Attention", "attention"),
        (EnsembleModel5_Deep, "DenseNet Deep", "deep"),
    ]

    # Train each model
    results = {}
    for model_class, model_name, model_type in ensemble_models:
        try:
            mean_error = train_single_model(
                model_class, model_name, model_type,
                tiles, coordinates, device, batch_size, num_workers, pin_memory
            )
            results[model_name] = mean_error
        except Exception as e:
            print(f"❌ Failed to train {model_name}: {e}")
            results[model_name] = float('inf')

    # Summary
    print(f"\\n{'='*70}")
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*70}")
    print("Individual Model Results:")

    for model_name, error in results.items():
        if error != float('inf'):
            status = "🏆 Excellent" if error < 35 else "✅ Good" if error < 45 else "⚠️ Fair"
            print(f"   {model_name:<20}: {error:6.1f}px {status}")
        else:
            print(f"   {model_name:<20}: FAILED ❌")

    baseline_error = 153.9
    universal_error = 38.6
    best_ensemble_error = min([e for e in results.values() if e != float('inf')])

    print(f"\\nComparison:")
    print(f"   Original Baseline:  {baseline_error:.1f}px")
    print(f"   Universal CNN:      {universal_error:.1f}px")
    print(f"   Best Ensemble Member: {best_ensemble_error:.1f}px")

    if best_ensemble_error < universal_error:
        improvement = ((universal_error - best_ensemble_error) / universal_error) * 100
        print(f"   🎉 Additional improvement: {improvement:.1f}%")

    print("\\n✅ Ready for ensemble prediction testing!")


if __name__ == "__main__":
    train_ensemble()