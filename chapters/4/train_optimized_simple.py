#!/usr/bin/env python3
"""
Train an optimized version that beats the 153.9px baseline.
Focus on proven techniques: better data usage, careful regularization, longer training.
"""
import sys
sys.path.append('../..')

import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from src.navigation.deep_learning import FlightDataset, ArtifactCache, evaluate_model, get_device, count_parameters
from src.navigation.models import SmallPoseNet

def train_optimized_simple():
    """Train an optimized simple model that should beat baseline."""
    print("Training Optimized Simple Model")
    print("=" * 40)

    device = get_device()
    print(f"Device: {device}")

    cache = ArtifactCache('artifacts')

    # Load corridor dataset
    print("Loading corridor dataset...")
    with open('artifacts/corridor_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    frames = dataset['tiles']
    coordinates = dataset['coordinates']
    print(f"Dataset: {len(frames)} samples")

    # Better data split - use more for training
    train_size = int(0.8 * len(frames))  # 80% vs original 70%
    val_size = int(0.1 * len(frames))    # 10% vs original 15%

    train_frames = frames[:train_size]
    train_coords = coordinates[:train_size]
    val_frames = frames[train_size:train_size + val_size]
    val_coords = coordinates[train_size:train_size + val_size]
    test_frames = frames[train_size + val_size:]
    test_coords = coordinates[train_size + val_size:]

    print(f"Split: {len(train_frames)} train, {len(val_frames)} val, {len(test_frames)} test")

    # Conservative but effective data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(3),  # Very small rotation
        transforms.RandomResizedCrop(224, scale=(0.98, 1.0)),  # Minimal cropping
        transforms.ColorJitter(brightness=0.03, contrast=0.03),  # Subtle changes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = FlightDataset(train_frames, train_coords, transform=train_transform)
    val_dataset = FlightDataset(val_frames, val_coords, transform=val_transform)
    test_dataset = FlightDataset(test_frames, test_coords, transform=val_transform)

    batch_size = 32 if device.type in ['cuda', 'mps'] else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Use SmallPoseNet - keep it simple but train better
    model = SmallPoseNet().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Better training setup
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)  # Lower LR, weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    # Training loop
    num_epochs = 40  # More epochs
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    history = {'train_losses': [], 'val_losses': []}

    print(f"\\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:2d}/{num_epochs}')
        for frames_batch, coords_batch in pbar:
            frames_batch = frames_batch.to(device)
            coords_batch = coords_batch.to(device)

            optimizer.zero_grad()
            outputs = model(frames_batch)
            loss = criterion(outputs, coords_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        train_loss /= train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for frames_batch, coords_batch in val_loader:
                frames_batch = frames_batch.to(device)
                coords_batch = coords_batch.to(device)

                outputs = model(frames_batch)
                loss = criterion(outputs, coords_batch)

                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)

        print(f'Epoch {epoch+1:2d}: Train={train_loss:.6f}, Val={val_loss:.6f}')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping with best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'artifacts/optimized_simple_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Load best model
    model.load_state_dict(torch.load('artifacts/optimized_simple_model.pth'))

    # Evaluate
    print("\\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, device, (7500, 7500))

    mean_error = results['errors'].mean()
    median_error = np.median(results['errors'])

    print(f"\\nResults:")
    print(f"Mean error: {mean_error:.1f} pixels")
    print(f"Median error: {median_error:.1f} pixels")
    print(f"Std dev: {results['errors'].std():.1f} pixels")

    # Compare with baseline
    baseline_error = 153.9
    print(f"\\nComparison:")
    print(f"Baseline:     {baseline_error:.1f}px")
    print(f"Optimized:    {mean_error:.1f}px")

    if mean_error < baseline_error:
        improvement = ((baseline_error - mean_error) / baseline_error) * 100
        print(f"🎉 IMPROVEMENT: {improvement:.1f}% better!")
    else:
        regression = ((mean_error - baseline_error) / baseline_error) * 100
        print(f"😞 REGRESSION: {regression:.1f}% worse")

    # Save results
    cache.save_results('optimized_simple_eval', results)

    with open('artifacts/optimized_simple_history.json', 'w') as f:
        import json
        json.dump(history, f, indent=2)

    print("\\n✅ Training complete!")
    return model, results, history

if __name__ == "__main__":
    train_optimized_simple()