"""
Train improved CoordConvPoseNet model for terrain navigation.
This script uses the same data and evaluation as the baseline CorridorCNN
but with the improved architecture that addresses spatial bias.
"""
import sys
sys.path.append('../..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle

# Import our improved model
from src.navigation.models import CoordConvPoseNet

def manual_transform(image, target_size=(224, 224)):
    """Manual image preprocessing without torchvision."""
    # Resize image
    resized = image.resize(target_size, Image.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(resized).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_array = (img_array - mean) / std

    # Convert to tensor and permute dimensions (H,W,C) -> (C,H,W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

    return img_tensor

class CorridorDataset(Dataset):
    """Dataset for terrain patches with their locations."""
    def __init__(self, map_path, samples=5000, tile_size=(1200, 675)):
        self.map_path = map_path
        self.tile_size = tile_size
        self.samples = samples

        # Load the full map
        self.full_map = Image.open(map_path).convert('RGB')
        self.map_width, self.map_height = self.full_map.size

        # Generate random samples
        np.random.seed(42)  # For reproducibility
        self.sample_coords = []

        for _ in range(samples):
            # Ensure we can crop a full tile
            x = np.random.randint(tile_size[0]//2, self.map_width - tile_size[0]//2)
            y = np.random.randint(tile_size[1]//2, self.map_height - tile_size[1]//2)
            self.sample_coords.append((x, y))

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        x, y = self.sample_coords[idx]

        # Crop the tile
        left = x - self.tile_size[0] // 2
        top = y - self.tile_size[1] // 2
        right = left + self.tile_size[0]
        bottom = top + self.tile_size[1]

        tile = self.full_map.crop((left, top, right, bottom))

        # Convert to tensor using manual transform
        tile_tensor = manual_transform(tile)

        # Normalize coordinates to [0, 1]
        norm_x = x / self.map_width
        norm_y = y / self.map_height

        return tile_tensor, torch.tensor([norm_x, norm_y], dtype=torch.float32)

def train_model(model, train_loader, val_loader, device, epochs=25):
    """Train the model and return training history."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []

    print(f"Training CoordConvPoseNet for {epochs} epochs...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                val_pbar.set_postfix({'val_loss': f'{criterion(output, target).item():.6f}'})

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return model, {'train_losses': train_losses, 'val_losses': val_losses}

def evaluate_on_flight_path(model, device):
    """Evaluate the trained model on the corridor flight path."""
    print("\n--- Evaluating Improved Model on Flight Path ---")

    map_path = '../../data/boneyard/davis_monthan_stitched_map.jpg'
    full_map = Image.open(map_path).convert('RGB')
    map_width, map_height = full_map.size

    # Define the same flight path as the original
    start_coord = (5500, 4500)  # Desert start
    end_coord = (4167, 4167)    # Boneyard end
    num_frames = 150

    # Create smooth path
    path_x = np.linspace(start_coord[0], end_coord[0], num_frames)
    path_y = np.linspace(start_coord[1], end_coord[1], num_frames)
    ground_truth = np.column_stack((path_x, path_y))

    # Model configuration
    tile_size = (1200, 675)
    zoom_factor = 4

    model.eval()
    predictions = []
    errors = []

    # Use manual transform function

    print("Running improved model inference on flight path...")
    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Processing frames"):
            gt_x, gt_y = ground_truth[i]

            # Create input image (same as original evaluation)
            crop_width = tile_size[0] // zoom_factor
            crop_height = tile_size[1] // zoom_factor

            left = int(gt_x - crop_width / 2)
            top = int(gt_y - crop_height / 2)
            right = left + crop_width
            bottom = top + crop_height

            # Crop and resize
            cropped = full_map.crop((left, top, right, bottom))
            input_image = cropped.resize(tile_size, Image.LANCZOS)

            # Predict
            input_tensor = manual_transform(input_image).unsqueeze(0).to(device)
            pred_norm = model(input_tensor).cpu().numpy()[0]

            # Denormalize predictions
            pred_x = pred_norm[0] * map_width
            pred_y = pred_norm[1] * map_height

            predictions.append([pred_x, pred_y])

            # Calculate error
            error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            errors.append(error)

    predictions = np.array(predictions)
    errors = np.array(errors)

    # Print statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)

    print(f"\nImproved Model Performance:")
    print(f"Mean error: {mean_error:.1f} pixels")
    print(f"Median error: {median_error:.1f} pixels")
    print(f"Max error: {max_error:.1f} pixels")

    # Save results
    results = {
        'ground_truth': ground_truth,
        'predictions': predictions,
        'errors': errors,
        'mean_error': mean_error,
        'median_error': median_error,
        'max_error': max_error
    }

    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/improved_model_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("✅ Improved model evaluation results saved to artifacts/improved_model_results.pkl")
    return results

def main():
    """Main training and evaluation pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configuration
    map_path = '../../data/boneyard/davis_monthan_stitched_map.jpg'
    batch_size = 16
    epochs = 25

    # Check if model already exists
    model_path = 'artifacts/improved_model.pth'
    if os.path.exists(model_path):
        print("Loading existing improved model...")
        model = CoordConvPoseNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Improved model loaded successfully")
    else:
        print("Training new improved model...")

        # Create datasets
        print("Creating training dataset...")
        train_dataset = CorridorDataset(map_path, samples=8000)
        val_dataset = CorridorDataset(map_path, samples=2000)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Train model
        model = CoordConvPoseNet()
        model, history = train_model(model, train_loader, val_loader, device, epochs)

        # Save model and history
        os.makedirs('artifacts', exist_ok=True)
        torch.save(model.state_dict(), model_path)

        with open('artifacts/improved_model_history.pkl', 'wb') as f:
            pickle.dump(history, f)

        print("✅ Improved model and history saved")

    # Evaluate on flight path
    results = evaluate_on_flight_path(model, device)

    print("--- Improved Model Training Complete ---")
    return model, results

if __name__ == "__main__":
    main()