"""
Complete pipeline: Train a model on corridor data and evaluate it frame-by-frame
on the actual flight path, visualizing predictions vs ground truth.
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
import torchvision.transforms as transforms

class CorridorCNN(nn.Module):
    """CNN designed for 1200x675 input images."""
    def __init__(self):
        super(CorridorCNN, self).__init__()
        # Input: 3 x 1200 x 675
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # -> 32 x 600 x 338
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32 x 300 x 169
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # -> 64 x 150 x 85
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64 x 75 x 42
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> 128 x 38 x 21
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 128 x 19 x 10
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # -> 256 x 10 x 5
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> 256 x 1 x 1
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

class CorridorDataset(Dataset):
    """Dataset from pre-generated corridor samples."""
    def __init__(self, tiles, coordinates, transform=None):
        self.tiles = tiles
        self.coordinates = coordinates
        self.transform = transform or transforms.Compose([
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

def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4):
    """Train the model."""
    print("\n--- Training Model ---")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, coords in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, coords = images.to(device), coords.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coords)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, coords in val_loader:
                images, coords = images.to(device), coords.to(device)
                outputs = model(images)
                loss = criterion(outputs, coords)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'artifacts/corridor_model_best.pth')
    
    print(f"✅ Training complete. Best val loss: {best_val_loss:.6f}")
    return model

def evaluate_on_flight_path(model, map_path, start_coord, end_coord, 
                            tile_size, zoom_factor, num_frames, device):
    """
    Run inference on every frame of the actual flight path.
    Returns ground truth and predicted coordinates for each frame.
    """
    print("\n--- Evaluating on Flight Path ---")
    model.to(device)
    model.eval()
    
    # Load map
    full_map = Image.open(map_path).convert('RGB')
    map_width, map_height = full_map.size
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Generate flight path
    path_x = np.linspace(start_coord[0], end_coord[0], num_frames)
    path_y = np.linspace(start_coord[1], end_coord[1], num_frames)
    
    crop_width = tile_size[0] // zoom_factor
    crop_height = tile_size[1] // zoom_factor
    
    gt_coords = []
    pred_coords = []
    
    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Running inference"):
            cam_x, cam_y = path_x[i], path_y[i]
            
            # Crop and resize
            left = int(cam_x - crop_width / 2)
            top = int(cam_y - crop_height / 2)
            right = left + crop_width
            bottom = top + crop_height
            
            frame = full_map.crop((left, top, right, bottom))
            frame = frame.resize(tile_size, Image.LANCZOS)
            
            # Run inference
            frame_tensor = transform(frame).unsqueeze(0).to(device)
            pred = model(frame_tensor).squeeze().cpu().numpy()
            
            # Denormalize
            pred_x = pred[0] * map_width
            pred_y = pred[1] * map_height
            
            gt_coords.append([cam_x, cam_y])
            pred_coords.append([pred_x, pred_y])
    
    gt_coords = np.array(gt_coords)
    pred_coords = np.array(pred_coords)
    
    # Calculate errors
    errors = np.sqrt(np.sum((gt_coords - pred_coords)**2, axis=1))
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    
    print(f"\n📊 Results:")
    print(f"   Mean Error: {mean_error:.1f} pixels")
    print(f"   Median Error: {median_error:.1f} pixels")
    print(f"   Min Error: {errors.min():.1f} pixels")
    print(f"   Max Error: {errors.max():.1f} pixels")
    
    return gt_coords, pred_coords, errors

def create_cnn_demonstration(model, map_path, gt_coords, pred_coords, errors,
                           tile_size, zoom_factor, device, base_dir):
    """Create educational visualization showing CNN input/output process."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import matplotlib.patches as patches

    print("\n--- Creating CNN Demonstration ---")

    # Select an interesting example (medium error around middle of flight)
    example_idx = len(errors) // 2

    # Find a frame with a reasonable error (not too small, not too large)
    sorted_indices = np.argsort(errors)
    middle_range_start = len(errors) // 3
    middle_range_end = 2 * len(errors) // 3
    example_idx = sorted_indices[middle_range_start + (middle_range_end - middle_range_start) // 2]

    gt_pos = gt_coords[example_idx]
    pred_pos = pred_coords[example_idx]
    error = errors[example_idx]

    print(f"Using frame {example_idx} with error: {error:.1f} pixels")

    # Create the input image that went into the CNN
    full_map = Image.open(map_path).convert('RGB')
    map_width, map_height = full_map.size

    # Calculate crop area (same as in evaluation)
    crop_width = tile_size[0] // zoom_factor
    crop_height = tile_size[1] // zoom_factor

    left = int(gt_pos[0] - crop_width / 2)
    top = int(gt_pos[1] - crop_height / 2)
    right = left + crop_width
    bottom = top + crop_height

    # Crop and resize to model input size
    cropped = full_map.crop((left, top, right, bottom))
    input_image = cropped.resize(tile_size, Image.LANCZOS)

    # Create the visualization
    fig = plt.figure(figsize=(20, 12))

    # 1. Input Image (large, left side)
    ax1 = plt.subplot(2, 3, (1, 4))
    ax1.imshow(input_image)

    # Mark the actual position on the input image
    actual_x_in_crop = gt_pos[0] - left
    actual_y_in_crop = gt_pos[1] - top

    # Scale to image coordinates
    img_x = (actual_x_in_crop / crop_width) * tile_size[0]
    img_y = (actual_y_in_crop / crop_height) * tile_size[1]

    # Draw crosshairs for actual position
    ax1.axhline(y=img_y, color='lime', linewidth=4, alpha=0.9)
    ax1.axvline(x=img_x, color='lime', linewidth=4, alpha=0.9)
    ax1.plot(img_x, img_y, 'o', color='lime', markersize=20,
             markeredgecolor='black', markeredgewidth=3, label='Ground Truth Location')

    ax1.set_title('CNN Input: Terrain Image (1200×675)', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('This is exactly what the model sees', fontsize=14, style='italic')
    ax1.legend(fontsize=14, loc='upper right')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 2. Model Architecture Diagram (top right)
    ax2 = plt.subplot(2, 3, 2)
    ax2.axis('off')

    arch_text = """CNN ARCHITECTURE

Input: 1200×675×3 RGB Image
         ↓
Convolutional Layers:
• 32 filters (7×7) → 600×338
• 64 filters (5×5) → 150×85
• 128 filters (3×3) → 38×21
• 256 filters (3×3) → 10×5
         ↓
Global Average Pool → 256
         ↓
Dense Layers:
• 256 → 128 → 2
         ↓
Sigmoid → [x, y] ∈ [0,1]²
         ↓
Scale to Map Coordinates"""

    ax2.text(0.05, 0.95, arch_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='navy'))

    # 3. Prediction Output (middle right)
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')

    # Calculate normalized coordinates for display
    pred_norm_x = pred_pos[0] / map_width
    pred_norm_y = pred_pos[1] / map_height
    gt_norm_x = gt_pos[0] / map_width
    gt_norm_y = gt_pos[1] / map_height

    output_text = f"""MODEL OUTPUT

Frame: {example_idx}/150

Normalized Prediction:
x = {pred_norm_x:.6f}
y = {pred_norm_y:.6f}

Map Coordinates:
Predicted: ({pred_pos[0]:.0f}, {pred_pos[1]:.0f})
Actual:    ({gt_pos[0]:.0f}, {gt_pos[1]:.0f})

Position Error: {error:.1f} pixels
Relative Error: {(error/map_width)*100:.3f}% of map

RESULT: {'✅ GOOD' if error < 100 else '⚠️ FAIR' if error < 200 else '❌ POOR'}"""

    color = 'lightgreen' if error < 100 else 'lightyellow' if error < 200 else 'lightcoral'
    ax3.text(0.05, 0.95, output_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.9, edgecolor='darkgreen'))

    # 4. Map Context (bottom right)
    ax4 = plt.subplot(2, 3, (5, 6))

    # Show zoomed area around the prediction
    context_size = 1500  # Show 1500x1500 pixel area
    context_left = max(0, int(gt_pos[0] - context_size/2))
    context_top = max(0, int(gt_pos[1] - context_size/2))
    context_right = min(map_width, context_left + context_size)
    context_bottom = min(map_height, context_top + context_size)

    context_map = full_map.crop((context_left, context_top, context_right, context_bottom))
    ax4.imshow(context_map, extent=[context_left, context_right, context_bottom, context_top])

    # Show the input crop area
    rect = patches.Rectangle((left, top), right-left, bottom-top,
                           linewidth=3, edgecolor='blue', facecolor='none',
                           linestyle='--', label='CNN Input Area')
    ax4.add_patch(rect)

    # Show ground truth and prediction
    ax4.plot(gt_pos[0], gt_pos[1], 'o', color='lime', markersize=15,
             markeredgecolor='black', markeredgewidth=3, label='Ground Truth')
    ax4.plot(pred_pos[0], pred_pos[1], 's', color='red', markersize=15,
             markeredgecolor='black', markeredgewidth=3, label='Model Prediction')

    # Draw error circle
    circle = Circle((pred_pos[0], pred_pos[1]), error, fill=False,
                   edgecolor='red', linewidth=3, linestyle='--', alpha=0.8, label='Error Radius')
    ax4.add_patch(circle)

    ax4.set_xlim(context_left, context_right)
    ax4.set_ylim(context_bottom, context_top)
    ax4.set_title('Map Context: Model vs Reality', fontsize=16, fontweight='bold')
    ax4.legend(fontsize=12, loc='upper left')
    ax4.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax4.set_ylabel('Y Coordinate (pixels)', fontsize=12)

    plt.suptitle('CNN Terrain Navigation: Live Prediction Analysis',
                 fontsize=20, fontweight='bold', y=0.96)
    plt.tight_layout()

    # Save the visualization
    output_path = os.path.join(base_dir, 'images/cnn_prediction_demo.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ CNN demonstration saved to {output_path}")

    plt.close()  # Close to free memory

    return example_idx, error

def main():
    """Complete pipeline."""
    print("="*60)
    print("CORRIDOR NAVIGATION EXPERIMENT")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Configuration
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, 'artifacts/corridor_dataset.pkl')
    map_path = os.path.join(base_dir, '../../data/boneyard/davis_monthan_stitched_map.jpg')
    
    start_coord = (5500, 4500)
    end_coord = (4167, 4167)
    tile_size = (1200, 675)
    zoom_factor = 4
    num_frames = 150
    
    # Load dataset
    print("\n--- Loading Dataset ---")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    tiles = data['tiles']
    coordinates = data['coordinates']
    print(f"Loaded {len(tiles)} samples")
    
    # Split data
    split_idx = int(0.8 * len(tiles))
    train_tiles, val_tiles = tiles[:split_idx], tiles[split_idx:]
    train_coords, val_coords = coordinates[:split_idx], coordinates[split_idx:]
    
    train_dataset = CorridorDataset(train_tiles, train_coords)
    val_dataset = CorridorDataset(val_tiles, val_coords)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train model
    model = CorridorCNN()
    model = train_model(model, train_loader, val_loader, device, epochs=20)
    
    # Evaluate on actual flight path
    gt_coords, pred_coords, errors = evaluate_on_flight_path(
        model, map_path, start_coord, end_coord,
        tile_size, zoom_factor, num_frames, device
    )

    # Create educational CNN demonstration
    demo_frame, demo_error = create_cnn_demonstration(
        model, map_path, gt_coords, pred_coords, errors,
        tile_size, zoom_factor, device, base_dir
    )

    # Save results
    results = {
        'ground_truth': gt_coords,
        'predictions': pred_coords,
        'errors': errors,
        'demo_frame': demo_frame,
        'demo_error': demo_error
    }

    results_path = os.path.join(base_dir, 'artifacts/flight_evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✅ Results saved to {results_path}")
    print(f"📊 Educational demonstration created for frame {demo_frame} (error: {demo_error:.1f}px)")
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
