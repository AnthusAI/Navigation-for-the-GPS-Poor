"""
Quick improvements to add to Chapter 4 notebook.
Copy these cells into your notebook after the current model definition.
"""

# ============================================================================
# NEW CELL: Improved PoseNet with Dropout and Batch Normalization
# ============================================================================

class ImprovedPoseNet(nn.Module):
    """Enhanced CNN for pose estimation with dropout and batch normalization."""
    
    def __init__(self, dropout_rate=0.5):
        super(ImprovedPoseNet, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size after convolutions
        # Input: 224x224 -> After conv layers: 7x7x512 = 25088
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # x, y position
    
    def forward(self, x):
        # Convolutional layers with batch norm and ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# ============================================================================
# NEW CELL: Enhanced Data Augmentation
# ============================================================================

# Training transform with data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),  # Simulate different flight orientations
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Lighting variations
    transforms.RandomHorizontalFlip(p=0.3),  # Mirror augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/test transform (no augmentation)
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("✅ Enhanced transforms created")
print(f"   Training: Includes rotation, color jitter, and flip augmentation")
print(f"   Validation/Test: Standard normalization only")

# ============================================================================
# NEW CELL: Generate Larger Dataset
# ============================================================================

print("Generating larger dataset (5000 samples)...")
print("This will take 1-2 minutes...")

# Generate more training data
frames, poses = generate_flight_dataset(
    aerial_image_rgb, 
    num_samples=5000,  # Increased from 1000
    frame_size=(224, 224)
)

print(f"\n✅ Generated {len(frames)} frames")
print(f"   Pose range: x=[{poses[:, 0].min():.3f}, {poses[:, 0].max():.3f}], "
      f"y=[{poses[:, 1].min():.3f}, {poses[:, 1].max():.3f}]")

# ============================================================================
# NEW CELL: Training with Learning Rate Scheduler
# ============================================================================

# Create improved model
model = ImprovedPoseNet(dropout_rate=0.5).to(device)

# Optimizer with lower initial learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Learning rate scheduler - reduces LR when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5,  # Reduce LR by half
    patience=5,   # Wait 5 epochs before reducing
    verbose=True
)

print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"✅ Optimizer: Adam with initial LR=0.0005, weight_decay=1e-4")
print(f"✅ Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")

# ============================================================================
# NEW CELL: Enhanced Training Loop
# ============================================================================

num_epochs = 50  # Increased from 30
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 15  # Stop if no improvement for 15 epochs

print(f"Starting enhanced training...")
print(f"  Epochs: {num_epochs}")
print(f"  Early stopping patience: {early_stop_patience}")
print(f"  Batch size: {BATCH_SIZE}")
print()

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
    for frames_batch, poses_batch in train_pbar:
        frames_batch = frames_batch.to(device)
        poses_batch = poses_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames_batch)
        loss = criterion(outputs, poses_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        for frames_batch, poses_batch in val_pbar:
            frames_batch = frames_batch.to(device)
            poses_batch = poses_batch.to(device)
            
            outputs = model(frames_batch)
            loss = criterion(outputs, poses_batch)
            val_loss += loss.item()
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    val_loss /= len(val_loader)
    
    # Learning rate scheduler step
    scheduler.step(val_loss)
    
    # Print epoch summary
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'images/best_improved_posenet.pth')
        print(f"✅ Saved best model (val_loss: {val_loss:.6f})")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= early_stop_patience:
        print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
        print(f"   No improvement for {early_stop_patience} consecutive epochs")
        break

print(f"\n✅ Training complete! Best validation loss: {best_val_loss:.6f}")

# ============================================================================
# NEW CELL: Compare Original vs Improved Model
# ============================================================================

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# Load and evaluate original model
original_model = PoseNet().to(device)
original_model.load_state_dict(torch.load('images/best_posenet.pth'))
original_model.eval()

# Load and evaluate improved model
improved_model = ImprovedPoseNet().to(device)
improved_model.load_state_dict(torch.load('images/best_improved_posenet.pth'))
improved_model.eval()

# Evaluate both on test set
with torch.no_grad():
    original_preds = []
    improved_preds = []
    test_targets_list = []
    
    for frames_batch, poses_batch in test_loader:
        frames_batch = frames_batch.to(device)
        
        original_out = original_model(frames_batch).cpu().numpy()
        improved_out = improved_model(frames_batch).cpu().numpy()
        
        original_preds.append(original_out)
        improved_preds.append(improved_out)
        test_targets_list.append(poses_batch.numpy())

original_preds = np.vstack(original_preds)
improved_preds = np.vstack(improved_preds)
test_targets = np.vstack(test_targets_list)

# Calculate errors in pixels
h, w = aerial_image_rgb.shape[:2]
frame_h, frame_w = 224, 224

def calc_pixel_errors(preds, targets):
    pred_x = preds[:, 0] * (w - frame_w) + frame_w // 2
    pred_y = preds[:, 1] * (h - frame_h) + frame_h // 2
    true_x = targets[:, 0] * (w - frame_w) + frame_w // 2
    true_y = targets[:, 1] * (h - frame_h) + frame_h // 2
    errors = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
    return errors

original_errors = calc_pixel_errors(original_preds, test_targets)
improved_errors = calc_pixel_errors(improved_preds, test_targets)

print("\nORIGINAL MODEL:")
print(f"  Mean error: {original_errors.mean():.1f} pixels")
print(f"  Median error: {np.median(original_errors):.1f} pixels")
print(f"  Max error: {original_errors.max():.1f} pixels")

print("\nIMPROVED MODEL:")
print(f"  Mean error: {improved_errors.mean():.1f} pixels")
print(f"  Median error: {np.median(improved_errors):.1f} pixels")
print(f"  Max error: {improved_errors.max():.1f} pixels")

improvement = (1 - improved_errors.mean() / original_errors.mean()) * 100
print(f"\n🚀 IMPROVEMENT: {improvement:.1f}% reduction in mean error")
print("=" * 70)

