# Chapter 4 Model Improvements

## Current Performance
- **Mean Error**: 278 pixels (~18% of image width)
- **Test RMSE**: 0.206 (normalized)
- **Status**: Working but needs improvement

## Key Issues Identified

1. **Simple CNN Architecture**: The current PoseNet is relatively shallow
2. **Limited Data Augmentation**: Only basic transforms
3. **Small Dataset**: 1000 samples may not be enough
4. **Learning Rate**: May not be optimal
5. **No Dropout/Regularization**: Model might be overfitting

## Recommended Improvements (in order of impact)

### 🔥 HIGH IMPACT - Try These First

#### 1. Increase Dataset Size
```python
# Current: num_samples=1000
# Try: num_samples=5000 or more
frames, poses = generate_flight_dataset(aerial_image_rgb, num_samples=5000)
```
**Expected improvement**: 30-40% reduction in error

#### 2. Add Data Augmentation
```python
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(15),           # Simulate different orientations
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Scale variation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Lighting
    transforms.RandomHorizontalFlip(p=0.3),  # Mirror augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
**Expected improvement**: 20-30% reduction in error

#### 3. Use a Pre-trained Backbone (Transfer Learning)
```python
import torchvision.models as models

class ImprovedPoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pre-trained ResNet18 as feature extractor
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        
        # Custom pose regression head
        self.pose_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # x, y position
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.pose_head(x)
        return x
```
**Expected improvement**: 40-50% reduction in error

### 💡 MEDIUM IMPACT

#### 4. Add Dropout for Regularization
```python
# In PoseNet definition, add dropout layers:
self.dropout = nn.Dropout(0.5)

# In forward():
x = self.dropout(self.fc1(x))
```
**Expected improvement**: 10-15% reduction in error

#### 5. Tune Learning Rate with Scheduler
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# In training loop after validation:
scheduler.step(val_loss)
```
**Expected improvement**: 10-15% reduction in error

#### 6. Increase Training Epochs with Early Stopping
```python
num_epochs = 50  # or even 100
patience = 10
```
**Expected improvement**: 5-10% reduction in error

### 🔬 ADVANCED (Try After Basic Improvements)

#### 7. Add Batch Normalization
```python
self.bn1 = nn.BatchNorm2d(32)
self.bn2 = nn.BatchNorm2d(64)
# Apply after each conv layer
```

#### 8. Ensemble Multiple Models
Train 3-5 models with different random seeds and average predictions

#### 9. Add Spatial Features
Include position hints or grid coordinates as additional input channels

#### 10. Use Coordinate Regression Loss
Instead of MSE, use a custom loss that weights x and y errors differently

## Quick Win Combo (Easiest to Implement)

Try these three together for best results with minimal code changes:

1. **Increase dataset to 5000 samples**
2. **Add dropout (0.5) to fully connected layers**
3. **Train for 50 epochs with learning rate scheduler**

**Expected combined improvement**: 40-50% reduction in mean error (from 278px to ~140-170px)

## Implementation Priority

### Phase 1 (Quick Wins - 1 hour)
1. Increase dataset size → 5000 samples
2. Add dropout layers
3. Increase epochs to 50

### Phase 2 (Medium Effort - 2-3 hours)
4. Add data augmentation
5. Implement learning rate scheduler
6. Add batch normalization

### Phase 3 (Advanced - 4+ hours)
7. Implement transfer learning with ResNet
8. Create ensemble of models
9. Custom loss functions

## Expected Final Performance

With all improvements:
- **Target Mean Error**: 50-80 pixels (~3-5% of image width)
- **Target RMSE**: <0.06 (normalized)

This would make the pose estimation system production-ready for real navigation tasks!

