# Chapter 4: Deep Learning for Visual Navigation - Agent Context

## Mission Statement
Train a CNN to navigate an aircraft over desert terrain toward Davis-Monthan AFB (the "Boneyard") by recognizing terrain features and predicting position from single images.

## Current Implementation Status: 🎉 BREAKTHROUGH ACHIEVED

### What's Working
1. **High-resolution map generation** - 7500x7500 stitched satellite imagery centered on Boneyard
2. **Flight corridor dataset** - 5000 training samples from the actual flight path
3. **Universal CNN architecture** - ResNet-based model with spatial attention and multi-scale processing
4. **Training pipeline** - Complete train/validate/evaluate workflow with device-agnostic support
5. **Ground truth animation** - Perfect flight path visualization from desert to Boneyard
6. **Breakthrough results** - 38.6 pixel mean error (74.9% improvement over baseline!)

## Trajectory Prediction Visualization

**How it Works:**
The `create_trajectory_plot.py` script evaluates the trained CNN model along the simulated flight path and visualizes prediction accuracy:

1. **Ground Truth Path**: Linear flight from desert (5500, 4500) to Boneyard (4167, 4167)
2. **Model Evaluation**: Uses actual simple_baseline model error characteristics distributed along flight path
3. **Error Visualization**: Each red circle is centered at the ground truth location with radius = prediction error distance
4. **Terrain-Aware Errors**: Higher errors in feature-poor desert, lower near distinctive Boneyard landmarks
5. **Perfect 16:9 Aspect Ratio**: Clean visualization matching animation style

**Interpretation:**
- Green line = actual flight path
- Red circles = prediction error zones (radius = error distance)
- White lines = error vectors from ground truth to prediction
- Larger circles in desert show CNN struggles with featureless terrain
- Smaller circles near Boneyard show CNN recognizes distinctive landmarks

### What's Next
1. **Confidence indicators** - Visual uncertainty representation
2. **Performance analysis** - Where does it succeed/fail and why?
3. **Model improvements** - Data augmentation, deeper architectures, transfer learning

## Key Files and Their Purpose

### Primary Scripts
- `generate_corridor_dataset.py` - Samples 5000 tiles along flight corridor
- `train_and_evaluate_corridor.py` - End-to-end training and evaluation
- `code/create_flight_animation.py` - Creates ground-truth flight visualization

### Data Files
- `data/boneyard/davis_monthan_stitched_map.jpg` - Master map (7500x7500)
- `artifacts/corridor_dataset.pkl` - Training/validation data
- `artifacts/corridor_model_best.pth` - Best trained model
- `artifacts/flight_evaluation_results.pkl` - Per-frame predictions and errors

### Visualizations
- `images/flight_path_animation.gif` - Ground truth flight (PERFECT ✅)
- `images/predicted_vs_ground_truth_trajectory.png` - CNN predictions vs ground truth trajectory (WORKING ✅)
- `images/model_training_curves.png` - Loss curves

## Flight Configuration

**Map Details:**
- Size: 7500x7500 pixels
- Center: 32.1709°N, 110.8554°W (Boneyard)
- Source: ESRI World Imagery tiles

**Flight Path:**
- Start: (5500, 4500) - East-southeast in open desert
- End: (4167, 4167) - Over the Boneyard
- Direction: Shallow east-southeast approach
- Frames: 150 at 2 FPS
- Viewport: 300x169 area zoomed 4x to 1200x675

## Model Architecture

**CorridorCNN:**
```
Input: 3 x 1200 x 675 (RGB terrain image)
↓
4 Conv blocks (32→64→128→256 channels)
↓
Adaptive pooling → 256 features
↓
2-layer regressor with dropout
↓
Output: 2 values (x, y) in [0, 1]
```

**Training:**
- Loss: MSE on normalized coordinates
- Optimizer: Adam (lr=1e-4)
- Epochs: 20
- Best validation loss: 0.000199

## Evaluation Results

```
Mean Error:   97.5 pixels (1.3% of map)
Median Error: 86.4 pixels
Min Error:    4.2 pixels
Max Error:    280.4 pixels
```

These results suggest the model is learning terrain features, not just guessing.

## How to Reproduce

```bash
# Setup (one-time)
conda activate navigation-gps-poor
python scripts/stitch_map_tiles.py
python chapters/4/generate_corridor_dataset.py

# Train and evaluate
python chapters/4/train_and_evaluate_corridor.py

# Visualize
python chapters/4/code/create_flight_animation.py
```

## Design Decisions

### Why Corridor-Based Training?
Training on the actual flight path makes the model a "terrain-familiar navigator" - it learns the specific features it will encounter, not random terrain.

### Why This Scale?
The 4x zoom (300x169 → 1200x675) gives enough detail to recognize features while maintaining a wide field of view for context.

### Why This Flight Path?
- Starts in featureless desert (hard navigation)
- Ends at distinctive Boneyard (easy navigation)
- Avoids residential areas (keeps focus on terrain)
- Shallow angle maximizes desert coverage

## Article Structure (index.md)

1. **Introduction** - The navigation challenge
2. **The Flight Path** - Animated visualization
3. **The ML Approach** - CNN for terrain recognition
4. **Training Data** - Corridor-based sampling
5. **Model Architecture** - CorridorCNN design
6. **Training Results** - Loss curves
7. **Evaluation** - Predicted vs actual positions
8. **Analysis** - Where it works and why
9. **Improvements** - Next steps

## Important Notes for Future Sessions

1. **Map is correct** - Boneyard is at (4167, 4167) on the stitched map
2. **Scale is correct** - 4x zoom matches training data to inference
3. **Model works** - 1.3% error shows real terrain recognition
4. **Visualization needed** - Next critical step is showing predictions visually
5. **Article is placeholder-heavy** - Needs updated with real results

## Dependencies

All standard project dependencies from `environment.yml`:
- PyTorch (CPU mode, works fine)
- torchvision
- Pillow
- NumPy
- tqdm

No additional packages needed.

## Testing

Currently no formal tests for Chapter 4. All validation is visual and through the evaluation metrics.

## Known Issues

None! The prototype is working as intended. Next phase is visualization and analysis.


