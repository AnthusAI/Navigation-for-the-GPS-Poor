# Chapter 4: Quick Start Guide

## TL;DR - What We Have

✅ **Working CNN that navigates by recognizing terrain**
- Trained on 5000 samples from flight corridor
- Achieves 97.5 pixel mean error (~1.3% position error)
- Can predict location from a single terrain image

✅ **Perfect flight path animation** showing ground truth trajectory

❌ **Missing:** Visualization of predicted vs actual positions

## Current Results

```
Training: 20 epochs, best val loss 0.000199
Evaluation: 150 frames along flight path
Mean Error: 97.5 pixels
Median Error: 86.4 pixels
```

## What to Do Next

### 1. Create Prediction Visualization (PRIORITY)
The model has made predictions for all 150 frames. We need to visualize them:

```python
# Load results
import pickle
with open('artifacts/flight_evaluation_results.pkl', 'rb') as f:
    results = pickle.load(f)

# results contains:
# - ground_truth: (150, 2) actual positions
# - predictions: (150, 2) predicted positions  
# - errors: (150,) per-frame errors
```

**Create:**
- Animated GIF showing both positions frame-by-frame
- Static plot with error bars
- Heat map showing where errors are highest

### 2. Update Article
Replace placeholders in `index.md` with:
- Real training curves
- Prediction visualizations
- Error analysis
- Discussion of results

### 3. Improve Model (Optional)
Try:
- Data augmentation (rotations, color jitter)
- More training samples (10k instead of 5k)
- Transfer learning (ResNet backbone)
- Ensemble methods

## File Locations

**Code:**
- `train_and_evaluate_corridor.py` - Main pipeline
- `generate_corridor_dataset.py` - Data generation
- `code/create_flight_animation.py` - Ground truth viz

**Data:**
- `data/boneyard/davis_monthan_stitched_map.jpg` - Map
- `artifacts/corridor_dataset.pkl` - Training data
- `artifacts/corridor_model_best.pth` - Model weights
- `artifacts/flight_evaluation_results.pkl` - **PREDICTIONS ARE HERE**

**Docs:**
- `STATUS.md` - Detailed status
- `AGENTS.md` - Full context for AI agents
- `QUICKSTART.md` - This file

## Run Everything

```bash
conda activate navigation-gps-poor

# Already done (don't need to rerun):
# python scripts/stitch_map_tiles.py
# python chapters/4/generate_corridor_dataset.py
# python chapters/4/train_and_evaluate_corridor.py

# Create prediction visualization:
python chapters/4/code/create_prediction_visualization.py  # <-- CREATE THIS
```

## Key Insight

The model achieves **1.3% position error**, which is actually quite good! This suggests it's genuinely recognizing terrain features, not just guessing. The next step is to **visualize where and why** it succeeds or fails.

## Questions to Answer

1. Does it do better over distinctive features (Boneyard) vs featureless desert?
2. Does error increase as it flies (drift) or stay constant?
3. Can we see it "lock onto" recognizable landmarks?
4. What does the error distribution look like spatially?

All the data to answer these questions is in `flight_evaluation_results.pkl`!


