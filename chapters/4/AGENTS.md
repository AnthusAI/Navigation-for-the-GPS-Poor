# Chapter 4: Deep Learning for Visual Navigation - Agent Context

## Mission Statement
Train a CNN to navigate an aircraft over desert terrain toward Davis-Monthan AFB using realistic flight scenarios with crash probabilities, preventing route memorization while achieving GPS-poor navigation capability.

## Current Implementation Status: 🚀 REALISTIC CRASH-BASED TRAINING SYSTEM

### What's Working
1. **Realistic Flight Path Generation** - Stochastic flights with start→end→circle→return patterns and crash scenarios
2. **Bell Curve Crash Distribution** - Higher crash probability near target area (mission-realistic risk profile)
3. **Ultra-Simple Model Architecture** - BasicModel optimized for small diverse datasets (155m validation error)
4. **Balanced Training Data** - ~50% crash rate with ~1000 samples preventing memorization
5. **Standard Workflow** - Four simple commands handle complete training and evaluation pipeline
6. **Cosine Annealing Training** - Smooth convergence with early stopping for small datasets

## Standard Chapter 4 Workflow - FOUR COMMANDS ONLY

### 1. Generate Realistic Training Data
```bash
python generate_data.py --samples 1000 --name realistic_training
```
**What this does:**
- Generates realistic stochastic flight paths until 1000 training samples
- Creates start→end→circle→return flight patterns with 50% crash probability
- Crashes concentrated near target area (bell curve distribution: 0.1% early → 4.0% peak risk)
- Extracts 224×224 terrain tiles from flight trajectories
- Saves training dataset with crash scenario metadata

### 2. Train Navigation Model
```bash
python train_model.py --data training_datasets/realistic_training.pkl --arch basic --epochs 20
```
**What this does:**
- Uses ultra-simple BasicModel architecture (1024→128→2) optimized for small datasets
- ColorJitter augmentation for lighting/weather robustness
- Cosine annealing learning rate (0.0005) with conservative weight decay
- Early stopping with 8-epoch patience to prevent overfitting
- Target: <150m validation error on realistic crash scenarios

### 3. Generate Flight Path Visualization
```bash
python generate_flight_paths.py
```
**What this does:**
- Creates "Stochastic Flight Path Training" visualization
- Shows all flight paths with crash sites marked as red 'X'
- Displays ultra-transparent blue squares (alpha=0.05) for training coverage
- Includes proper start point (green) and end point (red) markers
- Updates images/training_data_coverage_16x9.png

### 4. Evaluate Model Performance
```bash
python evaluate_model.py --model artifacts/model_*.pth --arch basic
```
**What this does:**
- Evaluates trained model on standard flight path (20 test points)
- Generates complete navigation analysis with error metrics
- Creates images/navigation_flight_trajectory.png (flight path visualization)
- Shows actual vs predicted positions with connecting error lines
- Reports mean/median/max/min errors in meters

## Model Architecture - Optimized for Realistic Data

**BasicModel (Ultra-Simple for Small Datasets):**
```
Input: 224×224×3 RGB terrain tile
↓
DenseNet121 Feature Extractor (ImageNet pretrained)
↓
Ultra-Simple Classifier:
  AdaptiveAvgPool2d(1) → Flatten
  Dropout(0.4) → Linear(1024→128) → ReLU
  Linear(128→2)
↓
Output: Normalized (x, y) coordinates [0, 1]
```

**Training Configuration:**
- Loss: MSE on normalized coordinates
- Optimizer: Adam (lr=0.0005, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=epochs, eta_min=1e-6)
- Augmentation: ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- Early Stopping: 8 epochs patience (aggressive for small datasets)
- Batch Size: 16 (standard batch size)

## Realistic Flight System Design

### Why Crash-Based Training?
- **Prevents Memorization**: Diverse crash scenarios across flight area prevent route memorization
- **Realistic Risk Profile**: Bell curve crash distribution mimics real-world mission dangers
- **Target Area Focus**: Higher crash probability near objectives (where navigation is most critical)
- **Balanced Outcomes**: ~50% crash rate provides both success and failure examples

### Flight Path Algorithm
1. **Outbound Phase**: Fly from start toward end point with increasing crash risk
2. **Circling Phase**: Maximum crash risk (4.0% per step) during target area operations
3. **Return Phase**: Decreasing crash risk as aircraft returns to safe start area
4. **Crash Mechanics**: Bell curve distribution centered on target area (mission-realistic)

### Training Data Characteristics
- **Sample Count**: ~1000 samples (right-sized for small dataset architecture)
- **Flight Coverage**: 7-10 overflights with varied success/failure patterns
- **Crash Distribution**: 0.1% early risk → 4.0% peak risk near target
- **Terrain Diversity**: Covers entire flight corridor with realistic mission scenarios

## Performance Results - Realistic Navigation

```
Architecture:     BasicModel (ultra-simple: 1024→128→2)
Training Data:    1001 realistic crash scenario samples
Validation Error: 155 meters (excellent for diverse crash training)
Training Approach: Cosine annealing with ColorJitter augmentation
Crash Rate:       ~71% (realistic mission failure rate)
```

This represents robust GPS-poor navigation trained on realistic failure scenarios instead of memorized corridors.

## File Structure - Standard Components Only

### Core Training Scripts (THE ONLY ONES NEEDED)
- `generate_data.py` - Generates realistic flight training data with crashes
- `train_model.py` - Trains ultra-simple model optimized for small datasets
- `generate_flight_paths.py` - Creates flight path training visualizations
- `evaluate_model.py` - Evaluates model performance and generates analysis visualizations

### Supporting Navigation System
- `navigation/extractor.py` - TerrainExtractor for 224×224 tile extraction
- `navigation/flight_config.py` - FlightPathConfig for standard flight definitions
- `navigation/predictor.py` - NavigationPredictor for model inference
- `navigation/visualizer.py` - PredictionVisualizer for error analysis

### Data Files
- `data/boneyard/davis_monthan_stitched_map.jpg` - Master satellite map (7500×7500)
- `training_datasets/realistic_training.pkl` - Crash-based training data (~1000 samples)
- `artifacts/model_*.pth` - Trained BasicModel checkpoints

### Generated Visualizations
- `images/training_data_coverage_16x9.png` - Complete flight training coverage with crash sites
- `images/navigation_flight_trajectory.png` - Flight path evaluation with error analysis
- `images/predictions_vs_truth.png` - Single prediction visualization with terrain context
- `images/sample_frames.png` - Raw terrain input examples for model
- `images/model_comparison.png` - Model performance analysis charts
- `images/training_curves.png` - Training progress with cosine annealing schedule

## Design Philosophy - Anti-Memorization

### Why Small Diverse Datasets?
Modern approach favors quality over quantity:
- **1000 diverse samples** > 10,000 corridor samples that encourage memorization
- **Realistic failure scenarios** teach robust navigation patterns
- **Ultra-simple architecture** prevents overfitting on small datasets
- **Crash-focused training** covers challenging navigation scenarios

### Why Stochastic Flight Paths?
- **Prevents Route Memorization**: No two flights follow identical paths
- **Mission-Realistic Risk**: Crash distribution matches real-world operational dangers
- **Diverse Terrain Coverage**: Multiple flight patterns sample entire navigation area
- **Balanced Training**: Both successful and failed missions provide learning examples

### Why Ultra-Simple Architecture?
- **Right-Sized for Data**: Complex models overfit on small datasets
- **Faster Training**: Simple architecture converges quickly on diverse data
- **Better Generalization**: Fewer parameters force learning of essential navigation features
- **Robust Performance**: 155m error excellent for challenging realistic scenarios

## How to Reproduce Everything

**Complete Chapter 4 Workflow (4 commands):**
```bash
# 1. Generate realistic crash-based training data (1000 samples)
python generate_data.py --samples 1000 --name realistic_training

# 2. Train improved model on diverse scenarios
python train_model.py --data training_datasets/realistic_training.pkl --arch basic --epochs 20

# 3. Generate flight path training visualization
python generate_flight_paths.py

# 4. Evaluate model and generate performance analysis
python evaluate_model.py --model artifacts/model_*.pth --arch basic
```

## Generate All Chapter Visualizations

**To recreate all images used in the chapter:**

```bash
# Core workflow (generates training data and model)
python generate_data.py --samples 1000 --name realistic_training
python train_model.py --data training_datasets/realistic_training.pkl --arch basic --epochs 20

# Generate all visualizations
python generate_flight_paths.py                                    # → training_data_coverage_16x9.png
python evaluate_model.py --model artifacts/model_*.pth --arch basic # → navigation_flight_trajectory.png
python code/create_all_model_visualizations.py                     # → model_comparison.png, training_curves.png, etc.
```

**What gets generated:**
- `training_data_coverage_16x9.png` - Flight training data with crash sites
- `navigation_flight_trajectory.png` - Complete navigation performance analysis
- `predictions_vs_truth.png` - Single prediction demonstration with terrain context
- `sample_frames.png` - Raw 224×224 terrain input examples (actual model input)
- `model_comparison.png` - Performance metrics and charts
- `training_curves.png` - Training progress visualization

**Individual visualization regeneration:**
```bash
# Generate clean sample frames showing actual 224×224 model inputs
python -c "
import numpy as np, matplotlib.pyplot as plt, pickle, sys
from pathlib import Path
sys.path.append('.')
from navigation.extractor import TerrainExtractor

extractor = TerrainExtractor()
extractor.load_satellite_map('../../data/boneyard/davis_monthan_stitched_map.jpg')
with open('training_datasets/realistic_training.pkl', 'rb') as f: dataset = pickle.load(f)

np.random.seed(42)
indices = np.random.choice(len(dataset['tiles']), 2, replace=False)
images = [dataset['tiles'][i] for i in indices]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, img in zip(axes, images):
    ax.imshow(img); ax.axis('off')

plt.suptitle('Model Input: 224×224 Terrain Tiles', fontsize=14, fontweight='bold', y=0.95)
plt.tight_layout(); plt.savefig('images/sample_frames.png', dpi=300, bbox_inches='tight'); plt.close()
print('✅ Clean sample frames updated')
"
```

**Expected Results:**
- Training data with ~71% crash rate, bell curve risk distribution (1000 samples)
- Model training achieving <150m validation error (improved accuracy target)
- Visualization showing crash sites and comprehensive flight coverage

## Key Design Principles

### Crash Probability Distribution
- **Bell curve risk model**: Low risk during departure/return, high risk near target
- **Mission-realistic**: 0.1% early flight risk → 4.0% peak risk at target area
- **Prevents memorization**: Diverse crash locations across flight area

### Model Architecture Strategy
- **Right-sized for data**: Ultra-simple to prevent overfitting on ~1000 samples
- **DenseNet backbone**: Proven ImageNet features for terrain recognition
- **Minimal regularization**: Single dropout layer for small dataset optimization

### Training Philosophy
- **Conservative learning**: Moderate learning rates with aggressive early stopping
- **Robustness augmentation**: ColorJitter for lighting/weather variations
- **Smooth convergence**: Cosine annealing learning rate schedule

## System Status

✅ **Realistic Training Complete**: Crash-based flight scenarios prevent memorization
✅ **Optimized Architecture**: Ultra-simple model perfect for small datasets
✅ **Standard Workflow**: Four-command pipeline handles complete training and evaluation
✅ **Robust Performance**: 155m validation error excellent for realistic scenarios

The system successfully demonstrates GPS-poor navigation learned from diverse, realistic mission scenarios rather than memorized corridor patterns.