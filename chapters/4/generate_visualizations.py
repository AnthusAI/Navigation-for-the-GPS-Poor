#!/usr/bin/env python3
"""
Generate ML-based visualizations for Chapter 4.

This script:
1. Trains models using the CLI/src code
2. Generates comparison plots
3. Creates prediction visualizations
4. Shows model performance

Everything uses DRY code from src/ - no duplication!

Note: Some images in index.md are static diagrams/animations (boneyard_flyover.gif,
neural_network_diagram.png, etc.) and don't need regeneration. This script focuses
on ML-generated visualizations.
"""

import sys
sys.path.insert(0, '../..')

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.navigation.deep_learning import (
    SmallPoseNet, ImprovedPoseNet, CoordConvPoseNet,
    FlightDataset, ArtifactCache, get_device, count_parameters,
    create_or_load_dataset, train_or_load_model, evaluate_model
)


def ensure_models_trained():
    """Train the models we need for visualizations using the CLI."""
    print("="*80)
    print("ENSURING MODELS ARE TRAINED")
    print("="*80)
    print()
    
    # We need these models for the visualizations
    models_needed = [
        ('simple_baseline', 'small', 'viz_1k', 1000, 10, False),
        ('improved_model', 'improved', 'viz_5k', 5000, 20, True),
        ('best_model', 'coordconv', 'viz_5k', 5000, 30, True),
    ]
    
    for name, arch, dataset, samples, epochs, augment in models_needed:
        cache = ArtifactCache('artifacts')
        if cache.exists(name, 'model'):
            print(f"✅ {name} already trained")
        else:
            print(f"🔄 Training {name}...")
            cmd = [
                'python', 'cli.py', 'train',
                '--name', name,
                '--model', arch,
                '--dataset', dataset,
                '--samples', str(samples),
                '--epochs', str(epochs)
            ]
            if augment:
                cmd.append('--augment')
            
            subprocess.run(cmd, check=True)
            print(f"✅ {name} trained\n")


def create_sample_frames_visualization():
    """Show sample frames from the flight dataset."""
    print("\nGenerating sample frames visualization...")
    
    cache = ArtifactCache('artifacts')
    image_path = Path('../../data/boneyard/davis_monthan_aerial.jpg')
    aerial_image = cv2.imread(str(image_path))
    aerial_image_rgb = cv2.cvtColor(aerial_image, cv2.COLOR_BGR2RGB)
    
    frames, poses = create_or_load_dataset(
        cache, 'viz_samples', aerial_image_rgb,
        num_samples=100, seed=42
    )
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        idx = np.random.randint(0, len(frames))
        ax.imshow(frames[idx])
        ax.set_title(f'Frame {idx}\nPose: ({poses[idx, 0]:.3f}, {poses[idx, 1]:.3f})')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/sample_frames.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✅ Saved images/sample_frames.png")


def create_model_comparison():
    """Compare different model architectures."""
    print("\nGenerating model comparison...")
    
    device = get_device()
    cache = ArtifactCache('artifacts')
    
    # Get evaluation results
    models_info = [
        ('simple_baseline', 'Small\n(250K params)', SmallPoseNet),
        ('improved_model', 'Improved\n(14M params)', ImprovedPoseNet),
        ('best_model', 'CoordConv\n(30M params)', CoordConvPoseNet),
    ]
    
    errors = []
    labels = []
    
    for name, label, model_class in models_info:
        try:
            results = cache.load_results(f'{name}_eval')
            errors.append(results['errors'])
            labels.append(label)
            print(f"  {label}: {results['errors'].mean():.1f} px")
        except FileNotFoundError:
            print(f"  {label}: No results found")
    
    if errors:
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot
        ax1.boxplot(errors, labels=labels)
        ax1.set_ylabel('Position Error (pixels)')
        ax1.set_title('Model Comparison: Position Error Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Bar plot of means
        means = [e.mean() for e in errors]
        ax2.bar(labels, means, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax2.set_ylabel('Mean Position Error (pixels)')
        ax2.set_title('Model Comparison: Mean Error')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, v in enumerate(means):
            ax2.text(i, v + 10, f'{v:.1f}px', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('images/model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✅ Saved images/model_comparison.png")


def create_training_curves():
    """Plot training curves for the best model."""
    print("\nGenerating training curves...")
    
    cache = ArtifactCache('artifacts')
    
    try:
        history = cache.load_history('best_model')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['train_losses']) + 1)
        ax.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training Progress: CoordConvPoseNet')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/training_curves.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✅ Saved images/training_curves.png")
    except FileNotFoundError:
        print("  No training history found for best_model")


def create_prediction_visualization():
    """Show predictions vs ground truth."""
    print("\nGenerating prediction visualization...")
    
    device = get_device()
    cache = ArtifactCache('artifacts')
    
    image_path = Path('../../data/boneyard/davis_monthan_aerial.jpg')
    aerial_image = cv2.imread(str(image_path))
    aerial_image_rgb = cv2.cvtColor(aerial_image, cv2.COLOR_BGR2RGB)
    
    # Load test data
    frames, poses = create_or_load_dataset(
        cache, 'viz_5k', aerial_image_rgb,
        num_samples=5000, seed=42
    )
    
    train_size = int(0.7 * len(frames))
    val_size = int(0.15 * len(frames))
    test_frames, test_poses = frames[train_size+val_size:], poses[train_size+val_size:]
    
    # Get first 50 test samples
    test_frames = test_frames[:50]
    test_poses = test_poses[:50]
    
    # Load model
    model = CoordConvPoseNet().to(device)
    cache.load_model('best_model', model)
    model.eval()
    
    # Make predictions
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    predictions = []
    with torch.no_grad():
        for frame in test_frames:
            frame_tensor = transform(frame).unsqueeze(0).to(device)
            pred = model(frame_tensor).cpu().numpy()[0]
            predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Denormalize to pixel coordinates
    img_h, img_w = aerial_image_rgb.shape[:2]
    frame_h, frame_w = 224, 224
    
    true_pixels = test_poses * np.array([img_w - frame_w, img_h - frame_h])
    pred_pixels = predictions * np.array([img_w - frame_w, img_h - frame_h])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(aerial_image_rgb, alpha=0.5)
    
    # Plot predictions and ground truth
    ax.scatter(true_pixels[:, 0], true_pixels[:, 1], c='green', s=100, alpha=0.7, label='Ground Truth', marker='o')
    ax.scatter(pred_pixels[:, 0], pred_pixels[:, 1], c='red', s=50, alpha=0.7, label='Predictions', marker='x')
    
    # Draw error lines
    for i in range(len(true_pixels)):
        ax.plot([true_pixels[i, 0], pred_pixels[i, 0]], 
                [true_pixels[i, 1], pred_pixels[i, 1]], 
                'b-', alpha=0.3, linewidth=1)
    
    ax.set_title('CNN Predictions vs Ground Truth (50 Test Samples)', fontsize=14)
    ax.legend(fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/predictions_vs_truth.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✅ Saved images/predictions_vs_truth.png")


def create_static_visualizations():
    """Generate static diagrams and animations."""
    print("\nGenerating static visualizations...")
    
    # Run the visualization scripts from code/ folder
    scripts = [
        ('code/create_boneyard_flyover.py', 'Boneyard flyover animation'),
        ('code/create_challenging_conditions.py', 'Challenging conditions'),
        ('code/create_neural_network_diagram.py', 'Neural network diagram'),
        ('code/create_cnn_filter_diagram.py', 'CNN filter diagram'),
        ('code/create_boneyard_sample.py', 'Boneyard sample image'),
        ('code/create_training_loop_diagram.py', 'Training loop diagram'),
    ]
    
    for script, desc in scripts:
        script_path = Path(script)
        if script_path.exists():
            print(f"  Running {desc}...")
            try:
                # Don't capture output so we can see progress
                subprocess.run(['python', str(script_path)], check=True)
                print(f"  ✅ {desc} complete")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️ {desc} failed: {e}")
        else:
            print(f"  ⚠️ {script} not found")


def main():
    """Generate all visualizations."""
    print("="*80)
    print("CHAPTER 4: GENERATING VISUALIZATIONS")
    print("="*80)
    print()
    
    # Make sure images directory exists
    Path('images').mkdir(exist_ok=True)
    
    # Step 1: Create static visualizations (diagrams, animations)
    print("\n" + "="*80)
    print("STATIC VISUALIZATIONS")
    print("="*80)
    create_static_visualizations()
    
    # Step 2: Train models if needed
    ensure_models_trained()
    
    # Step 3: Generate ML-based visualizations
    print("\n" + "="*80)
    print("ML-BASED VISUALIZATIONS")
    print("="*80)
    
    create_sample_frames_visualization()
    create_model_comparison()
    create_training_curves()
    create_prediction_visualization()
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS GENERATED")
    print("="*80)
    print("\nGenerated files:")
    for img in sorted(Path('images').glob('*.png')):
        print(f"  - {img}")
    for img in sorted(Path('images').glob('*.gif')):
        print(f"  - {img}")


if __name__ == '__main__':
    main()
