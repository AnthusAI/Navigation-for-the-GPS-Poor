#!/usr/bin/env python3
"""
Command-line interface for Chapter 4 deep learning experiments.

This lets you run each part separately without notebooks.
"""

import sys
sys.path.insert(0, '../..')

import argparse
import numpy as np
import cv2
import torch
from pathlib import Path

from src.navigation.deep_learning import (
    PoseNet, ImprovedPoseNet, SmallPoseNet, MediumPoseNet, LargePoseNet,
    ResNetPoseNet, CoordConvPoseNet, AttentionPoseNet,
    FlightDataset, train_model, evaluate_model,
    ArtifactCache, create_or_load_dataset, train_or_load_model,
    get_device, count_parameters
)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def load_image():
    """Load the aerial image."""
    image_path = Path('../../data/boneyard/davis_monthan_aerial.jpg')
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    aerial_image = cv2.imread(str(image_path))
    aerial_image_rgb = cv2.cvtColor(aerial_image, cv2.COLOR_BGR2RGB)
    print(f"✅ Loaded image: {aerial_image_rgb.shape}")
    return aerial_image_rgb


def cmd_generate_dataset(args):
    """Generate a flight dataset."""
    print(f"\n{'='*80}")
    print(f"GENERATING DATASET: {args.name}")
    print(f"{'='*80}\n")
    
    image = load_image()
    cache = ArtifactCache('artifacts')
    
    frames, poses = create_or_load_dataset(
        cache, args.name, image,
        num_samples=args.samples,
        frame_size=(224, 224),
        seed=42,
        force_regenerate=args.force
    )
    
    print(f"\n✅ Dataset ready: {len(frames)} samples")
    print(f"   Poses shape: {poses.shape}")
    print(f"   Range: x=[{poses[:, 0].min():.3f}, {poses[:, 0].max():.3f}], y=[{poses[:, 1].min():.3f}, {poses[:, 1].max():.3f}]")


def cmd_train(args):
    """Train a model."""
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL: {args.model}")
    print(f"{'='*80}\n")
    
    device = get_device()
    print(f"Device: {device}\n")
    
    cache = ArtifactCache('artifacts')
    image = load_image()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    frames, poses = create_or_load_dataset(
        cache, args.dataset, image,
        num_samples=args.samples,
        frame_size=(224, 224),
        seed=42
    )
    
    # Split data
    train_size = int(0.7 * len(frames))
    val_size = int(0.15 * len(frames))
    
    train_frames, train_poses = frames[:train_size], poses[:train_size]
    val_frames, val_poses = frames[train_size:train_size+val_size], poses[train_size:train_size+val_size]
    test_frames, test_poses = frames[train_size+val_size:], poses[train_size+val_size:]
    
    print(f"Train: {len(train_frames)}, Val: {len(val_frames)}, Test: {len(test_frames)}\n")
    
    # Create datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transform
    
    train_dataset = FlightDataset(train_frames, train_poses, transform=train_transform)
    val_dataset = FlightDataset(val_frames, val_poses, transform=transform)
    test_dataset = FlightDataset(test_frames, test_poses, transform=transform)
    
    batch_size = 32 if device.type in ['cuda', 'mps'] else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model_classes = {
        'posenet': PoseNet,
        'improved': ImprovedPoseNet,
        'small': SmallPoseNet,
        'medium': MediumPoseNet,
        'large': LargePoseNet,
        'resnet': ResNetPoseNet,
        'coordconv': CoordConvPoseNet,
        'attention': AttentionPoseNet,
    }
    
    if args.model not in model_classes:
        print(f"❌ Unknown model: {args.model}")
        print(f"Available: {', '.join(model_classes.keys())}")
        sys.exit(1)
    
    model = model_classes[args.model]().to(device)
    print(f"Model: {args.model}")
    print(f"Parameters: {count_parameters(model):,}\n")
    
    # Train
    model, history = train_or_load_model(
        cache, args.name, model,
        train_loader, val_loader, device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        use_scheduler=True,
        verbose=True,
        force_retrain=args.force
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Final val loss: {history['val_losses'][-1]:.6f}")
    
    # Evaluate on test set
    if not args.no_eval:
        print(f"\nEvaluating on test set...")
        results = evaluate_model(model, test_loader, device, image.shape[:2])
        print(f"   Mean error: {results['errors'].mean():.1f} pixels")
        print(f"   Median error: {np.median(results['errors']):.1f} pixels")
        
        cache.save_results(f"{args.name}_eval", results)
        print(f"   Saved results to artifacts/{args.name}_eval_results.pkl")


def cmd_evaluate(args):
    """Evaluate a trained model."""
    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL: {args.name}")
    print(f"{'='*80}\n")
    
    device = get_device()
    cache = ArtifactCache('artifacts')
    image = load_image()
    
    # Load dataset
    frames, poses = create_or_load_dataset(
        cache, args.dataset, image,
        num_samples=args.samples,
        seed=42
    )
    
    # Test set
    train_size = int(0.7 * len(frames))
    val_size = int(0.15 * len(frames))
    test_frames, test_poses = frames[train_size+val_size:], poses[train_size+val_size:]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = FlightDataset(test_frames, test_poses, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Load model
    model_classes = {
        'posenet': PoseNet,
        'improved': ImprovedPoseNet,
        'small': SmallPoseNet,
        'medium': MediumPoseNet,
        'large': LargePoseNet,
        'resnet': ResNetPoseNet,
        'coordconv': CoordConvPoseNet,
        'attention': AttentionPoseNet,
    }
    
    model = model_classes[args.model]().to(device)
    
    try:
        cache.load_model(args.name, model)
        print(f"✅ Loaded model from artifacts/{args.name}_model.pth\n")
    except FileNotFoundError:
        print(f"❌ Model not found: {args.name}")
        sys.exit(1)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, image.shape[:2])
    
    print(f"\nResults:")
    print(f"  Mean error: {results['errors'].mean():.1f} pixels")
    print(f"  Median error: {np.median(results['errors']):.1f} pixels")
    print(f"  Std dev: {results['errors'].std():.1f} pixels")
    print(f"  Min error: {results['errors'].min():.1f} pixels")
    print(f"  Max error: {results['errors'].max():.1f} pixels")


def cmd_list(args):
    """List cached artifacts."""
    cache = ArtifactCache('artifacts')
    
    print(f"\n{'='*80}")
    print("CACHED ARTIFACTS")
    print(f"{'='*80}\n")
    
    if not cache.cache_dir.exists():
        print("No artifacts directory")
        return
    
    datasets = list(cache.cache_dir.glob('*_dataset.pkl'))
    models = list(cache.cache_dir.glob('*_model.pth'))
    results = list(cache.cache_dir.glob('*_results.pkl'))
    
    if datasets:
        print(f"Datasets ({len(datasets)}):")
        for d in sorted(datasets):
            size = d.stat().st_size / 1024 / 1024
            print(f"  - {d.stem.replace('_dataset', ''):<30} ({size:.1f} MB)")
    
    if models:
        print(f"\nModels ({len(models)}):")
        for m in sorted(models):
            size = m.stat().st_size / 1024 / 1024
            print(f"  - {m.stem.replace('_model', ''):<30} ({size:.1f} MB)")
    
    if results:
        print(f"\nResults ({len(results)}):")
        for r in sorted(results):
            size = r.stat().st_size / 1024
            print(f"  - {r.stem.replace('_results', ''):<30} ({size:.1f} KB)")
    
    if not (datasets or models or results):
        print("No cached artifacts found")


def cmd_clear(args):
    """Clear cached artifacts."""
    cache = ArtifactCache('artifacts')
    
    if args.type:
        print(f"Clearing {args.type} artifacts...")
        cache.clear(args.type)
    else:
        print(f"Clearing all artifacts...")
        cache.clear()
    
    print("✅ Done")


def main():
    parser = argparse.ArgumentParser(
        description='Chapter 4: Deep Learning for Visual Navigation CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a dataset
  python cli.py generate --name train_1000 --samples 1000
  
  # Train a small model (fast)
  python cli.py train --name small_v1 --model small --dataset train_1000 --epochs 20
  
  # Train with more data and augmentation
  python cli.py train --name improved_v1 --model improved --dataset train_5000 --samples 5000 --epochs 40 --augment
  
  # Evaluate a model
  python cli.py evaluate --name improved_v1 --model improved --dataset train_5000 --samples 5000
  
  # List what's cached
  python cli.py list
  
  # Clear cache
  python cli.py clear
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate dataset
    gen_parser = subparsers.add_parser('generate', help='Generate a flight dataset')
    gen_parser.add_argument('--name', required=True, help='Dataset name')
    gen_parser.add_argument('--samples', type=int, default=1000, help='Number of samples (default: 1000)')
    gen_parser.add_argument('--force', action='store_true', help='Force regeneration')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--name', required=True, help='Model name for saving')
    train_parser.add_argument('--model', required=True, help='Model architecture (posenet, small, improved, coordconv, etc.)')
    train_parser.add_argument('--dataset', required=True, help='Dataset name to use')
    train_parser.add_argument('--samples', type=int, default=1000, help='Number of samples (default: 1000)')
    train_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    train_parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    train_parser.add_argument('--force', action='store_true', help='Force retraining')
    train_parser.add_argument('--no-eval', action='store_true', help='Skip test evaluation')
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--name', required=True, help='Model name to load')
    eval_parser.add_argument('--model', required=True, help='Model architecture')
    eval_parser.add_argument('--dataset', required=True, help='Dataset name')
    eval_parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    
    # List artifacts
    subparsers.add_parser('list', help='List cached artifacts')
    
    # Clear cache
    clear_parser = subparsers.add_parser('clear', help='Clear cached artifacts')
    clear_parser.add_argument('--type', choices=['dataset', 'model', 'history', 'results'], help='Type to clear (default: all)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    commands = {
        'generate': cmd_generate_dataset,
        'train': cmd_train,
        'evaluate': cmd_evaluate,
        'list': cmd_list,
        'clear': cmd_clear,
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()

