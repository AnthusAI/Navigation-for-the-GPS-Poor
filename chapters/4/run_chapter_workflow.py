#!/usr/bin/env python3
"""
Standardized, Repeatable Workflow for Chapter 4: Visual Navigation with Deep Learning

This script provides a complete, automated pipeline for all chapter components:
- Training data generation with realistic flight conditions
- Model training with DenseNet121
- Model evaluation with live image generation
- Visualization generation including animated validation
- Results compilation and reporting

Usage:
    # Run complete workflow
    python run_chapter_workflow.py --full-pipeline

    # Run specific components
    python run_chapter_workflow.py --generate-data --train --evaluate

    # Quick test workflow
    python run_chapter_workflow.py --quick-test
"""

import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import shutil

# Ensure we can import our navigation modules
sys.path.append(str(Path(__file__).parent))


class Chapter4Workflow:
    """Complete automated workflow for Chapter 4."""

    def __init__(self, working_dir: str = None, verbose: bool = True):
        """
        Initialize workflow manager.

        Args:
            working_dir: Working directory (default: current directory)
            verbose: Enable verbose output
        """
        self.working_dir = Path(working_dir or ".")
        self.verbose = verbose
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Directory structure
        self.artifacts_dir = self.working_dir / "artifacts"
        self.training_dir = self.working_dir / "training_datasets"
        self.results_dir = self.working_dir / "workflow_results" / self.timestamp

        # Ensure directories exist
        for dir_path in [self.artifacts_dir, self.training_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Workflow configuration
        self.config = {
            'data_generation': {
                'flight_name': 'main_evaluation',
                'num_samples': 5000,
                'corridor_width': 1000.0,
                'tile_size': 224,
                'effect_probability': 0.7
            },
            'training': {
                'arch': 'basic',
                'epochs': 50,
                'lr': 0.0001,
                'batch_size': 16,
                'noise_prob': 0.7,
                'predict_uncertainty': True,
                'uncertainty_arch': 'scalar'
            },
            'evaluation': {
                'num_points': 20,
                'fps': 2
            }
        }

        self.log(f"🚀 Chapter 4 Workflow Initialized")
        self.log(f"   Working directory: {self.working_dir}")
        self.log(f"   Results directory: {self.results_dir}")
        self.log(f"   Timestamp: {self.timestamp}")

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def run_command(self, command: List[str], description: str) -> Tuple[bool, str, str]:
        """
        Run shell command and capture output.

        Args:
            command: Command and arguments as list
            description: Human-readable description

        Returns:
            Tuple of (success, stdout, stderr)
        """
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                check=True
            )
            self.log(f"✅ Completed: {description}")
            return True, result.stdout, result.stderr

        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed: {description}", "ERROR")
            self.log(f"Exit code: {e.returncode}", "ERROR")
            self.log(f"Stderr: {e.stderr}", "ERROR")
            return False, e.stdout, e.stderr

    def generate_training_data(self) -> Dict:
        """Generate realistic training dataset with aircraft perspective."""
        self.log(f"\n📊 Step 1: Generating Training Data")
        self.log("=" * 50)

        config = self.config['data_generation']

        # Use realistic training data generator
        from navigation.augmented_training_generator import AugmentedTrainingGenerator

        try:
            generator = AugmentedTrainingGenerator()
            results = generator.generate_flight_training_dataset(
                flight_name=config['flight_name'],
                num_samples=config['num_samples'],
                corridor_width=config['corridor_width'],
                tile_size=config.get('tile_size', 224),
                effect_probability=config.get('effect_probability', 0.7)
            )

            self.log(f"✅ Training data generated successfully")
            self.log(f"   Dataset: {results['dataset_path']}")
            self.log(f"   Samples: {results['num_samples']}")
            self.log(f"   Features: Aircraft perspective, realistic effects, variable scaling")

            return {
                'success': True,
                'dataset_path': results['dataset_path'],
                'num_samples': results['num_samples'],
                'summary': results['summary']
            }

        except Exception as e:
            self.log(f"❌ Data generation failed: {e}", "ERROR")
            return {'success': False, 'error': str(e)}

    def train_model(self, dataset_path: str) -> Dict:
        """Train navigation model with realistic conditions."""
        self.log(f"\n🎯 Step 2: Training Navigation Model")
        self.log("=" * 45)

        config = self.config['training']
        model_id = f"model_{self.timestamp}"

        command = [
            "python", "train_model.py",
            "--data", dataset_path,
            "--arch", config['arch'],
            "--epochs", str(config['epochs']),
            "--lr", str(config['lr']),
            "--batch", str(config['batch_size']),
            "--enable-augmentation",
            "--flight-name", self.config['data_generation']['flight_name'],
            "--noise-prob", str(config['noise_prob'])
        ]

        # Add uncertainty estimation if enabled
        if config.get('predict_uncertainty', False):
            command.append("--predict-uncertainty")
            if 'uncertainty_arch' in config:
                command.extend(["--uncertainty-arch", config['uncertainty_arch']])

        success, stdout, stderr = self.run_command(
            command,
            f"Training model ({config['arch']}, {config['epochs']} epochs)"
        )

        if success:
            # Find the most recent model file
            model_files = list(self.artifacts_dir.glob("model_*.pth"))
            if model_files:
                model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            else:
                model_path = None

            return {
                'success': True,
                'model_id': model_id,
                'model_path': str(model_path) if model_path else None,
                'config': config,
                'stdout': stdout,
                'stderr': stderr
            }
        else:
            return {
                'success': False,
                'error': stderr,
                'config': config
            }

    def evaluate_model(self, model_path: str) -> Dict:
        """Evaluate model with live generation."""
        self.log(f"\n📈 Step 3: Model Evaluation")
        self.log("=" * 40)

        config = self.config['evaluation']

        command = [
            "python", "evaluate_augmented_model_live.py",
            "--model", model_path,
            "--flight", self.config['data_generation']['flight_name'],
            "--points", str(config['num_points']),
            "--fps", str(config['fps'])
        ]

        success, stdout, stderr = self.run_command(
            command,
            f"Evaluating model on flight path ({config['num_points']} points)"
        )

        if success:
            return {
                'success': True,
                'model_path': model_path,
                'num_points': config['num_points'],
                'visualizations': {
                    'trajectory': 'images/navigation_flight_trajectory.png',
                    'animation': 'images/evaluation_flight_path.gif'
                },
                'stdout': stdout,
                'stderr': stderr
            }
        else:
            return {
                'success': False,
                'error': stderr
            }

    def compile_results(self,
                       data_result: Dict,
                       training_result: Dict,
                       evaluation_result: Dict) -> Dict:
        """Compile final workflow results."""
        self.log(f"\n📋 Step 4: Compiling Results")
        self.log("=" * 35)

        workflow_summary = {
            'workflow_info': {
                'timestamp': self.timestamp,
                'working_directory': str(self.working_dir),
                'results_directory': str(self.results_dir),
                'config': self.config
            },
            'data_generation': data_result,
            'training': training_result,
            'evaluation': evaluation_result,
            'success': all([
                data_result.get('success', False),
                training_result.get('success', False),
                evaluation_result.get('success', False)
            ])
        }

        # Save results summary
        results_file = self.results_dir / "workflow_summary.json"
        with open(results_file, 'w') as f:
            json.dump(workflow_summary, f, indent=2, default=str)

        self.log(f"✅ Results compiled: {results_file}")

        return workflow_summary

    def run_full_pipeline(self) -> Dict:
        """Run complete workflow pipeline."""
        self.log(f"\n🚀 Running Complete Chapter 4 Workflow Pipeline")
        self.log("=" * 60)

        # Step 1: Generate training data
        data_result = self.generate_training_data()
        if not data_result['success']:
            self.log("❌ Pipeline stopped: Data generation failed", "ERROR")
            return data_result

        dataset_path = data_result['dataset_path']

        # Step 2: Train model
        training_result = self.train_model(dataset_path)
        if not training_result['success']:
            self.log("❌ Pipeline stopped: Training failed", "ERROR")
            return training_result

        model_path = training_result['model_path']

        # Step 3: Evaluate model
        evaluation_result = self.evaluate_model(model_path)

        # Step 4: Compile results
        final_results = self.compile_results(
            data_result, training_result, evaluation_result
        )

        # Print summary
        self.print_workflow_summary(final_results)

        return final_results

    def run_quick_test(self) -> Dict:
        """Run quick test with reduced parameters."""
        self.log(f"\n⚡ Running Quick Test Workflow")
        self.log("=" * 40)

        # Modify config for quick test
        original_config = self.config.copy()
        self.config['data_generation']['num_samples'] = 500
        self.config['training']['epochs'] = 10
        self.config['evaluation']['num_points'] = 10

        try:
            results = self.run_full_pipeline()
            return results
        finally:
            # Restore original config
            self.config = original_config

    def print_workflow_summary(self, results: Dict):
        """Print workflow summary."""
        self.log(f"\n📊 CHAPTER 4 WORKFLOW SUMMARY")
        self.log("=" * 50)

        success_emoji = "✅" if results['success'] else "❌"
        self.log(f"{success_emoji} Overall Success: {results['success']}")

        self.log(f"\n📂 Results Location: {results['workflow_info']['results_directory']}")

        # Component status
        components = [
            ('Data Generation', results['data_generation']),
            ('Model Training', results['training']),
            ('Model Evaluation', results['evaluation'])
        ]

        self.log(f"\n🔍 Component Status:")
        for name, component_result in components:
            status = "✅" if component_result.get('success', False) else "❌"
            self.log(f"   {status} {name}")

        # Visualizations created
        if results['evaluation'].get('success'):
            self.log(f"\n🎨 Visualizations Created:")
            for viz_name, viz_path in results['evaluation'].get('visualizations', {}).items():
                self.log(f"   📊 {viz_name}: {viz_path}")

        # Next steps
        self.log(f"\n🎯 Next Steps:")
        self.log(f"   1. Review evaluation animation showing aircraft camera view")
        self.log(f"   2. Check trajectory visualization for navigation accuracy")
        self.log(f"   3. View training curves and validation metrics")
        self.log(f"   4. Update article with generated visualizations")


def main():
    parser = argparse.ArgumentParser(description="Chapter 4 Standardized Workflow")

    # Workflow modes
    parser.add_argument("--full-pipeline", action="store_true",
                       help="Run complete workflow pipeline")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test workflow with reduced parameters")

    # Individual components
    parser.add_argument("--generate-data", action="store_true",
                       help="Generate training data only")
    parser.add_argument("--train", action="store_true",
                       help="Train model only (requires existing data)")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate model only (requires trained model)")

    # Configuration
    parser.add_argument("--working-dir", default=".",
                       help="Working directory (default: current)")
    parser.add_argument("--config-file", help="JSON config file to override defaults")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose output")

    args = parser.parse_args()

    # Create workflow manager
    workflow = Chapter4Workflow(
        working_dir=args.working_dir,
        verbose=args.verbose
    )

    # Load custom config if provided
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file) as f:
            custom_config = json.load(f)
            workflow.config.update(custom_config)
            workflow.log(f"✅ Loaded custom config: {args.config_file}")

    try:
        # Run requested workflow
        if args.full_pipeline:
            results = workflow.run_full_pipeline()
        elif args.quick_test:
            results = workflow.run_quick_test()
        elif args.generate_data:
            results = {'data_generation': workflow.generate_training_data()}
        else:
            # Default to full pipeline if no specific mode chosen
            workflow.log("No specific mode chosen, running full pipeline...")
            results = workflow.run_full_pipeline()

        # Exit with appropriate code
        success = results.get('success', False)
        if isinstance(results, dict) and 'data_generation' in results and len(results) == 1:
            success = results['data_generation'].get('success', False)

        workflow.log(f"\n🏁 Workflow completed {'successfully' if success else 'with errors'}")
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        workflow.log("\n🛑 Workflow interrupted by user", "ERROR")
        sys.exit(1)
    except Exception as e:
        workflow.log(f"\n💥 Workflow failed with exception: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
