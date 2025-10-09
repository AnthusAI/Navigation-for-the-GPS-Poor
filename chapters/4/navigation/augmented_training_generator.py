"""
Augmented Training Data Generator

Generates realistic training data using the validated augmentation pipeline with:
- Proper aircraft perspective rotation based on flight paths
- Realistic environmental effects from curated presets
- Variable scaling to simulate different altitudes
- DRY TerrainWindow integration for zero letterboxing
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import random
from datetime import datetime

from .terrain_window import TerrainWindow
from .flight_config import FlightPathConfig
from .environmental_presets import EnvironmentalPresets


class AugmentedTrainingGenerator:
    """Generate realistic training data with proper augmentation."""

    def __init__(self):
        """Initialize augmented training data generator."""
        self.terrain_window = TerrainWindow()
        print("✅ Augmented Training Data Generator initialized")

    def generate_flight_training_dataset(self,
                                       flight_name: str = "main_evaluation",
                                       num_samples: int = 5000,
                                       corridor_width: float = 1000.0,
                                       tile_size: int = 224,
                                       effect_probability: float = 0.8,
                                       save_path: Optional[str] = None) -> Dict:
        """
        Generate augmented training dataset along flight corridor.

        Args:
            flight_name: Name of flight path to follow
            num_samples: Number of training samples to generate
            corridor_width: Width of corridor around flight path (meters)
            tile_size: Size of terrain tiles (pixels)
            effect_probability: Probability of applying environmental effects
            save_path: Optional path to save dataset

        Returns:
            Dictionary with training data and metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n🎯 Generating Augmented Training Dataset")
        print("=" * 50)
        print(f"   Flight: {flight_name}")
        print(f"   Samples: {num_samples}")
        print(f"   Corridor width: {corridor_width}m")
        print(f"   Environmental effects: {effect_probability:.1%} probability")
        print(f"   Tile size: {tile_size}×{tile_size}")

        # Get flight path configuration
        flight_path = FlightPathConfig.get_flight_path(flight_name)
        flight_coordinates = FlightPathConfig.create_flight_coordinates(flight_path)

        print(f"   Route: {flight_path.description}")
        print(f"   Flight points: {len(flight_coordinates)}")

        # Convert to pixel coordinates for terrain extraction
        pixel_coords = flight_coordinates * 7500  # Scale to map coordinates

        # Calculate corridor sampling bounds
        max_distance_from_path = corridor_width / 10  # Convert meters to map pixels (rough)

        # Generate training samples
        terrain_tiles = []
        coordinates = []
        headings = []
        environmental_configs = []
        altitude_configs = []

        print(f"\n🔄 Generating {num_samples} augmented training samples...")

        progress_interval = max(1, num_samples // 20)  # Show progress every 5%

        for i in range(num_samples):
            try:
                # Select random point along flight path
                path_idx = random.randint(0, len(pixel_coords) - 1)
                base_coord = pixel_coords[path_idx]

                # Add random offset within corridor
                offset_x = random.uniform(-max_distance_from_path, max_distance_from_path)
                offset_y = random.uniform(-max_distance_from_path, max_distance_from_path)

                sample_x = base_coord[0] + offset_x
                sample_y = base_coord[1] + offset_y

                # Calculate aircraft heading based on flight path direction
                aircraft_heading = self._calculate_aircraft_heading(
                    pixel_coords, path_idx, sample_x, sample_y
                )

                # Generate altitude simulation parameters
                altitude_params = EnvironmentalPresets.get_altitude_simulation_params()
                zoom_factor = altitude_params['zoom']

                # Generate environmental effects (if enabled)
                environmental_effects = None
                if EnvironmentalPresets.should_apply_effects(effect_probability):
                    environmental_effects = EnvironmentalPresets.get_realistic_environmental_effects()
                    preset_name = environmental_effects.pop('_preset_name', 'Custom')
                else:
                    preset_name = 'No Effects'

                # Extract augmented terrain using enhanced TerrainWindow
                terrain_tile = self.terrain_window.extract_model_input(
                    sample_x, sample_y,
                    model_input_size=tile_size,
                    zoom=zoom_factor,
                    aircraft_heading=aircraft_heading,
                    environmental_effects=environmental_effects
                )

                # Convert to normalized coordinates
                normalized_coord = (sample_x / 7500, sample_y / 7500)

                # Store sample data
                terrain_tiles.append(terrain_tile)
                coordinates.append(normalized_coord)
                headings.append(aircraft_heading)
                environmental_configs.append(preset_name)
                altitude_configs.append(altitude_params['altitude_scenario'])

                # Show progress
                if i % progress_interval == 0 or i == num_samples - 1:
                    completion = (i + 1) / num_samples
                    print(f"   Progress: {i+1:5d}/{num_samples} ({completion:.1%}) | "
                          f"Heading: {aircraft_heading:5.1f}° | Zoom: {zoom_factor:.2f}x | {preset_name}")

            except (ValueError, IndexError) as e:
                # Skip samples that are outside map bounds or have other issues
                print(f"   Warning: Skipping sample {i} - {e}")
                continue

        print(f"\n✅ Generated {len(terrain_tiles)} augmented training samples")

        # Prepare dataset
        dataset = {
            'tiles': terrain_tiles,
            'coordinates': np.array(coordinates),
            'headings': np.array(headings),
            'environmental_presets': environmental_configs,
            'altitude_scenarios': altitude_configs,
            'metadata': {
                'flight_name': flight_name,
                'flight_description': flight_path.description,
                'num_samples': len(terrain_tiles),
                'corridor_width_meters': corridor_width,
                'tile_size': tile_size,
                'effect_probability': effect_probability,
                'generation_timestamp': timestamp,
                'augmentation_type': 'realistic_aircraft_perspective'
            }
        }

        # Save dataset
        if save_path is None:
            save_path = f"training_datasets/augmented_flight_dataset_{flight_name}_{timestamp}.pkl"

        self._save_dataset(dataset, save_path)

        # Generate summary statistics
        summary = self._generate_dataset_summary(dataset)

        return {
            'dataset_path': save_path,
            'num_samples': len(terrain_tiles),
            'summary': summary,
            'timestamp': timestamp
        }

    def _calculate_aircraft_heading(self, pixel_coords: np.ndarray, path_idx: int,
                                  sample_x: float, sample_y: float) -> float:
        """
        Calculate aircraft heading based on flight path direction.

        Args:
            pixel_coords: Flight path coordinates in pixels
            path_idx: Index of nearest flight path point
            sample_x, sample_y: Sample coordinates

        Returns:
            Aircraft heading in degrees (0-360)
        """
        # Calculate direction based on flight path progression
        if path_idx < len(pixel_coords) - 1:
            # Use forward direction
            current_point = pixel_coords[path_idx]
            next_point = pixel_coords[path_idx + 1]
            direction_vector = next_point - current_point
        elif path_idx > 0:
            # Use backward direction (at end of path)
            current_point = pixel_coords[path_idx]
            prev_point = pixel_coords[path_idx - 1]
            direction_vector = current_point - prev_point
        else:
            # Single point or start of path
            return 296.2  # Default heading for main_evaluation

        # Calculate heading from direction vector
        heading_rad = np.arctan2(direction_vector[0], -direction_vector[1])
        heading_deg = np.degrees(heading_rad)

        # Normalize to [0, 360] range
        heading_deg = (heading_deg + 360) % 360

        # Add small random variation for turbulence (±5 degrees)
        turbulence = random.uniform(-5, 5)
        heading_deg = (heading_deg + turbulence) % 360

        return heading_deg

    def _save_dataset(self, dataset: Dict, save_path: str):
        """Save dataset to pickle file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"\n💾 Dataset saved: {save_path}")
        print(f"   File size: {file_size_mb:.1f} MB")

    def _generate_dataset_summary(self, dataset: Dict) -> Dict:
        """Generate summary statistics for the dataset."""
        headings = np.array(dataset['headings'])
        coordinates = dataset['coordinates']

        # Environmental preset distribution
        preset_counts = {}
        for preset in dataset['environmental_presets']:
            preset_counts[preset] = preset_counts.get(preset, 0) + 1

        # Altitude scenario distribution
        altitude_counts = {}
        for scenario in dataset['altitude_scenarios']:
            altitude_counts[scenario] = altitude_counts.get(scenario, 0) + 1

        summary = {
            'coordinate_range': {
                'x_min': float(coordinates[:, 0].min()),
                'x_max': float(coordinates[:, 0].max()),
                'y_min': float(coordinates[:, 1].min()),
                'y_max': float(coordinates[:, 1].max()),
            },
            'heading_statistics': {
                'mean': float(headings.mean()),
                'std': float(headings.std()),
                'min': float(headings.min()),
                'max': float(headings.max()),
            },
            'environmental_presets': preset_counts,
            'altitude_scenarios': altitude_counts,
        }

        return summary

    def generate_multiple_flight_datasets(self,
                                        flight_names: List[str],
                                        samples_per_flight: int = 2000,
                                        **kwargs) -> Dict:
        """
        Generate training datasets for multiple flight paths.

        Args:
            flight_names: List of flight path names
            samples_per_flight: Number of samples per flight
            **kwargs: Additional arguments for dataset generation

        Returns:
            Dictionary with results for all flights
        """
        print(f"\n🚁 Generating Multi-Flight Training Dataset")
        print("=" * 50)
        print(f"   Flights: {len(flight_names)}")
        print(f"   Samples per flight: {samples_per_flight}")
        print(f"   Total samples: {len(flight_names) * samples_per_flight}")

        results = {}
        all_datasets = []

        for flight_name in flight_names:
            print(f"\n🛩️  Processing flight: {flight_name}")

            result = self.generate_flight_training_dataset(
                flight_name=flight_name,
                num_samples=samples_per_flight,
                **kwargs
            )

            results[flight_name] = result
            all_datasets.append(result['dataset_path'])

        # Create combined dataset summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_summary = {
            'flights': flight_names,
            'individual_datasets': all_datasets,
            'total_samples': len(flight_names) * samples_per_flight,
            'generation_timestamp': timestamp
        }

        results['combined_summary'] = combined_summary

        print(f"\n✅ Multi-flight dataset generation complete!")
        print(f"   Total flights: {len(flight_names)}")
        print(f"   Total samples: {len(flight_names) * samples_per_flight}")

        return results


def main():
    """Demonstrate augmented training data generation."""
    generator = AugmentedTrainingGenerator()

    # Test environmental presets
    print("🌤️  Testing Environmental Presets...")
    EnvironmentalPresets.demonstrate_presets()

    # Generate small test dataset
    print(f"\n🎯 Generating Test Dataset...")
    result = generator.generate_flight_training_dataset(
        flight_name="main_evaluation",
        num_samples=100,  # Small test
        corridor_width=500.0,
        effect_probability=0.9  # High probability for testing
    )

    print(f"\n📊 Dataset Summary:")
    for key, value in result['summary'].items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()