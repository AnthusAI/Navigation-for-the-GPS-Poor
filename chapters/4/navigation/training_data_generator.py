"""
DRY Training Data Generator - Generate training datasets along flight paths

This module uses the DRY TerrainWindow and FlightPathConfig systems to create
training datasets that are properly aligned with the actual flight routes.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pickle
from tqdm import tqdm
import random

from .terrain_window import TerrainWindow
from .flight_config import FlightPathConfig


class TrainingDataGenerator:
    """
    Generate training data along flight paths using DRY terrain projection.

    This class creates training datasets by sampling terrain images and coordinates
    along and around the simulated flight path, ensuring the model trains on
    relevant terrain that it will encounter during navigation.
    """

    def __init__(self, max_distance_from_path: float = 500.0):
        """
        Initialize the training data generator.

        Args:
            max_distance_from_path: Maximum distance in pixels from flight path for sampling
        """
        self.terrain_window = TerrainWindow()
        self.max_distance_from_path = max_distance_from_path

        print(f"✅ Training Data Generator initialized")
        print(f"   Max distance from flight path: {max_distance_from_path} pixels")

    def generate_flight_corridor_samples(self,
                                       flight_name: str = "main_evaluation",
                                       num_samples: int = 5000,
                                       tile_size: int = 224,
                                       corridor_width: float = 1000.0) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        """
        Generate training samples along a flight corridor.

        Args:
            flight_name: Name of the flight path to use
            num_samples: Number of training samples to generate
            tile_size: Size of terrain tiles to extract
            corridor_width: Width of corridor around flight path (pixels)

        Returns:
            Tuple of (terrain_tiles, normalized_coordinates)
        """
        print(f"\n🎯 Generating Flight Corridor Training Data")
        print("=" * 50)

        # Get flight path configuration
        flight_path = FlightPathConfig.get_flight_path(flight_name)
        flight_coords = FlightPathConfig.create_flight_coordinates(flight_path)
        flight_info = FlightPathConfig.get_flight_info(flight_name)

        print(f"Flight: {flight_info['name']}")
        print(f"Route: {flight_info['description']}")
        print(f"Distance: {flight_info['distance_pixels']:.1f} pixels")
        print(f"Corridor width: {corridor_width} pixels")
        print(f"Samples to generate: {num_samples}")

        # Convert to pixel coordinates for sampling
        pixel_coords = flight_coords * np.array([7500, 7500])

        # Generate samples along the corridor
        terrain_tiles = []
        sample_coordinates = []
        sample_locations = []  # Store pixel locations for visualization

        print(f"\n🔄 Sampling terrain along flight corridor...")

        for i in tqdm(range(num_samples), desc="Generating samples"):
            # Choose a random point along the flight path
            path_index = random.randint(0, len(pixel_coords) - 1)
            base_coord = pixel_coords[path_index]

            # Add random offset within corridor width
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, corridor_width / 2)

            offset_x = distance * np.cos(angle)
            offset_y = distance * np.sin(angle)

            sample_x = base_coord[0] + offset_x
            sample_y = base_coord[1] + offset_y

            # Ensure sample is within map bounds
            sample_x = np.clip(sample_x, tile_size//2, 7500 - tile_size//2)
            sample_y = np.clip(sample_y, tile_size//2, 7500 - tile_size//2)

            try:
                # Extract terrain tile using DRY TerrainWindow
                terrain_tile = self.terrain_window.extract_window(
                    sample_x, sample_y, tile_size
                )
                terrain_tiles.append(terrain_tile)

                # Store normalized coordinates for training
                norm_x = sample_x / 7500
                norm_y = sample_y / 7500
                sample_coordinates.append((norm_x, norm_y))
                sample_locations.append((sample_x, sample_y))

            except ValueError as e:
                print(f"Warning: Skipping sample {i} at ({sample_x:.0f}, {sample_y:.0f}): {e}")
                continue

        print(f"\n✅ Generated {len(terrain_tiles)} training samples")
        print(f"   Tile size: {tile_size}×{tile_size}")
        print(f"   Coordinate range: X[{min(c[0] for c in sample_coordinates):.3f}, {max(c[0] for c in sample_coordinates):.3f}]")
        print(f"   Coordinate range: Y[{min(c[1] for c in sample_coordinates):.3f}, {max(c[1] for c in sample_coordinates):.3f}]")

        # Store sample locations for visualization
        self._sample_locations = sample_locations
        self._flight_path_pixels = pixel_coords

        return terrain_tiles, sample_coordinates

    def create_training_coverage_visualization(self,
                                             sample_locations: List[Tuple[float, float]],
                                             flight_path_pixels: np.ndarray,
                                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization showing training data coverage relative to flight path.

        Args:
            sample_locations: List of sample locations in pixel coordinates
            flight_path_pixels: Flight path coordinates in pixels
            save_path: Path to save the visualization

        Returns:
            matplotlib Figure object
        """
        print(f"\n🎨 Creating Training Data Coverage Visualization...")

        # Create 16:9 figure with no margins
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Calculate zoom area around flight path
        # End point (airbase) at 1/3 horizontal, start point (ESE) at 2/3 horizontal
        start_pixel = flight_path_pixels[0]   # ESE desert start
        end_pixel = flight_path_pixels[-1]    # Airbase end

        # Calculate bounds for the zoomed view
        # We want: start (ESE) at 2/3 horizontal, end (airbase) at 1/3 horizontal
        flight_width = start_pixel[0] - end_pixel[0]  # Total flight distance horizontally

        # Calculate view width so start appears at 2/3 and end at 1/3
        # This means the flight spans 1/3 of the view width (from 1/3 to 2/3)
        view_width = flight_width * 3  # Flight occupies middle third

        # Position view so end is at 1/3 from left edge
        view_x_min = end_pixel[0] - view_width / 3
        view_x_max = view_x_min + view_width

        # Calculate Y bounds to maintain 16:9 aspect with flight path centered
        flight_center_y = (start_pixel[1] + end_pixel[1]) / 2
        view_height = view_width * (9/16)  # 16:9 aspect ratio
        view_y_min = flight_center_y - view_height / 2
        view_y_max = flight_center_y + view_height / 2

        # Extract and show zoomed satellite map
        if self.terrain_window.stitched_map is not None:
            # Clip bounds to map limits
            view_x_min = max(0, int(view_x_min))
            view_x_max = min(7500, int(view_x_max))
            view_y_min = max(0, int(view_y_min))
            view_y_max = min(7500, int(view_y_max))

            # Extract zoomed area
            zoomed_map = self.terrain_window.stitched_map[view_y_min:view_y_max, view_x_min:view_x_max]

            # Display map filling entire viewport with no margins
            ax.imshow(zoomed_map, extent=[view_x_min, view_x_max, view_y_max, view_y_min])

        # Set exact view bounds
        ax.set_xlim(view_x_min, view_x_max)
        ax.set_ylim(view_y_max, view_y_min)  # Flip Y for image coordinates

        # Plot flight path
        ax.plot(flight_path_pixels[:, 0], flight_path_pixels[:, 1],
               color='red', linewidth=3, alpha=0.9, zorder=10)

        # Plot training sample locations as actual 224×224 pixel squares
        sample_array = np.array(sample_locations)

        # Add rectangles showing actual training image areas (224×224 pixels each)
        for i, (x, y) in enumerate(sample_array[::10]):  # Show every 10th sample to avoid overcrowding
            # Create rectangle representing actual 224×224 training image
            rect = patches.Rectangle(
                (x - 112, y - 112),  # Center the 224×224 square
                224, 224,            # Actual training image dimensions
                linewidth=0.5,
                edgecolor='blue',
                facecolor='blue',
                alpha=0.08,          # Very transparent so terrain shows through
                zorder=5
            )
            ax.add_patch(rect)

        # Mark start and end points (positioned as requested)
        ax.scatter(flight_path_pixels[0, 0], flight_path_pixels[0, 1],
                  s=150, c='green', marker='o', edgecolors='white', linewidth=2,
                  zorder=15)  # Start at ~2/3 horizontal
        ax.scatter(flight_path_pixels[-1, 0], flight_path_pixels[-1, 1],
                  s=150, c='red', marker='s', edgecolors='white', linewidth=2,
                  zorder=15)  # End at ~1/3 horizontal

        # Remove all axes, labels, and margins for clean terrain-filled view
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

        # Remove any padding/margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            # Save with no padding and proper aspect ratio
            fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
            print(f"✅ Coverage visualization saved: {save_path}")

        return fig

    def save_training_dataset(self,
                            terrain_tiles: List[np.ndarray],
                            coordinates: List[Tuple[float, float]],
                            save_path: str,
                            metadata: Optional[Dict] = None) -> str:
        """
        Save training dataset in pickle format.

        Args:
            terrain_tiles: List of terrain tile arrays
            coordinates: List of normalized coordinates
            save_path: Path to save the dataset
            metadata: Optional metadata to include

        Returns:
            Path to saved dataset
        """
        print(f"\n💾 Saving Training Dataset...")

        # Prepare dataset
        dataset = {
            'tiles': terrain_tiles,
            'coordinates': np.array(coordinates),
            'metadata': metadata or {},
            'generation_info': {
                'generator': 'DRY TrainingDataGenerator',
                'terrain_system': 'TerrainWindow',
                'flight_system': 'FlightPathConfig',
                'num_samples': len(terrain_tiles),
                'tile_size': terrain_tiles[0].shape if terrain_tiles else None,
                'coordinate_range': {
                    'x_min': float(min(c[0] for c in coordinates)),
                    'x_max': float(max(c[0] for c in coordinates)),
                    'y_min': float(min(c[1] for c in coordinates)),
                    'y_max': float(max(c[1] for c in coordinates))
                }
            }
        }

        # Save dataset
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        print(f"✅ Dataset saved: {output_path}")
        print(f"   Samples: {len(terrain_tiles)}")
        print(f"   File size: {file_size_mb:.1f} MB")
        print(f"   Tile shape: {terrain_tiles[0].shape if terrain_tiles else 'N/A'}")

        return str(output_path)

    def create_complete_training_dataset(self,
                                       flight_name: str = "main_evaluation",
                                       num_samples: int = 5000,
                                       corridor_width: float = 1000.0,
                                       tile_size: int = 224,
                                       save_dir: str = "training_datasets") -> Dict[str, str]:
        """
        Complete workflow to create training dataset with visualization.

        Args:
            flight_name: Name of the flight path to use
            num_samples: Number of training samples to generate
            corridor_width: Width of sampling corridor around flight path
            tile_size: Size of terrain tiles
            save_dir: Directory to save outputs

        Returns:
            Dictionary with paths to generated files
        """
        print(f"\n🚀 Creating Complete DRY Training Dataset")
        print("=" * 55)

        # Generate training samples
        terrain_tiles, coordinates = self.generate_flight_corridor_samples(
            flight_name=flight_name,
            num_samples=num_samples,
            tile_size=tile_size,
            corridor_width=corridor_width
        )

        # Create visualization
        coverage_viz = self.create_training_coverage_visualization(
            sample_locations=self._sample_locations,
            flight_path_pixels=self._flight_path_pixels,
            save_path=f"{save_dir}/training_data_coverage_{flight_name}.png"
        )

        # Save dataset
        dataset_path = self.save_training_dataset(
            terrain_tiles=terrain_tiles,
            coordinates=coordinates,
            save_path=f"{save_dir}/flight_corridor_dataset_{flight_name}.pkl",
            metadata={
                'flight_name': flight_name,
                'corridor_width': corridor_width,
                'tile_size': tile_size,
                'generation_method': 'DRY_corridor_sampling'
            }
        )

        # Generate sample visualization
        sample_viz_path = self._create_sample_visualization(
            terrain_tiles, coordinates, f"{save_dir}/training_samples_{flight_name}.png"
        )

        results = {
            'dataset_path': dataset_path,
            'coverage_visualization': f"{save_dir}/training_data_coverage_{flight_name}.png",
            'sample_visualization': sample_viz_path,
            'num_samples': len(terrain_tiles)
        }

        print(f"\n🎉 Complete Training Dataset Created!")
        print(f"   Dataset: {dataset_path}")
        print(f"   Coverage viz: {results['coverage_visualization']}")
        print(f"   Sample viz: {results['sample_visualization']}")

        return results

    def _create_sample_visualization(self,
                                   terrain_tiles: List[np.ndarray],
                                   coordinates: List[Tuple[float, float]],
                                   save_path: str) -> str:
        """Create a visualization showing sample terrain tiles."""
        print(f"\n🖼️ Creating sample visualization...")

        # Select 12 random samples for display
        num_display = min(12, len(terrain_tiles))
        sample_indices = random.sample(range(len(terrain_tiles)), num_display)

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        for i, idx in enumerate(sample_indices):
            tile = terrain_tiles[idx]
            coord = coordinates[idx]

            axes[i].imshow(tile)
            axes[i].set_title(f'Sample {idx}\nCoord: ({coord[0]:.3f}, {coord[1]:.3f})', fontsize=10)
            axes[i].axis('off')

        plt.suptitle('Training Dataset Sample Tiles\n(Generated using DRY TerrainWindow system)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Sample visualization saved: {save_path}")
        return save_path


# Convenience function for easy dataset generation
def create_flight_training_dataset(flight_name: str = "main_evaluation",
                                 num_samples: int = 5000,
                                 corridor_width: float = 1000.0) -> Dict[str, str]:
    """
    Convenience function to create training dataset using DRY systems.

    Args:
        flight_name: Name of the flight path to use
        num_samples: Number of training samples
        corridor_width: Sampling corridor width in pixels

    Returns:
        Dictionary with paths to generated files
    """
    generator = TrainingDataGenerator()
    return generator.create_complete_training_dataset(
        flight_name=flight_name,
        num_samples=num_samples,
        corridor_width=corridor_width
    )