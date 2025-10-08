#!/usr/bin/env python3
"""
Standard command to generate proper corridor training data.
Usage: python generate_data.py --samples 10000
"""
import argparse
import pickle
import numpy as np
import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from navigation.extractor import TerrainExtractor
from navigation.flight_config import FlightPathConfig

def point_to_line_distance(point, line_start, line_end):
    """Calculate minimum distance from point to line segment."""
    line_vec = line_end - line_start
    point_vec = point - line_start

    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        return np.linalg.norm(point_vec)

    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
    projection = line_start + t * line_vec

    return np.linalg.norm(point - projection)

def is_point_in_corridor(point, flight_path, max_distance):
    """Check if point is within max_distance of any segment of the flight path."""
    for i in range(len(flight_path) - 1):
        segment_start = flight_path[i]
        segment_end = flight_path[i + 1]

        distance = point_to_line_distance(point, segment_start, segment_end)
        if distance <= max_distance:
            return True

    return False

def generate_training_data(num_samples, output_name="training_data"):
    """Generate realistic flight path training data with crashes."""
    print(f"🛩️  Generating Realistic Flight Training Data")
    print(f"   Target samples: {num_samples}")

    # Import flight path generation
    from generate_flight_paths import generate_multiple_flight_paths

    # Generate realistic flight paths until target samples reached
    flight_paths, crash_sites = generate_multiple_flight_paths(target_samples=num_samples)

    print(f"🎯 Extracting terrain tiles from realistic flights...")

    # Setup extractor
    extractor = TerrainExtractor()
    extractor.load_satellite_map("../../data/boneyard/davis_monthan_stitched_map.jpg")

    # Extract training samples from flight paths
    all_training_samples = []

    for i, path in enumerate(flight_paths):
        # Sample exactly 200 training points from this path
        target_samples_per_path = 200
        if len(path) >= target_samples_per_path:
            # Use evenly spaced indices to get exactly 200 points
            indices = np.linspace(0, len(path) - 1, target_samples_per_path, dtype=int)
            training_samples = path[indices]
        else:
            # If path is shorter than 200 points, use all points
            training_samples = path

        all_training_samples.extend(training_samples)

    # Extract terrain tiles and normalize coordinates
    tiles = []
    coordinates = []
    failed_extractions = 0

    for i, (x, y) in enumerate(all_training_samples):
        try:
            # Extract 224x224 tile centered at this position
            tile = extractor.extract_tile(int(x), int(y), 224)
            tiles.append(tile)

            # Normalize coordinates to 0-1 range
            normalized_x = x / 7500.0
            normalized_y = y / 7500.0
            coordinates.append([normalized_x, normalized_y])

            if (i + 1) % 200 == 0:
                success_rate = (len(tiles) / (i + 1)) * 100
                print(f"     {i+1}/{len(all_training_samples)} processed (success: {success_rate:.1f}%)")

        except Exception as e:
            failed_extractions += 1
            continue

    print(f"✅ Successfully extracted {len(tiles)} tiles ({failed_extractions} failed)")

    # Save dataset
    dataset = {
        'tiles': tiles,
        'coordinates': np.array(coordinates, dtype=np.float32),
        'metadata': {
            'num_samples': len(tiles),
            'num_flight_paths': len(flight_paths),
            'completed_flights': len(flight_paths) - len(crash_sites),
            'crashed_flights': len(crash_sites),
            'tile_size': 224,
            'failed_extractions': failed_extractions,
            'extraction_method': 'realistic_flight_paths_with_crashes'
        }
    }

    output_path = f"training_datasets/{output_name}.pkl"
    Path(output_path).parent.mkdir(exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024

    print(f"✅ Saved {len(tiles)} realistic flight samples to {output_path}")
    print(f"   Flight paths: {len(flight_paths)} ({len(flight_paths) - len(crash_sites)} completed, {len(crash_sites)} crashed)")
    print(f"   File size: {file_size_mb:.1f} MB")

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--name", type=str, default="training_data", help="Output dataset name")
    args = parser.parse_args()

    generate_training_data(args.samples, args.name)