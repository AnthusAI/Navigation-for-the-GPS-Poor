#!/usr/bin/env python3
"""
Generate realistic flight paths with stochastic deviations for training data.
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from navigation.extractor import TerrainExtractor
from navigation.flight_config import FlightPathConfig

class Aircraft:
    """Simple aircraft model with position, bearing, and speed."""

    def __init__(self, x, y, bearing_deg, speed_pixels_per_step=15):
        self.x = x
        self.y = y
        self.bearing = math.radians(bearing_deg)  # Convert to radians
        self.speed = speed_pixels_per_step

    def update_position(self, deviation_angle_deg=0):
        """Update aircraft position based on current bearing + deviation."""
        # Add deviation to current bearing
        actual_bearing = self.bearing + math.radians(deviation_angle_deg)

        # Update position
        self.x += self.speed * math.cos(actual_bearing)
        self.y += self.speed * math.sin(actual_bearing)

        # Gradually adjust bearing toward deviation (aircraft turns)
        self.bearing += math.radians(deviation_angle_deg * 0.1)  # Gradual turn

    def get_position(self):
        """Get current position as (x, y)."""
        return (self.x, self.y)

def calculate_bearing_to_target(current_pos, target_pos):
    """Calculate bearing in degrees from current position to target."""
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    bearing_rad = math.atan2(dy, dx)
    return math.degrees(bearing_rad)

def generate_stochastic_flight_path(start_pos, end_pos, max_steps=600):
    """Generate realistic flight path: start->end->circle->return OR crash."""

    # Decide if this flight will crash (50% probability - balanced crash rate)
    will_crash = random.random() < 0.5
    print(f"    Flight initialized: {'WILL CRASH' if will_crash else 'NORMAL FLIGHT'}")

    # Calculate initial bearing toward target
    initial_bearing = calculate_bearing_to_target(start_pos, end_pos)

    # Create aircraft
    aircraft = Aircraft(start_pos[0], start_pos[1], initial_bearing, speed_pixels_per_step=20)

    path = [aircraft.get_position()]
    flight_phase = "outbound"  # outbound, circling, returning
    passed_target = False
    circle_center = None
    circle_radius = 300  # pixels

    for step in range(max_steps):
        current_pos = aircraft.get_position()

        # Check for crash during flight (if will_crash is True)
        if will_crash and step > 20:  # Don't crash immediately
            # Calculate bell curve crash probability based on mission progress
            # Higher risk near target (middle of mission), lower risk near start

            # Distance from start point (normalized 0-1)
            start_distance = math.sqrt((current_pos[0] - start_pos[0])**2 + (current_pos[1] - start_pos[1])**2)
            max_distance = math.sqrt((start_pos[0] - end_pos[0])**2 + (start_pos[1] - end_pos[1])**2)
            distance_from_start_norm = min(1.0, start_distance / max_distance)

            # Distance from end point (normalized 0-1, inverted so closer = higher risk)
            end_distance = math.sqrt((current_pos[0] - end_pos[0])**2 + (current_pos[1] - end_pos[1])**2)
            distance_from_end_norm = max(0.0, 1.0 - (end_distance / max_distance))

            # Bell curve factor: peaks when near target, low when near start
            # Combine distance from start (0->1) and proximity to end (1->0->1)
            if flight_phase == "outbound":
                # Risk increases as we approach target
                mission_progress = distance_from_start_norm
            elif flight_phase == "circling":
                # Highest risk during circling (near target)
                mission_progress = 1.0
            else:  # returning
                # Risk decreases as we approach start
                mission_progress = distance_from_start_norm

            # Much steeper bell curve: peaks sharply at mission_progress = 1.0 (near target)
            bell_curve_factor = math.exp(-8.0 * (mission_progress - 1.0)**2)

            # Low base rate, moderate peak rate for balanced crashes
            base_crash_rate = 0.001  # 0.1% - very low early risk
            max_crash_rate = 0.040   # 4.0% - moderate risk near target
            crash_probability = base_crash_rate + (max_crash_rate - base_crash_rate) * bell_curve_factor

            if random.random() < crash_probability:
                # Add final crash waypoint
                path.append(current_pos)
                print(f"    Aircraft CRASHED at step {step} (risk: {crash_probability:.1%}, progress: {mission_progress:.2f})")
                break

        # PHASE 1: Outbound flight toward target
        if flight_phase == "outbound":
            # Check if we've passed the target
            distance_to_target = math.sqrt(
                (current_pos[0] - end_pos[0])**2 + (current_pos[1] - end_pos[1])**2
            )

            # Navigate toward target with deviations
            target_bearing = calculate_bearing_to_target(current_pos, end_pos)
            current_bearing_deg = math.degrees(aircraft.bearing)

            bearing_diff = target_bearing - current_bearing_deg
            while bearing_diff > 180:
                bearing_diff -= 360
            while bearing_diff < -180:
                bearing_diff += 360

            target_correction = bearing_diff * 0.3
            random_deviation = random.gauss(0, 20)

            # Add course changes
            if random.random() < 0.15:
                random_deviation += random.choice([-45, -30, 30, 45])

            total_deviation = target_correction + random_deviation
            total_deviation = max(-75, min(75, total_deviation))

            # Check if passed target (behind us now)
            if distance_to_target < 200 and not passed_target:
                passed_target = True
                # Set up circling phase
                flight_phase = "circling"
                # Circle center is offset from end point
                offset_x = random.uniform(-400, 400)
                offset_y = random.uniform(-400, 400)
                circle_center = (end_pos[0] + offset_x, end_pos[1] + offset_y)
                print(f"    Aircraft passed target, beginning circling phase")

        # PHASE 2: Circling phase
        elif flight_phase == "circling":
            # Navigate in wide arc around circle center
            if circle_center:
                # Calculate bearing to circle center
                center_bearing = calculate_bearing_to_target(current_pos, circle_center)
                current_bearing_deg = math.degrees(aircraft.bearing)

                # Add 90 degrees to create circular motion (left turn)
                circular_bearing = center_bearing + 90

                bearing_diff = circular_bearing - current_bearing_deg
                while bearing_diff > 180:
                    bearing_diff -= 360
                while bearing_diff < -180:
                    bearing_diff += 360

                # Gentle turn toward circular path
                circle_correction = bearing_diff * 0.2
                random_deviation = random.gauss(0, 15)

                total_deviation = circle_correction + random_deviation
                total_deviation = max(-60, min(60, total_deviation))

                # Check if we should start returning (after some circling)
                if step > 100:  # After sufficient circling
                    start_distance = math.sqrt(
                        (current_pos[0] - start_pos[0])**2 + (current_pos[1] - start_pos[1])**2
                    )

                    # Randomly decide to head back when reasonably positioned
                    if random.random() < 0.05 and start_distance > 300:  # 5% chance per step
                        flight_phase = "returning"
                        print(f"    Aircraft beginning return flight")

        # PHASE 3: Return to start
        elif flight_phase == "returning":
            # Navigate back to start position
            start_bearing = calculate_bearing_to_target(current_pos, start_pos)
            current_bearing_deg = math.degrees(aircraft.bearing)

            bearing_diff = start_bearing - current_bearing_deg
            while bearing_diff > 180:
                bearing_diff -= 360
            while bearing_diff < -180:
                bearing_diff += 360

            target_correction = bearing_diff * 0.4  # Stronger homing
            random_deviation = random.gauss(0, 15)

            total_deviation = target_correction + random_deviation
            total_deviation = max(-60, min(60, total_deviation))

            # Check if we've reached start position
            start_distance = math.sqrt(
                (current_pos[0] - start_pos[0])**2 + (current_pos[1] - start_pos[1])**2
            )

            if start_distance < 150:  # Close enough to start
                break

        # Update aircraft position
        aircraft.update_position(total_deviation)
        path.append(aircraft.get_position())

    return np.array(path)

def generate_multiple_flight_paths(target_samples=5000):
    """Generate stochastic flight paths until reaching target training samples."""

    # Get start and end positions from flight config
    flight_config = FlightPathConfig.get_default_flight_path()
    flight_coords = FlightPathConfig.create_pixel_coordinates(flight_config)

    start_pos = flight_coords[0]
    end_pos = flight_coords[-1]

    print(f"Generating stochastic flight paths until {target_samples} training samples...")
    print(f"  Start: ({start_pos[0]:.0f}, {start_pos[1]:.0f})")
    print(f"  End: ({end_pos[0]:.0f}, {end_pos[1]:.0f})")

    flight_paths = []
    crash_sites = []  # Store crash locations for visualization
    total_samples = 0
    flight_num = 0

    # Set seed for reproducible paths
    random.seed(42)
    np.random.seed(42)

    while total_samples < target_samples:
        flight_num += 1

        # Set different random seed for each path to get different deviations
        random.seed(42 + flight_num * 100)
        np.random.seed(42 + flight_num * 100)

        path = generate_stochastic_flight_path(start_pos, end_pos)
        flight_paths.append(path)

        # Calculate training samples from this path (200 max per path)
        target_samples_per_path = 200
        if len(path) >= target_samples_per_path:
            samples_from_path = target_samples_per_path
        else:
            samples_from_path = len(path)

        total_samples += samples_from_path

        # Check if flight ended near start (successful return) or crashed
        final_pos = path[-1]
        start_distance = math.sqrt((final_pos[0] - start_pos[0])**2 + (final_pos[1] - start_pos[1])**2)

        if start_distance < 200:
            print(f"  Path {flight_num}: {len(path)} waypoints → {samples_from_path} samples (COMPLETED) | Total: {total_samples}")
        else:
            print(f"  Path {flight_num}: {len(path)} waypoints → {samples_from_path} samples (CRASHED) | Total: {total_samples}")
            crash_sites.append(final_pos)  # Store crash location

    print(f"✅ Generated {len(flight_paths)} flights with {total_samples} training samples")
    print(f"   Completed flights: {len(flight_paths) - len(crash_sites)}")
    print(f"   Crashed flights: {len(crash_sites)}")

    return flight_paths, crash_sites

def create_flight_paths_visualization(flight_paths, crash_sites=[]):
    """Create visualization showing multiple flight paths, training samples, and crash sites."""
    print("🎨 Creating flight paths visualization...")

    # Load satellite background
    extractor = TerrainExtractor()
    extractor.load_satellite_map("../../data/boneyard/davis_monthan_stitched_map.jpg")

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    # Calculate zoom bounds to show all flight paths
    all_points = np.vstack(flight_paths)
    padding = 800
    x_min = max(0, int(all_points[:, 0].min() - padding))
    x_max = min(7500, int(all_points[:, 0].max() + padding))
    y_min = max(0, int(all_points[:, 1].min() - padding))
    y_max = min(7500, int(all_points[:, 1].max() + padding))

    # Show satellite background
    zoomed_map = extractor.satellite_map[y_min:y_max, x_min:x_max]
    ax.imshow(zoomed_map, extent=[x_min, x_max, y_max, y_min])

    # First pass: collect training samples and draw blue tiles
    all_training_samples = []

    for i, path in enumerate(flight_paths):
        # Sample exactly 200 training points from this path
        target_samples = 200
        if len(path) >= target_samples:
            # Use evenly spaced indices to get exactly 200 points
            indices = np.linspace(0, len(path) - 1, target_samples, dtype=int)
            training_samples = path[indices]
        else:
            # If path is shorter than 200 points, use all points
            training_samples = path

        all_training_samples.extend(training_samples)

        # Draw training sample tiles FIRST (lower z-order)
        # With variable scale and rotation
        base_tile_size = 224

        for idx, sample in enumerate(training_samples):
            x, y = sample

            # Only draw if within zoom area
            if x_min <= x <= x_max and y_min <= y <= y_max:
                from matplotlib.patches import Rectangle
                from matplotlib.transforms import Affine2D

                # Variable scale (altitude simulation) - zoom factor between 0.5 and 2.0
                # Use index-based deterministic variation for consistency
                zoom = 0.5 + (1.5 * ((idx * 17 + i * 37) % 100) / 100.0)  # Deterministic 0.5-2.0 range
                tile_size = base_tile_size * zoom

                # Calculate heading based on aircraft movement at this sample
                # Find corresponding index in original path
                if len(path) > 1:
                    # Find closest point in path
                    distances = np.sqrt((path[:, 0] - x)**2 + (path[:, 1] - y)**2)
                    path_idx = np.argmin(distances)

                    # Calculate heading from movement
                    if path_idx < len(path) - 1:
                        dx = path[path_idx + 1][0] - path[path_idx][0]
                        dy = path[path_idx + 1][1] - path[path_idx][1]
                        heading = np.degrees(np.arctan2(dy, dx))
                    else:
                        heading = 0
                else:
                    heading = 0

                # Create rotated rectangle
                tile_rect = Rectangle(
                    (-tile_size/2, -tile_size/2),  # Centered at origin
                    tile_size, tile_size,
                    linewidth=0.1,
                    edgecolor='blue',
                    facecolor='blue',
                    alpha=0.003,  # Extremely transparent for heavy coverage
                    zorder=6
                )

                # Apply rotation and translation
                t = Affine2D().rotate_deg(heading).translate(x, y) + ax.transData
                tile_rect.set_transform(t)
                ax.add_patch(tile_rect)

    # Second pass: draw ALL flight paths as white lines ON TOP of blue tiles
    for i, path in enumerate(flight_paths):
        # Draw flight path with white line and black border for visibility (higher z-order)
        ax.plot(path[:, 0], path[:, 1], color='black', linewidth=5,
               alpha=1.0, zorder=10)  # Black border, higher z-order

        # Only add label for the first path to avoid cluttered legend
        label = 'Flight Paths' if i == 0 else None
        ax.plot(path[:, 0], path[:, 1], color='white', linewidth=3,
               alpha=1.0, label=label, zorder=11)  # White line on top

    # Add start/end markers from flight configuration
    flight_config = FlightPathConfig.get_default_flight_path()
    flight_coords = FlightPathConfig.create_pixel_coordinates(flight_config)

    config_start = flight_coords[0]  # Actual start point from configuration
    config_end = flight_coords[-1]   # Actual end point from configuration

    ax.scatter(config_start[0], config_start[1], s=150, c='lime', marker='o',
              edgecolors='black', linewidth=2, label='Start Point', zorder=15)
    ax.scatter(config_end[0], config_end[1], s=150, c='red', marker='s',
              edgecolors='black', linewidth=2, label='End Point', zorder=15)

    # Add crash site markers
    if crash_sites:
        crash_x = [pos[0] for pos in crash_sites]
        crash_y = [pos[1] for pos in crash_sites]
        label = 'Crash Sites' if len(crash_sites) > 0 else None
        ax.scatter(crash_x, crash_y, s=200, c='red', marker='x',
                  linewidth=4, label=label, zorder=16)

    # Add info box
    total_samples = len(all_training_samples)
    completed_flights = len(flight_paths) - len(crash_sites)
    info_text = f"""Simulated Reconnaissance Flights
• {len(flight_paths)} total overflights
• {completed_flights} completed, {len(crash_sites)} crashed
• {total_samples} training samples total
• Variable scale and rotation (blue squares)"""

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           zorder=20)

    # Set view limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Flip Y for image coordinates
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    output_path = "images/training_data_coverage_16x9.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', pad_inches=0)
    plt.close()

    print(f"✅ Flight paths visualization saved: {output_path}")
    print(f"   Total training samples: {total_samples}")
    print(f"   Zoom view: X({x_min:.0f}-{x_max:.0f}) Y({y_min:.0f}-{y_max:.0f})")

    return all_training_samples

def main():
    """Generate stochastic flight paths and create visualization."""
    print("🛩️  Stochastic Flight Path Generation")
    print("=" * 45)

    # Generate flight paths until we have 5000 training samples
    flight_paths, crash_sites = generate_multiple_flight_paths(target_samples=5000)

    # Create visualization with crash sites
    training_samples = create_flight_paths_visualization(flight_paths, crash_sites)

    print(f"\n✅ Generated {len(flight_paths)} stochastic flight paths")
    print(f"   Ready for realistic navigation training!")

    return flight_paths, training_samples

if __name__ == "__main__":
    main()