#!/usr/bin/env python3
"""
Create visualization of error analysis and terrain recognition patterns.
"""
import sys
sys.path.append('../../..')

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def load_results():
    """Load the flight evaluation results."""
    results_path = '../artifacts/flight_evaluation_results.pkl'
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results['ground_truth'], results['predictions'], results['errors']

def create_error_analysis():
    """Create detailed error analysis visualizations."""
    print("Creating error analysis...")
    gt_coords, pred_coords, errors = load_results()

    # Create analysis figure with 16:9 aspect ratio
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9))

    # 1. Error over flight path
    frames = np.arange(len(errors))
    ax1.plot(frames, errors, 'b-', linewidth=2, alpha=0.8)
    ax1.fill_between(frames, errors, alpha=0.3, color='blue')

    mean_error = np.mean(errors)
    median_error = np.median(errors)
    ax1.axhline(y=mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.1f}px')
    ax1.axhline(y=median_error, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_error:.1f}px')

    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_ylabel('Position Error (pixels)', fontsize=12)
    ax1.set_title('Position Error Along Flight Path', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Error histogram
    ax2.hist(errors, bins=25, alpha=0.7, color='purple', edgecolor='black')
    ax2.axvline(x=mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.1f}px')
    ax2.axvline(x=median_error, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_error:.1f}px')
    ax2.set_xlabel('Position Error (pixels)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 3. X vs Y error components
    x_errors = pred_coords[:, 0] - gt_coords[:, 0]
    y_errors = pred_coords[:, 1] - gt_coords[:, 1]

    ax3.scatter(x_errors, y_errors, alpha=0.6, s=30, c=frames, cmap='viridis')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('X Error (pixels)', fontsize=12)
    ax3.set_ylabel('Y Error (pixels)', fontsize=12)
    ax3.set_title('X vs Y Error Components', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # 4. Confidence over time (based on local error stability)
    window_size = 5
    confidence_scores = np.zeros(len(errors))

    for i in range(len(errors)):
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(errors), i + window_size//2 + 1)
        local_errors = errors[start_idx:end_idx]

        # Confidence = combination of local stability and absolute error
        local_stability = 1.0 / (1.0 + np.std(local_errors))
        error_confidence = 1.0 / (1.0 + errors[i] / 100.0)
        confidence_scores[i] = (local_stability + error_confidence) / 2.0

    ax4.plot(frames, confidence_scores, 'purple', linewidth=2, alpha=0.8, label='Confidence Score')
    ax4.fill_between(frames, confidence_scores, alpha=0.3, color='purple')

    # Add error overlay (inverted scale)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(frames, errors, 'red', linewidth=1, alpha=0.5, label='Position Error')
    ax4_twin.set_ylabel('Position Error (pixels)', fontsize=12, color='red')
    ax4_twin.tick_params(axis='y', labelcolor='red')

    ax4.set_xlabel('Frame Number', fontsize=12)
    ax4.set_ylabel('Confidence Score', fontsize=12, color='purple')
    ax4.set_title('Model Confidence Over Flight Path', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='purple')
    ax4.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

    plt.tight_layout()

    # Save the analysis plot
    output_path = '../images/model_training_curves.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Error analysis saved to {output_path}")

    # Print detailed analysis
    print(f"\nDetailed Error Analysis:")
    print(f"={'='*50}")
    print(f"Total frames analyzed: {len(errors)}")
    print(f"Mean error: {mean_error:.1f} pixels")
    print(f"Median error: {median_error:.1f} pixels")
    print(f"Standard deviation: {np.std(errors):.1f} pixels")
    print(f"Min error: {errors.min():.1f} pixels")
    print(f"Max error: {errors.max():.1f} pixels")
    print(f"95th percentile: {np.percentile(errors, 95):.1f} pixels")
    print(f"5th percentile: {np.percentile(errors, 5):.1f} pixels")

    # Direction bias analysis
    print(f"\nDirection Bias Analysis:")
    print(f"{'='*50}")
    print(f"Mean X error: {np.mean(x_errors):.1f} pixels")
    print(f"Mean Y error: {np.mean(y_errors):.1f} pixels")
    print(f"X error std: {np.std(x_errors):.1f} pixels")
    print(f"Y error std: {np.std(y_errors):.1f} pixels")

    # Performance assessment
    good_frames = np.sum(errors < 100)
    excellent_frames = np.sum(errors < 50)

    print(f"\nPerformance Assessment:")
    print(f"{'='*50}")
    print(f"Frames with <100px error: {good_frames}/{len(errors)} ({good_frames/len(errors)*100:.1f}%)")
    print(f"Frames with <50px error: {excellent_frames}/{len(errors)} ({excellent_frames/len(errors)*100:.1f}%)")
    print(f"Relative accuracy: {(mean_error/7500)*100:.2f}% of map width")

    if median_error < 100:
        print(f"✅ EXCELLENT: Sub-100px median error suggests genuine terrain recognition")
    if good_frames/len(errors) > 0.8:
        print(f"✅ CONSISTENT: >80% of frames have <100px error")
    if np.std(errors) < mean_error:
        print(f"✅ STABLE: Low error variance indicates consistent performance")

    return mean_error, median_error, errors

if __name__ == "__main__":
    create_error_analysis()



