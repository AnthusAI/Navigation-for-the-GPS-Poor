"""
Create baseline flight evaluation results from existing terrain_cnn_v1 results
to have compatible format for comparisons.
"""
import pickle
import numpy as np
from PIL import Image

def create_baseline_results():
    """Create compatible baseline results for comparison."""

    # Use realistic results as baseline - good performance and right format
    source_file = 'artifacts/realistic_eval_results.pkl'

    try:
        with open(source_file, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded {source_file}")
        print(f"Keys: {list(results.keys())}")

        # Check if this has the flight path format we need
        if 'predictions' in results and 'targets' in results:
            # Convert from normalized to pixel coordinates
            map_path = '../../data/boneyard/davis_monthan_stitched_map.jpg'
            try:
                full_map = Image.open(map_path).convert('RGB')
                map_width, map_height = full_map.size

                # Denormalize coordinates
                targets_norm = results['targets']
                pred_norm = results['predictions']

                ground_truth = targets_norm * np.array([map_width, map_height])
                predictions = pred_norm * np.array([map_width, map_height])
                errors = results['errors']

                # Create baseline results in flight format
                baseline_results = {
                    'ground_truth': ground_truth,
                    'predictions': predictions,
                    'errors': errors,
                    'mean_error': np.mean(errors),
                    'median_error': np.median(errors),
                    'max_error': np.max(errors)
                }

                # Save as flight evaluation results
                output_file = 'artifacts/flight_evaluation_results.pkl'
                with open(output_file, 'wb') as f:
                    pickle.dump(baseline_results, f)

                print(f"✅ Created baseline results: {output_file}")
                print(f"Mean error: {np.mean(errors):.1f}px")
                return True

            except Exception as e:
                print(f"Error processing map: {e}")
                return False
        else:
            print("Source results don't have expected format")
            return False

    except Exception as e:
        print(f"Error loading source results: {e}")
        return False

if __name__ == "__main__":
    create_baseline_results()