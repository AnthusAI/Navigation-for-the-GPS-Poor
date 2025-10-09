"""
Realistic Environmental Effect Presets for Training Data Generation

This module defines realistic combinations of environmental effects that aircraft
might encounter during GPS-poor navigation missions. These presets ensure
training data covers a diverse range of operational conditions.
"""

import random
from typing import Dict, List, Optional
import numpy as np


class EnvironmentalPresets:
    """Manages realistic environmental effect presets for training data generation."""

    # Define realistic environmental conditions
    PRESETS = {
        'clear_day': {
            'name': 'Clear Day Conditions',
            'probability': 0.25,
            'effects': {
                'brightness': (1.1, 1.3),
                'contrast': (1.0, 1.2),
                'fog_intensity': (0.0, 0.05),
                'noise_std': (1.0, 3.0),
                'motion_blur': (0.0, 0.5),
            }
        },

        'overcast': {
            'name': 'Overcast Conditions',
            'probability': 0.20,
            'effects': {
                'brightness': (0.7, 0.9),
                'contrast': (0.8, 1.0),
                'fog_intensity': (0.1, 0.2),
                'noise_std': (2.0, 4.0),
                'motion_blur': (0.2, 0.8),
            }
        },

        'hazy': {
            'name': 'Hazy/Dusty Conditions',
            'probability': 0.20,
            'effects': {
                'brightness': (0.8, 1.1),
                'contrast': (0.7, 0.9),
                'fog_intensity': (0.15, 0.35),
                'noise_std': (2.5, 5.0),
                'motion_blur': (0.3, 1.0),
            }
        },

        'low_visibility': {
            'name': 'Low Visibility (Fog/Smoke)',
            'probability': 0.15,
            'effects': {
                'brightness': (0.6, 0.8),
                'contrast': (0.6, 0.8),
                'fog_intensity': (0.3, 0.5),
                'noise_std': (4.0, 8.0),
                'motion_blur': (0.5, 1.5),
            }
        },

        'high_altitude': {
            'name': 'High Altitude (Bright, Clear)',
            'probability': 0.10,
            'effects': {
                'brightness': (1.2, 1.5),
                'contrast': (1.1, 1.3),
                'fog_intensity': (0.0, 0.02),
                'noise_std': (1.5, 3.5),
                'motion_blur': (0.0, 0.3),
            }
        },

        'dawn_dusk': {
            'name': 'Dawn/Dusk Lighting',
            'probability': 0.10,
            'effects': {
                'brightness': (0.5, 0.8),
                'contrast': (0.9, 1.2),
                'fog_intensity': (0.05, 0.15),
                'noise_std': (3.0, 6.0),
                'motion_blur': (0.2, 0.8),
            }
        }
    }

    @classmethod
    def get_random_preset(cls) -> Dict:
        """
        Select a random environmental preset based on realistic probabilities.

        Returns:
            Dictionary containing environmental effects for the selected preset
        """
        # Create weighted selection based on probabilities
        preset_names = list(cls.PRESETS.keys())
        probabilities = [cls.PRESETS[name]['probability'] for name in preset_names]

        # Normalize probabilities (should sum to 1.0)
        prob_sum = sum(probabilities)
        probabilities = [p / prob_sum for p in probabilities]

        # Select preset
        selected_preset = np.random.choice(preset_names, p=probabilities)
        return cls.PRESETS[selected_preset]

    @classmethod
    def generate_effects_from_preset(cls, preset: Dict) -> Dict:
        """
        Generate specific environmental effects from a preset.

        Args:
            preset: Environmental preset dictionary

        Returns:
            Dictionary of specific environmental effect values
        """
        effects = {}

        for effect_name, value_range in preset['effects'].items():
            if isinstance(value_range, tuple) and len(value_range) == 2:
                # Generate random value within range
                min_val, max_val = value_range
                effects[effect_name] = random.uniform(min_val, max_val)
            else:
                # Use fixed value
                effects[effect_name] = value_range

        return effects

    @classmethod
    def get_realistic_environmental_effects(cls) -> Dict:
        """
        Generate realistic environmental effects for training data.

        Returns:
            Dictionary of environmental effects ready for TerrainWindow
        """
        preset = cls.get_random_preset()
        effects = cls.generate_effects_from_preset(preset)

        # Add preset name for debugging/logging
        effects['_preset_name'] = preset['name']

        return effects

    @classmethod
    def get_altitude_simulation_params(cls) -> Dict:
        """
        Generate realistic altitude simulation parameters.

        Returns:
            Dictionary containing zoom factor and related effects
        """
        # Realistic altitude ranges and corresponding zoom factors
        altitude_scenarios = [
            {'name': 'Low Altitude (500-800m)', 'zoom': (1.2, 1.6), 'probability': 0.3},
            {'name': 'Normal Altitude (800-1500m)', 'zoom': (0.9, 1.2), 'probability': 0.4},
            {'name': 'High Altitude (1500-2500m)', 'zoom': (0.6, 0.9), 'probability': 0.3},
        ]

        # Select scenario based on probability
        probabilities = [s['probability'] for s in altitude_scenarios]
        selected_scenario = np.random.choice(altitude_scenarios, p=probabilities)

        zoom_min, zoom_max = selected_scenario['zoom']
        zoom_factor = random.uniform(zoom_min, zoom_max)

        return {
            'zoom': zoom_factor,
            'altitude_scenario': selected_scenario['name']
        }

    @classmethod
    def should_apply_effects(cls, effect_probability: float = 0.8) -> bool:
        """
        Determine if environmental effects should be applied to this sample.

        Args:
            effect_probability: Probability of applying effects (0.0 to 1.0)

        Returns:
            Boolean indicating whether to apply effects
        """
        return random.random() < effect_probability


def demonstrate_presets():
    """Demonstrate environmental preset usage."""
    print("🌤️  Environmental Presets Demonstration")
    print("=" * 50)

    # Show all available presets
    print("Available Environmental Presets:")
    for name, preset in EnvironmentalPresets.PRESETS.items():
        print(f"  • {preset['name']} (probability: {preset['probability']:.1%})")

    print(f"\n📊 Generating 10 Random Environmental Conditions:")
    for i in range(10):
        effects = EnvironmentalPresets.get_realistic_environmental_effects()
        altitude_params = EnvironmentalPresets.get_altitude_simulation_params()

        preset_name = effects.pop('_preset_name')
        print(f"\n  Sample {i+1}: {preset_name}")
        print(f"    Zoom: {altitude_params['zoom']:.2f}x ({altitude_params['altitude_scenario']})")
        print(f"    Brightness: {effects['brightness']:.2f}")
        print(f"    Fog: {effects['fog_intensity']:.2f}")
        print(f"    Noise: {effects['noise_std']:.1f}")


if __name__ == "__main__":
    demonstrate_presets()