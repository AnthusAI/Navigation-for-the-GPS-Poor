#!/usr/bin/env python3
"""
Creates a clear diagram showing the anatomy of a neural network.
Shows input layer, hidden layers with neurons, and output layer.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_neural_network_diagram():
    """
    Creates a simple, clear diagram of a neural network architecture.
    Shows layers, neurons, connections, and labels.
    """
    print("Creating neural network diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define layer positions
    layers = [
        {'x': 2, 'neurons': 6, 'label': 'Input Layer\n(Image Pixels)', 'color': '#3498db'},
        {'x': 5, 'neurons': 5, 'label': 'Hidden Layer 1\n(Features)', 'color': '#e74c3c'},
        {'x': 8, 'neurons': 4, 'label': 'Hidden Layer 2\n(Combinations)', 'color': '#e74c3c'},
        {'x': 11, 'neurons': 3, 'label': 'Hidden Layer 3\n(High-level)', 'color': '#e74c3c'},
        {'x': 13.5, 'neurons': 2, 'label': 'Output Layer\n(x, y position)', 'color': '#2ecc71'},
    ]
    
    # Draw connections between layers (before neurons so they're behind)
    for i in range(len(layers) - 1):
        layer1 = layers[i]
        layer2 = layers[i + 1]
        
        # Calculate neuron positions
        y1_positions = [4 + (j - (layer1['neurons']-1)/2) * 0.8 for j in range(layer1['neurons'])]
        y2_positions = [4 + (j - (layer2['neurons']-1)/2) * 0.8 for j in range(layer2['neurons'])]
        
        # Draw connections
        for y1 in y1_positions:
            for y2 in y2_positions:
                ax.plot([layer1['x'], layer2['x']], [y1, y2], 
                       color='gray', alpha=0.2, linewidth=0.5, zorder=1)
    
    # Draw neurons
    neuron_coords = []
    for layer_idx, layer in enumerate(layers):
        x = layer['x']
        neurons = layer['neurons']
        y_positions = [4 + (i - (neurons-1)/2) * 0.8 for i in range(neurons)]
        
        coords = []
        for y in y_positions:
            circle = patches.Circle((x, y), 0.25, 
                                   facecolor=layer['color'], 
                                   edgecolor='white', 
                                   linewidth=2,
                                   zorder=2)
            ax.add_patch(circle)
            coords.append((x, y))
        
        neuron_coords.append(coords)
        
        # Add layer label
        ax.text(x, 1.2, layer['label'], 
               ha='center', va='top', 
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='white', 
                        edgecolor=layer['color'], 
                        linewidth=2))
    
    # Add title
    ax.text(7, 7.5, 'Anatomy of a Neural Network', 
           ha='center', va='top', 
           fontsize=18, fontweight='bold')
    
    # Add explanation annotations
    ax.annotate('Each circle is a "neuron"\nthat processes information',
               xy=(5, 4.8), xytext=(5, 6.5),
               ha='center',
               fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.annotate('Connections carry\nweighted signals',
               xy=(6.5, 4), xytext=(6.5, 2.5),
               ha='center',
               fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add flow arrows at top
    for i in range(len(layers) - 1):
        x1 = layers[i]['x']
        x2 = layers[i + 1]['x']
        ax.annotate('', xy=(x2 - 0.3, 7), xytext=(x1 + 0.3, 7),
                   arrowprops=dict(arrowstyle='->', lw=3, color='darkblue'))
    
    ax.text(7, 7.2, 'Information flows forward →', 
           ha='center', va='bottom', 
           fontsize=11, style='italic', color='darkblue')
    
    plt.tight_layout()
    
    # Default output path based on current working directory
    import os
    if os.path.basename(os.getcwd()) == 'code':
        output_path = Path('../images/neural_network_diagram.png')
    else:
        output_path = Path('images/neural_network_diagram.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved neural network diagram to {output_path}")
    return output_path


if __name__ == "__main__":
    create_neural_network_diagram()
    print("\n✨ Neural network diagram created!")

