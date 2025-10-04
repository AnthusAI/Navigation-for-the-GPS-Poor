#!/usr/bin/env python3
"""
Creates a diagram showing the deep learning training loop.
Shows the iterative process: forward pass, loss calculation, backward pass, weight update.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_training_loop_diagram():
    """
    Creates a clear, circular diagram of the training loop.
    Shows all key steps in the iterative training process.
    """
    print("Creating training loop diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'The Deep Learning Training Loop', 
           ha='center', va='top', 
           fontsize=20, fontweight='bold')
    
    ax.text(7, 9, 'An iterative process that gradually improves the model', 
           ha='center', va='top', 
           fontsize=12, style='italic', color='gray')
    
    # Define the circular layout of steps
    import numpy as np
    center_x, center_y = 7, 5
    radius = 3
    
    steps = [
        {
            'angle': 90,  # Top
            'title': '1. Forward Pass',
            'desc': 'Feed input through\nthe network',
            'color': '#3498db',
            'icon': '→'
        },
        {
            'angle': 30,  # Top-right
            'title': '2. Make Prediction',
            'desc': 'Network outputs\nits guess',
            'color': '#9b59b6',
            'icon': '?'
        },
        {
            'angle': -30,  # Bottom-right
            'title': '3. Calculate Loss',
            'desc': 'How wrong is\nthe prediction?',
            'color': '#e74c3c',
            'icon': '✗'
        },
        {
            'angle': -90,  # Bottom
            'title': '4. Backward Pass',
            'desc': 'Compute gradients\n(backpropagation)',
            'color': '#e67e22',
            'icon': '←'
        },
        {
            'angle': -150,  # Bottom-left
            'title': '5. Update Weights',
            'desc': 'Adjust network\nparameters',
            'color': '#f39c12',
            'icon': '⚙'
        },
        {
            'angle': 150,  # Top-left
            'title': '6. Repeat',
            'desc': 'Next batch\nof data',
            'color': '#2ecc71',
            'icon': '↻'
        }
    ]
    
    # Draw circular arrows connecting steps
    for i, step in enumerate(steps):
        angle1 = np.radians(step['angle'])
        angle2 = np.radians(steps[(i + 1) % len(steps)]['angle'])
        
        # Calculate arrow start and end points (on the circle)
        x1 = center_x + radius * np.cos(angle1)
        y1 = center_y + radius * np.sin(angle1)
        x2 = center_x + radius * np.cos(angle2)
        y2 = center_y + radius * np.sin(angle2)
        
        # Draw curved arrow
        ax.annotate('', 
                   xy=(x2, y2), 
                   xytext=(x1, y1),
                   arrowprops=dict(
                       arrowstyle='->,head_width=0.6,head_length=0.8',
                       connectionstyle='arc3,rad=0.3',
                       lw=3,
                       color='darkgray',
                       alpha=0.6
                   ))
    
    # Draw step boxes
    for i, step in enumerate(steps):
        angle = np.radians(step['angle'])
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # Box
        box = patches.FancyBboxPatch(
            (x - 0.9, y - 0.5), 1.8, 1.0,
            boxstyle="round,pad=0.1",
            linewidth=3,
            edgecolor=step['color'],
            facecolor='white',
            zorder=10
        )
        ax.add_patch(box)
        
        # Icon/symbol
        ax.text(x, y + 0.15, step['icon'], 
               ha='center', va='center',
               fontsize=20, fontweight='bold',
               color=step['color'])
        
        # Title below icon
        ax.text(x, y - 0.25, step['title'],
               ha='center', va='center',
               fontsize=9, fontweight='bold',
               color=step['color'])
        
        # Description outside the circle
        desc_angle = np.radians(step['angle'])
        desc_radius = radius + 1.5
        desc_x = center_x + desc_radius * np.cos(desc_angle)
        desc_y = center_y + desc_radius * np.sin(desc_angle)
        
        ax.text(desc_x, desc_y, step['desc'],
               ha='center', va='center',
               fontsize=9, style='italic',
               color='gray',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='lightyellow', 
                        alpha=0.7,
                        edgecolor='none'))
    
    # Center text
    center_circle = patches.Circle((center_x, center_y), 0.8,
                                  facecolor='lightblue',
                                  edgecolor='darkblue',
                                  linewidth=2,
                                  zorder=5)
    ax.add_patch(center_circle)
    
    ax.text(center_x, center_y + 0.15, 'Training',
           ha='center', va='center',
           fontsize=12, fontweight='bold',
           color='darkblue')
    ax.text(center_x, center_y - 0.2, 'Loop',
           ha='center', va='center',
           fontsize=12, fontweight='bold',
           color='darkblue')
    
    # Add key concepts at bottom
    ax.text(1, 1.5, '💡 Key Concepts:', fontsize=12, fontweight='bold')
    
    concepts = [
        '• Epoch: One complete pass through all training data',
        '• Batch: A subset of data processed together',
        '• Learning Rate: How big the weight updates are',
        '• Gradient: Direction to adjust weights to reduce loss'
    ]
    
    for i, concept in enumerate(concepts):
        ax.text(1.2, 1.0 - i * 0.3, concept,
               fontsize=9,
               va='top')
    
    # Add iteration counter example
    ax.text(13, 1.5, '📊 Progress:', fontsize=12, fontweight='bold', ha='right')
    ax.text(13, 1.0, 'Each loop = 1 iteration', fontsize=9, ha='right')
    ax.text(13, 0.7, 'Thousands of iterations', fontsize=9, ha='right')
    ax.text(13, 0.4, '= One trained model!', fontsize=9, ha='right', style='italic')
    
    plt.tight_layout()
    
    # Default output path based on current working directory
    import os
    if os.path.basename(os.getcwd()) == 'code':
        output_path = Path('../images/training_loop_diagram.png')
    else:
        output_path = Path('images/training_loop_diagram.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved training loop diagram to {output_path}")
    return output_path


if __name__ == "__main__":
    create_training_loop_diagram()
    print("\n✨ Training loop diagram created!")

