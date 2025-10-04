#!/usr/bin/env python3
"""
Creates a diagram showing how a CNN filter works on a color image.
Shows the convolution operation with a 3x3 filter on RGB channels.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def create_cnn_filter_diagram():
    """
    Creates a visual explanation of CNN convolution on color images.
    Shows input RGB image, filter kernels, and output feature map.
    """
    print("Creating CNN filter diagram...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create sample 5x5 RGB image data
    np.random.seed(42)
    sample_image = np.random.rand(5, 5, 3)
    
    # Define a 3x3 filter for each channel
    filter_r = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Edge detection (vertical)
    filter_g = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_b = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    # Main title
    fig.suptitle('How a CNN Filter Works on a Color Image', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Input Image (RGB)
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(sample_image)
    ax1.set_title('1. Input Image (5×5 pixels)\n3 channels: Red, Green, Blue', 
                 fontsize=12, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # Add grid
    for i in range(6):
        ax1.axhline(i - 0.5, color='white', linewidth=1.5)
        ax1.axvline(i - 0.5, color='white', linewidth=1.5)
    
    # 2. Show RGB channels separately
    ax2 = plt.subplot(2, 3, 2)
    channel_viz = np.zeros((5, 17, 3))
    channel_viz[:, 0:5, 0] = sample_image[:, :, 0]  # R channel
    channel_viz[:, 6:11, 1] = sample_image[:, :, 1]  # G channel
    channel_viz[:, 12:17, 2] = sample_image[:, :, 2]  # B channel
    ax2.imshow(channel_viz)
    ax2.set_title('2. Separate RGB Channels', fontsize=12, fontweight='bold', pad=10)
    ax2.text(2, -0.8, 'R', ha='center', fontsize=11, fontweight='bold', color='red')
    ax2.text(8, -0.8, 'G', ha='center', fontsize=11, fontweight='bold', color='green')
    ax2.text(14, -0.8, 'B', ha='center', fontsize=11, fontweight='bold', color='blue')
    ax2.axis('off')
    
    # 3. Filter kernels
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_xlim(0, 12)
    ax3.set_ylim(0, 4)
    ax3.axis('off')
    ax3.set_title('3. Convolution Filter (3×3)\nOne kernel per channel', 
                 fontsize=12, fontweight='bold', pad=10)
    
    # Draw filter kernels
    colors = ['red', 'green', 'blue']
    for idx, (filt, color) in enumerate(zip([filter_r, filter_g, filter_b], colors)):
        x_offset = idx * 4
        for i in range(3):
            for j in range(3):
                rect = patches.Rectangle((x_offset + j, 2 - i), 1, 1, 
                                        linewidth=2, 
                                        edgecolor=color, 
                                        facecolor='white')
                ax3.add_patch(rect)
                ax3.text(x_offset + j + 0.5, 2 - i + 0.5, 
                        f'{filt[i, j]:.0f}',
                        ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax3.text(x_offset + 1.5, 3.5, f'{color[0].upper()} channel', 
                ha='center', fontsize=10, fontweight='bold', color=color)
    
    # 4. Convolution operation visualization
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_xlim(0, 14)
    ax4.set_ylim(0, 8)
    ax4.axis('off')
    ax4.set_title('4. Convolution Operation\n(Element-wise multiply & sum)', 
                 fontsize=12, fontweight='bold', pad=10)
    
    # Draw a sample 3x3 region
    for i in range(3):
        for j in range(3):
            # Input patch
            rect = patches.Rectangle((j, 5 - i), 1, 1, 
                                    linewidth=1.5, 
                                    edgecolor='blue', 
                                    facecolor='lightblue', alpha=0.5)
            ax4.add_patch(rect)
            ax4.text(j + 0.5, 5 - i + 0.5, f'{sample_image[i, j, 0]:.2f}',
                    ha='center', va='center', fontsize=8)
    
    ax4.text(1.5, 6.5, 'Image patch (3×3)', ha='center', fontsize=10, fontweight='bold')
    
    # Multiply symbol
    ax4.text(4, 4, '×', ha='center', va='center', fontsize=24, fontweight='bold')
    
    # Filter
    for i in range(3):
        for j in range(3):
            rect = patches.Rectangle((5 + j, 5 - i), 1, 1, 
                                    linewidth=1.5, 
                                    edgecolor='red', 
                                    facecolor='lightcoral', alpha=0.5)
            ax4.add_patch(rect)
            ax4.text(5 + j + 0.5, 5 - i + 0.5, f'{filter_r[i, j]:.0f}',
                    ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax4.text(6.5, 6.5, 'Filter (3×3)', ha='center', fontsize=10, fontweight='bold')
    
    # Arrow and result
    ax4.text(9, 4, '→', ha='center', va='center', fontsize=24, fontweight='bold')
    
    result_val = np.sum(sample_image[0:3, 0:3, 0] * filter_r)
    rect = patches.Rectangle((10.5, 3), 2, 2, 
                            linewidth=2, 
                            edgecolor='green', 
                            facecolor='lightgreen', alpha=0.7)
    ax4.add_patch(rect)
    ax4.text(11.5, 4, f'{result_val:.2f}',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax4.text(11.5, 1.5, 'One output\nvalue', ha='center', fontsize=9, style='italic')
    
    # 5. Sliding window
    ax5 = plt.subplot(2, 3, 5)
    # Create visualization of sliding window
    output_map = np.zeros((3, 3))
    ax5.imshow(output_map, cmap='coolwarm', vmin=-2, vmax=2)
    ax5.set_title('5. Slide Filter Across Image\nCreate output feature map (3×3)', 
                 fontsize=12, fontweight='bold', pad=10)
    
    # Draw grid
    for i in range(4):
        ax5.axhline(i - 0.5, color='black', linewidth=2)
        ax5.axvline(i - 0.5, color='black', linewidth=2)
    
    # Add arrows showing sliding
    ax5.annotate('', xy=(1.5, 0.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax5.annotate('', xy=(0.5, 1.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    ax5.text(1, 2.8, 'Slide across\nall positions', 
            ha='center', fontsize=9, style='italic')
    
    # 6. Result
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 6)
    ax6.axis('off')
    ax6.set_title('6. Output Feature Map\nDetects patterns (edges, textures)', 
                 fontsize=12, fontweight='bold', pad=10)
    
    # Create sample feature map
    feature_map = np.random.randn(3, 3)
    
    # Draw feature map as colored grid
    for i in range(3):
        for j in range(3):
            val = feature_map[i, j]
            color = plt.cm.RdBu_r((val + 2) / 4)  # Normalize to 0-1
            rect = patches.Rectangle((2 + j * 1.5, 3.5 - i * 1.5), 1.3, 1.3,
                                    linewidth=2,
                                    edgecolor='black',
                                    facecolor=color)
            ax6.add_patch(rect)
    
    ax6.text(4, 0.5, 'This feature map might detect\nvertical edges in the image',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Default output path based on current working directory
    import os
    if os.path.basename(os.getcwd()) == 'code':
        output_path = Path('../images/cnn_filter_diagram_color.png')
    else:
        output_path = Path('images/cnn_filter_diagram_color.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved CNN filter diagram to {output_path}")
    return output_path


if __name__ == "__main__":
    create_cnn_filter_diagram()
    print("\n✨ CNN filter diagram created!")

