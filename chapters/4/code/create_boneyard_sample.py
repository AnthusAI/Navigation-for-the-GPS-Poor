#!/usr/bin/env python3
"""
Creates a sample image from the Davis-Monthan AFB Boneyard aerial imagery.
Shows a nice crop with visible aircraft for the article.
"""

from PIL import Image
from pathlib import Path

def create_boneyard_sample():
    """
    Extracts a representative sample from the boneyard aerial image.
    Shows multiple aircraft clearly visible.
    """
    print("Creating boneyard sample image...")
    
    # Default paths based on current working directory
    import os
    if os.path.basename(os.getcwd()) == 'code':
        image_path = '../../../data/boneyard/davis_monthan_aerial.jpg'
        output_dir = '../images'
    else:
        image_path = '../../data/boneyard/davis_monthan_aerial.jpg'
        output_dir = 'images'
    
    # Load source imagery
    source_img = Image.open(image_path)
    
    # Crop a good section showing multiple aircraft
    # This section has nice variety and clear aircraft
    crop_box = (200, 150, 1000, 650)  # 800x500 section
    sample_img = source_img.crop(crop_box)
    
    output_path = Path(output_dir) / 'boneyard_sample.png'
    sample_img.save(output_path, quality=95)
    
    file_size_kb = output_path.stat().st_size / 1024
    print(f"✅ Saved boneyard sample to {output_path}")
    print(f"   Size: 800x500 pixels ({file_size_kb:.1f} KB)")
    print(f"   Shows: Davis-Monthan AFB aircraft storage area")
    
    return output_path


if __name__ == "__main__":
    create_boneyard_sample()
    print("\n✨ Boneyard sample image created!")

