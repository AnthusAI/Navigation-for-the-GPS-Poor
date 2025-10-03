#!/usr/bin/env python3
"""
Creates visualizations showing challenging conditions where classical
computer vision fails but deep learning can still work.
"""

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_challenging_conditions_comparison():
    """
    Creates a side-by-side comparison showing:
    - Clear conditions (where classical CV works)
    - Challenging conditions (fog, low light, texture-less)
    """
    print("Creating challenging conditions comparison...")
    
    # Load source imagery
    source_img = Image.open('data/boneyard/davis_monthan_aerial.jpg')
    
    # Crop a good section showing aircraft
    crop_box = (300, 200, 900, 600)  # 600x400 section
    clear_img = source_img.crop(crop_box)
    
    # Create three challenging variations
    
    # 1. Dense fog
    fog_img = clear_img.copy()
    fog_img = fog_img.filter(ImageFilter.GaussianBlur(radius=5))
    # Add white overlay for fog effect
    fog_overlay = Image.new('RGB', fog_img.size, (240, 240, 245))
    fog_img = Image.blend(fog_img, fog_overlay, alpha=0.5)
    
    # 2. Low light / darkness
    dark_img = clear_img.copy()
    enhancer = ImageEnhance.Brightness(dark_img)
    dark_img = enhancer.enhance(0.3)  # Very dark
    # Add noise
    dark_array = np.array(dark_img)
    noise = np.random.normal(0, 15, dark_array.shape)
    dark_array = np.clip(dark_array + noise, 0, 255).astype(np.uint8)
    dark_img = Image.fromarray(dark_array)
    
    # 3. Sandstorm / texture-less
    sand_img = clear_img.copy()
    sand_img = sand_img.filter(ImageFilter.GaussianBlur(radius=8))
    # Add sandy color overlay
    sand_overlay = Image.new('RGB', sand_img.size, (210, 180, 140))
    sand_img = Image.blend(sand_img, sand_overlay, alpha=0.6)
    
    # Combine into a 2x2 grid
    grid_img = Image.new('RGB', (1200, 800))
    
    # Top left: Clear
    clear_resized = clear_img.resize((600, 400))
    grid_img.paste(clear_resized, (0, 0))
    
    # Top right: Fog
    fog_resized = fog_img.resize((600, 400))
    grid_img.paste(fog_resized, (600, 0))
    
    # Bottom left: Low light
    dark_resized = dark_img.resize((600, 400))
    grid_img.paste(dark_resized, (0, 400))
    
    # Bottom right: Sandstorm
    sand_resized = sand_img.resize((600, 400))
    grid_img.paste(sand_resized, (600, 400))
    
    output_path = Path('chapters/4/images/challenging_conditions.png')
    grid_img.save(output_path, quality=95)
    
    print(f"✅ Saved comparison to {output_path}")
    return output_path


def create_foggy_flyover():
    """
    Creates an animated fly-over showing increasingly challenging conditions.
    Starts clear, progressively adds fog to show how visibility degrades.
    """
    print("\nCreating foggy fly-over animation...")
    
    source_img = Image.open('data/boneyard/davis_monthan_aerial.jpg')
    src_width, src_height = source_img.size
    
    frames = []
    frame_width, frame_height = 1200, 675
    step_size = 20
    num_frames = (src_width - frame_width) // step_size
    
    print(f"Creating {num_frames} frames with progressive fog...")
    
    for i in tqdm(range(num_frames), desc='Generating foggy frames'):
        x = i * step_size
        y = (src_height - frame_height) // 2
        frame = source_img.crop((x, y, x + frame_width, y + frame_height))
        
        # Progressive fog: increases as we fly
        fog_intensity = min(0.7, i / num_frames * 1.2)  # Caps at 70% fog
        
        if fog_intensity > 0.05:
            # Blur the image
            blur_radius = int(fog_intensity * 8)
            frame = frame.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Add white/gray fog overlay
            fog_color = (235, 235, 240)
            fog_overlay = Image.new('RGB', frame.size, fog_color)
            frame = Image.blend(frame, fog_overlay, alpha=fog_intensity * 0.7)
        
        # Convert to palette for smaller file size
        frame = frame.convert('P', palette=Image.ADAPTIVE, colors=128)
        frames.append(frame)
    
    output_path = Path('chapters/4/images/foggy_flyover.gif')
    print(f"Saving {len(frames)} frames...")
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=266,
        loop=0
    )
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Created foggy fly-over: {size_mb:.1f} MB")
    print(f"   Shows progressive visibility degradation")
    
    return output_path


if __name__ == "__main__":
    # Create both visualizations
    create_challenging_conditions_comparison()
    create_foggy_flyover()
    
    print("\n✨ All challenging conditions visualizations created!")


