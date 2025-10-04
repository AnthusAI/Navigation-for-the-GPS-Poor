#!/usr/bin/env python3
"""
Creates visualizations showing challenging conditions where classical
computer vision fails but deep learning can still work.
"""

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_challenging_conditions_comparison(image_path=None, output_dir=None):
    """
    Creates a side-by-side comparison showing:
    - Clear conditions (where classical CV works)
    - Challenging conditions (fog, low light, texture-less)
    """
    print("Creating challenging conditions comparison...")
    
    # Default paths based on current working directory
    import os
    if image_path is None:
        if os.path.basename(os.getcwd()) == 'code':
            image_path = '../../../data/boneyard/davis_monthan_aerial.jpg'
        else:
            image_path = '../../data/boneyard/davis_monthan_aerial.jpg'
    
    if output_dir is None:
        output_dir = '../images' if os.path.basename(os.getcwd()) == 'code' else 'images'
    
    # Load source imagery
    source_img = Image.open(image_path)
    
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
    
    output_path = Path(output_dir) / 'challenging_conditions.png'
    grid_img.save(output_path, quality=95)
    
    print(f"✅ Saved comparison to {output_path}")
    return output_path


def create_foggy_flyover(image_path=None, output_dir=None):
    """
    Creates an animated fly-over showing increasingly challenging conditions.
    Starts clear, progressively adds fog to show how visibility degrades.
    """
    print("\nCreating foggy fly-over animation...")
    
    # Default paths based on current working directory
    import os
    if image_path is None:
        if os.path.basename(os.getcwd()) == 'code':
            image_path = '../../../data/boneyard/davis_monthan_aerial.jpg'
        else:
            image_path = '../../data/boneyard/davis_monthan_aerial.jpg'
    
    if output_dir is None:
        output_dir = '../images' if os.path.basename(os.getcwd()) == 'code' else 'images'
    
    source_img = Image.open(image_path)
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
    
    output_path = Path(output_dir) / 'foggy_flyover.gif'
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


def create_challenging_conditions_gif(image_path=None, output_dir=None):
    """
    Creates an animated GIF cycling through challenging conditions.
    Shows clear -> fog -> low light -> sandstorm -> repeat
    """
    print("\nCreating challenging conditions animated GIF...")
    
    # Default paths based on current working directory
    import os
    if image_path is None:
        if os.path.basename(os.getcwd()) == 'code':
            image_path = '../../../data/boneyard/davis_monthan_aerial.jpg'
        else:
            image_path = '../../data/boneyard/davis_monthan_aerial.jpg'
    
    if output_dir is None:
        output_dir = '../images' if os.path.basename(os.getcwd()) == 'code' else 'images'
    
    source_img = Image.open(image_path)
    
    # Crop a good section showing aircraft
    crop_box = (300, 200, 1100, 600)  # 800x400 section
    clear_img = source_img.crop(crop_box)
    
    frames = []
    
    # Frame 1: Clear (hold longer)
    for _ in range(3):
        frames.append(clear_img.convert('P', palette=Image.ADAPTIVE, colors=128))
    
    # Frame 2: Dense fog
    fog_img = clear_img.copy()
    fog_img = fog_img.filter(ImageFilter.GaussianBlur(radius=5))
    fog_overlay = Image.new('RGB', fog_img.size, (240, 240, 245))
    fog_img = Image.blend(fog_img, fog_overlay, alpha=0.5)
    for _ in range(3):
        frames.append(fog_img.convert('P', palette=Image.ADAPTIVE, colors=128))
    
    # Frame 3: Low light / darkness
    dark_img = clear_img.copy()
    enhancer = ImageEnhance.Brightness(dark_img)
    dark_img = enhancer.enhance(0.3)
    # Add noise
    dark_np = np.array(dark_img)
    noise = np.random.randint(-20, 20, dark_np.shape, dtype=np.int16)
    dark_np = np.clip(dark_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    dark_img = Image.fromarray(dark_np)
    for _ in range(3):
        frames.append(dark_img.convert('P', palette=Image.ADAPTIVE, colors=128))
    
    # Frame 4: Sandstorm
    sand_img = clear_img.copy()
    sand_img = sand_img.filter(ImageFilter.GaussianBlur(radius=3))
    sand_overlay = Image.new('RGB', sand_img.size, (210, 180, 140))
    sand_img = Image.blend(sand_img, sand_overlay, alpha=0.6)
    for _ in range(3):
        frames.append(sand_img.convert('P', palette=Image.ADAPTIVE, colors=128))
    
    output_path = Path(output_dir) / 'challenging_conditions.gif'
    print(f"Saving {len(frames)} frames...")
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # 500ms per frame
        loop=0
    )
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Created challenging conditions animation: {file_size_mb:.1f} MB")
    print(f"   Cycles through: clear → fog → low light → sandstorm")
    
    return output_path


if __name__ == "__main__":
    # Create all visualizations
    create_challenging_conditions_comparison()
    create_challenging_conditions_gif()
    create_foggy_flyover()
    
    print("\n✨ All challenging conditions visualizations created!")


