#!/usr/bin/env python3
"""
Creates a smooth fly-over animation from the Davis-Monthan boneyard imagery.
This provides a realistic simulation of aerial navigation over continuous terrain.
"""

from PIL import Image
from pathlib import Path
from tqdm import tqdm

def create_flyover_animation(
    image_path: str = "data/boneyard/davis_monthan_aerial.jpg",
    output_path: str = "chapters/4/images/boneyard_flyover.gif",
    frame_width: int = 1280,
    frame_height: int = 720,
    step_size: int = 10,
    duration: int = 50
):
    """
    Creates a smooth 16:9 fly-over animation from a large aerial image.
    
    Args:
        image_path: Path to the source aerial image
        output_path: Where to save the GIF
        frame_width: Width of each frame (16:9 aspect ratio)
        frame_height: Height of each frame
        step_size: Pixels to move per frame (smaller = smoother)
        duration: Milliseconds per frame (50ms = 20 FPS)
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")
    
    print(f"Loading source image: {image_path}")
    source_img = Image.open(image_path)
    src_width, src_height = source_img.size
    print(f"Source dimensions: {src_width}x{src_height}")
    
    # Calculate how many frames we can create
    num_frames = (src_width - frame_width) // step_size
    print(f"\nGenerating {num_frames} frames at {1000/duration:.0f} FPS")
    print(f"Frame size: {frame_width}x{frame_height} (16:9)")
    print(f"Step size: {step_size} pixels/frame")
    
    frames = []
    
    # Pan horizontally across the image
    for i in tqdm(range(num_frames), desc="Creating frames"):
        x_offset = i * step_size
        # Center vertically
        y_offset = (src_height - frame_height) // 2
        
        # Crop the frame
        box = (x_offset, y_offset, x_offset + frame_width, y_offset + frame_height)
        frame = source_img.crop(box)
        frames.append(frame)
    
    print(f"\n💾 Optimizing and saving animation to {output_path}...")
    # Convert to palette mode with reduced colors for much smaller file size
    optimized_frames = []
    for frame in tqdm(frames, desc="Optimizing frames"):
        # Reduce to 32 colors for better compression (aerial imagery compresses well)
        frame_pal = frame.convert('P', palette=Image.ADAPTIVE, colors=32)
        optimized_frames.append(frame_pal)
    
    optimized_frames[0].save(
        output_path,
        save_all=True,
        append_images=optimized_frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    duration_sec = (num_frames * duration) / 1000
    print(f"✅ Done! Created {num_frames}-frame animation")
    print(f"   File size: {file_size_mb:.1f} MB")
    print(f"   Duration: {duration_sec:.1f} seconds")
    print(f"   Smooth fly-over across Davis-Monthan AFB Boneyard")


if __name__ == "__main__":
    # Create a smooth, high-quality fly-over animation
    # Longer flight distance with faster movement per frame (~10 second animation)
    create_flyover_animation(
        image_path="data/boneyard/davis_monthan_aerial.jpg",
        output_path="chapters/4/images/boneyard_flyover.gif",
        frame_width=1280,  # 16:9 aspect ratio
        frame_height=720,
        step_size=8,  # Faster movement = covers more distance per frame
        duration=250  # 4 FPS for smooth ~10 second animation
    )

