"""
Generates the flight path animation for Chapter 4.

This script creates an animated GIF showing the flight path with consistent
styling to match the trajectory visualization. The aircraft stays fixed in
center while terrain scrolls underneath, with 4x zoom for detail.
"""
import sys
sys.path.append('../../')

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm

def create_flight_animation():
    """Main function to generate the flight path animation."""
    print("--- Generating Flight Path Animation ---")

    # --- 1. Configuration ---
    map_path = os.path.join(os.path.dirname(__file__), '../../../data/boneyard/davis_monthan_stitched_map.jpg')
    output_path = os.path.join(os.path.dirname(__file__), '../images/flight_path_animation.gif')

    # Animation parameters - aggressively optimized for 10MB limit
    width, height = 960, 540   # Smaller 16:9 for file size
    duration_ms = 200  # 5 FPS (200ms per frame)

    # Flight timing: 10 seconds of flight + 5 seconds pause
    flight_frames = 50   # 10 seconds at 5 FPS
    pause_frames = 25    # 5 seconds at 5 FPS
    total_frames = flight_frames + pause_frames

    zoom_factor = 8    # Pull back 2x from 16 (now 2x more zoom than original 4)

    # Flight path coordinates (same as trajectory visualization)
    start_coord = (5500, 4500)  # Desert start
    end_coord = (4167, 4167)    # Boneyard end

    # Styling to match trajectory visualization
    # Title text (easy to update as requested)
    title_text = "Path of Simulated Aircraft Flight"

    # Colors matching the trajectory plot
    path_color = (0, 255, 0)      # Lime green (matching actual path)
    aircraft_color = (255, 255, 255)  # White center dot
    aircraft_edge = (0, 0, 0)     # Black edge
    start_marker_color = (50, 205, 50)   # Lime green
    end_marker_color = (255, 0, 0)       # Red

    # --- 2. Load Map ---
    if not os.path.exists(map_path):
        print(f"❌ Map image not found at {map_path}")
        return
    
    print(f"Loading map from {map_path}...")
    full_map = Image.open(map_path).convert('RGB')
    map_width, map_height = full_map.size
    print(f"Map size: {map_width}x{map_height}")

    # --- 3. Generate Frames ---
    print(f"Generating {total_frames} frames ({flight_frames} flight + {pause_frames} pause)...")
    frames = []

    # Create the smooth path for the camera (only for flight portion)
    path_x = np.linspace(start_coord[0], end_coord[0], flight_frames)
    path_y = np.linspace(start_coord[1], end_coord[1], flight_frames)

    for i in tqdm(range(total_frames), desc="Creating frames"):
        # For flight frames, use interpolated position
        # For pause frames, stay at final position
        if i < flight_frames:
            cam_x, cam_y = path_x[i], path_y[i]
            frame_progress = i / flight_frames  # 0 to 1
        else:
            # Pause at final position
            cam_x, cam_y = end_coord[0], end_coord[1]
            frame_progress = 1.0

        # Define the crop box for the current frame (much smaller area due to 16x zoom)
        crop_width = width // zoom_factor
        crop_height = height // zoom_factor
        left = int(cam_x - crop_width / 2)
        top = int(cam_y - crop_height / 2)
        right = left + crop_width
        bottom = top + crop_height

        # Ensure crop stays within map bounds
        left = max(0, min(left, map_width - crop_width))
        top = max(0, min(top, map_height - crop_height))
        right = left + crop_width
        bottom = top + crop_height

        # Crop the frame from the main map and scale up to output size
        frame = full_map.crop((left, top, right, bottom)).convert("RGBA")
        frame = frame.resize((width, height), Image.LANCZOS)

        # Create a drawing context
        draw = ImageDraw.Draw(frame)

        # Draw the flight path trail up to the current point (lime green like visualization)
        trail_points = []

        # Determine how much of the trail to show
        if i < flight_frames:
            # During flight: show trail up to current position
            trail_end = i + 1
        else:
            # During pause: show full trail
            trail_end = flight_frames

        for j in range(max(0, min(trail_end - 30, trail_end)), trail_end):  # Show last 30 points
            # Convert map coordinates to cropped frame coordinates, then scale to output size
            trail_x_crop = path_x[j] - left
            trail_y_crop = path_y[j] - top
            # Scale from crop size to output size
            trail_x_scaled = (trail_x_crop / crop_width) * width
            trail_y_scaled = (trail_y_crop / crop_height) * height
            trail_points.append((trail_x_scaled, trail_y_scaled))

        if len(trail_points) > 1:
            # Draw white outline first (like visualization)
            draw.line(trail_points, fill=(255, 255, 255), width=8)
            # Draw lime green path on top
            draw.line(trail_points, fill=path_color, width=5)

        # Draw start point if visible in current frame
        start_x_crop = start_coord[0] - left
        start_y_crop = start_coord[1] - top
        if 0 <= start_x_crop <= crop_width and 0 <= start_y_crop <= crop_height:
            start_x_scaled = (start_x_crop / crop_width) * width
            start_y_scaled = (start_y_crop / crop_height) * height
            draw.ellipse(
                (start_x_scaled - 15, start_y_scaled - 15,
                 start_x_scaled + 15, start_y_scaled + 15),
                fill=start_marker_color, outline=(0, 0, 0), width=3
            )

        # Draw end point if visible in current frame
        end_x_crop = end_coord[0] - left
        end_y_crop = end_coord[1] - top
        if 0 <= end_x_crop <= crop_width and 0 <= end_y_crop <= crop_height:
            end_x_scaled = (end_x_crop / crop_width) * width
            end_y_scaled = (end_y_crop / crop_height) * height
            # Draw as square (like visualization)
            draw.rectangle(
                (end_x_scaled - 15, end_y_scaled - 15,
                 end_x_scaled + 15, end_y_scaled + 15),
                fill=end_marker_color, outline=(0, 0, 0), width=3
            )

        # Draw the aircraft marker at the center (white dot like visualization)
        center_x, center_y = width / 2, height / 2
        draw.ellipse(
            (center_x - 8, center_y - 8, center_x + 8, center_y + 8),
            fill=aircraft_color, outline=aircraft_edge, width=2
        )

        # Add title at bottom center (matching visualization style)
        # Use a font size that provides good readability and matches visual weight
        # of the trajectory plot title (not mathematical scaling)
        font_size = 18  # Readable size for 960x540 resolution

        try:
            # Try to load a font, fall back to default if not available
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None

        if font:
            # Get text size for centering
            bbox = draw.textbbox((0, 0), title_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position at bottom center
            text_x = (width - text_width) // 2
            text_y = height - text_height - 30

            # Draw text with black background (like visualization)
            padding = 15
            draw.rectangle(
                (text_x - padding, text_y - padding,
                 text_x + text_width + padding, text_y + text_height + padding),
                fill=(0, 0, 0, 200), outline=(255, 255, 255), width=2
            )
            draw.text((text_x, text_y), title_text, fill=(255, 255, 255), font=font)

        # Convert to palette mode for smaller file size
        frame_rgb = frame.convert("RGB")
        # Quantize to 128 colors for smaller file size while maintaining quality
        frame_quantized = frame_rgb.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        frames.append(frame_quantized)
    
    # --- 4. Save GIF ---
    if frames:
        print(f"Saving animated GIF to {output_path}...")

        # Create duration list: normal speed for flight, longer for pause
        durations = [duration_ms] * flight_frames + [duration_ms] * pause_frames

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,  # Enable optimization to reduce file size
            duration=durations,
            loop=0
        )

        # Check file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Animation saved successfully. File size: {file_size_mb:.2f} MB")

        if file_size_mb > 10:
            print("⚠️  Warning: File size exceeds 10MB. Consider reducing frames or resolution.")
    else:
        print("No frames were generated. Skipping GIF creation.")

    print("--- Animation Generation Finished ---")

if __name__ == "__main__":
    create_flight_animation()
