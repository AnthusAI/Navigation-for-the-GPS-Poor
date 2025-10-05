"""
Generates simple placeholder images for the conceptual visuals in Chapter 4.

This script uses Pillow to create and save PNG images with descriptive text,
which serve as placeholders until the final, more complex visuals are created.
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder(width, height, text, output_path):
    """Creates a single placeholder image."""
    img = Image.new('RGB', (width, height), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    
    # Use a basic font
    try:
        font = ImageFont.truetype("Arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text size and position
    text_bbox = d.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) / 2
    text_y = (height - text_height) / 2
    
    # Draw text
    d.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    # Save image
    img.save(output_path)
    print(f"✅ Created placeholder: {output_path}")

def main():
    """Generates all required placeholder images for Chapter 4."""
    print("--- Generating Placeholder Images ---")
    
    base_dir = os.path.dirname(__file__)
    image_dir = os.path.join(base_dir, '../images')
    os.makedirs(image_dir, exist_ok=True)

    placeholders = {
        "flight_path_overview.png": "Placeholder: 16x9 Overview\nof Flight Path from Desert to Base",
        "challenging_desert_flyover.gif": "Placeholder: Animation of Feature-Poor\nDesert Terrain Flyover",
        "ml_experiment_pipeline.png": "Placeholder: Diagram of ML Experiment Pipeline\n(Data -> Train -> Eval -> Viz)",
        "training_data_sampling.png": "Placeholder: Image of Full Map with\nGrid Overlay Showing Sampled Tiles",
        "simulated_flight_evaluation.gif": "Placeholder: Split-Screen Animation\n(Camera View vs. Map Prediction)"
    }
    
    # For GIF placeholders, we'll just create PNGs for now
    for filename, text in placeholders.items():
        output_path = os.path.join(image_dir, filename)
        if filename.endswith(".gif"):
            output_path = output_path.replace(".gif", ".png")
            
        # Create 16:9 aspect ratio images
        create_placeholder(1280, 720, text, output_path)
        
    print("--- Placeholders Generated ---")

if __name__ == "__main__":
    main()

