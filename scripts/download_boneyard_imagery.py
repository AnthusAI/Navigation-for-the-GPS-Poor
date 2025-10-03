#!/usr/bin/env python3
"""
Downloads high-resolution aerial imagery of Davis-Monthan AFB Boneyard.

This script creates a large, seamless aerial image suitable for pose estimation
demonstrations in Chapter 4. We use freely available satellite imagery.

Davis-Monthan AFB Boneyard location: 32.1665° N, 110.8563° W
"""

import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import math

def download_boneyard_imagery(output_path: str = "data/boneyard/davis_monthan_aerial.jpg",
                              zoom: int = 17, tiles_wide: int = 4, tiles_high: int = 3):
    """
    Downloads aerial imagery of Davis-Monthan AFB using OpenStreetMap tiles.
    
    We'll use the ESRI World Imagery basemap which provides high-resolution
    aerial photography worldwide, including military installations that are
    publicly visible.
    
    Args:
        output_path: Where to save the final stitched image
        zoom: Zoom level (17 = ~2.4m/pixel, good detail)
        tiles_wide: Number of tiles horizontally
        tiles_high: Number of tiles vertically
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"✅ Imagery already exists at {output_path}")
        return output_path
    
    # Davis-Monthan coordinates
    lat, lon = 32.1665, -110.8563
    
    print(f"📍 Downloading aerial imagery of Davis-Monthan AFB Boneyard")
    print(f"   Location: {lat}°N, {lon}°W")
    print(f"   Zoom level: {zoom} (~{2.4 * (2 ** (19 - zoom)):.1f}m per pixel)")
    print(f"   Tiles: {tiles_wide}x{tiles_high}")
    
    # Convert lat/lon to tile coordinates at given zoom level
    def lat_lon_to_tile(lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x_tile, y_tile
    
    center_x, center_y = lat_lon_to_tile(lat, lon, zoom)
    
    # Calculate tile range (center the view on the boneyard)
    x_start = center_x - tiles_wide // 2
    y_start = center_y - tiles_high // 2
    
    # Use ESRI World Imagery (high quality, publicly available)
    tile_url_template = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    
    tile_size = 256
    final_width = tiles_wide * tile_size
    final_height = tiles_high * tile_size
    
    print(f"   Output size: {final_width}x{final_height} pixels")
    print(f"\nDownloading tiles...")
    
    # Create final image
    final_image = Image.new('RGB', (final_width, final_height))
    
    # Download and stitch tiles
    for dy in range(tiles_high):
        for dx in range(tiles_wide):
            x = x_start + dx
            y = y_start + dy
            
            tile_url = tile_url_template.format(z=zoom, y=y, x=x)
            
            try:
                response = requests.get(tile_url, timeout=10)
                response.raise_for_status()
                tile_img = Image.open(BytesIO(response.content))
                
                # Paste into final image
                paste_x = dx * tile_size
                paste_y = dy * tile_size
                final_image.paste(tile_img, (paste_x, paste_y))
                
                print(f"   ✓ Downloaded tile ({dx+1}/{tiles_wide}, {dy+1}/{tiles_high})")
                
            except Exception as e:
                print(f"   ✗ Failed to download tile ({dx+1}/{tiles_wide}, {dy+1}/{tiles_high}): {e}")
                # Continue with other tiles
    
    # Save final image
    print(f"\n💾 Saving stitched image to {output_path}...")
    final_image.save(output_path, 'JPEG', quality=95)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Done! Saved {final_width}x{final_height} image ({file_size_mb:.1f} MB)")
    print(f"   This covers approximately {tiles_wide * 0.6:.1f} x {tiles_high * 0.6:.1f} km")
    
    return output_path


if __name__ == "__main__":
    # Download a large, high-resolution image of the boneyard
    # Zoom 17 gives ~2.4m/pixel resolution
    # 6x4 tiles = 1536x1024 pixels ≈ 3.7 x 2.5 km coverage
    download_boneyard_imagery(
        output_path="data/boneyard/davis_monthan_aerial.jpg",
        zoom=17,
        tiles_wide=6,
        tiles_high=4
    )


