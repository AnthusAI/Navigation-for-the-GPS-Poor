import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from pathlib import Path
from PIL import Image
import glob
from tqdm import tqdm
import json
import geopandas as gpd
from shapely.geometry import box
import subprocess
import tempfile
import shutil
import xml.etree.ElementTree as ET

# This script will be used to generate visualizations for Chapter 4.

def read_geotransform(aux_xml_path):
    """
    Reads the GeoTransform from a .aux.xml file.
    Returns a tuple of (min_lon, pixel_width, _, max_lat, _, -pixel_height).
    """
    try:
        tree = ET.parse(aux_xml_path)
        root = tree.getroot()
        geotransform_elem = root.find('GeoTransform')
        if geotransform_elem is not None:
            # Parse the GeoTransform string
            # Format: minX, pixelWidth, 0, maxY, 0, -pixelHeight
            values = [float(x) for x in geotransform_elem.text.strip().split(',')]
            return tuple(values)
    except Exception as e:
        print(f"Error reading geotransform from {aux_xml_path}: {e}")
    return None


def stitch_tiles_by_coordinates(tile_paths):
    """
    Stitches georeferenced tiles by reading their .aux.xml files
    and placing them at their correct geographic positions.
    This is a simple custom implementation that doesn't require GDAL.
    """
    print("Stitching tiles using geographic coordinates...")
    
    # Load all tiles and their geotransforms
    tiles_data = []
    for tile_path in tile_paths:
        aux_path = tile_path + '.aux.xml'
        if not Path(aux_path).exists():
            continue
            
        geotransform = read_geotransform(aux_path)
        if geotransform is None:
            continue
            
        img = cv2.imread(tile_path)
        if img is None:
            continue
            
        tiles_data.append({
            'path': tile_path,
            'image': img,
            'geotransform': geotransform
        })
    
    if len(tiles_data) < 2:
        print("Not enough tiles with valid geotransforms.")
        return None
    
    # Extract geographic bounds for all tiles
    # GeoTransform format: (min_lon, pixel_width, 0, max_lat, 0, -pixel_height)
    all_bounds = []
    for td in tiles_data:
        gt = td['geotransform']
        h, w = td['image'].shape[:2]
        min_lon = gt[0]
        max_lat = gt[3]
        pixel_width = gt[1]
        pixel_height = -gt[5]  # negative in geotransform
        max_lon = min_lon + w * pixel_width
        min_lat = max_lat - h * pixel_height
        all_bounds.append((min_lon, min_lat, max_lon, max_lat))
    
    # Calculate the overall bounding box
    global_min_lon = min(b[0] for b in all_bounds)
    global_min_lat = min(b[1] for b in all_bounds)
    global_max_lon = max(b[2] for b in all_bounds)
    global_max_lat = max(b[3] for b in all_bounds)
    
    # Use the pixel size from the first tile (assuming all tiles have the same resolution)
    pixel_width = tiles_data[0]['geotransform'][1]
    pixel_height = -tiles_data[0]['geotransform'][5]
    
    # Calculate output canvas size
    output_width = int((global_max_lon - global_min_lon) / pixel_width)
    output_height = int((global_max_lat - global_min_lat) / pixel_height)
    
    print(f"Creating mosaic of size {output_width}x{output_height} pixels...")
    
    if output_width > 10000 or output_height > 10000:
        print("Mosaic would be too large. Reducing to manageable size.")
        return None
    
    # Create output canvas
    output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Place each tile at its correct position
    for td in tqdm(tiles_data, desc="Placing tiles"):
        gt = td['geotransform']
        img = td['image']
        h, w = img.shape[:2]
        
        # Calculate pixel position in output canvas
        min_lon = gt[0]
        max_lat = gt[3]
        
        x_offset = int((min_lon - global_min_lon) / pixel_width)
        y_offset = int((global_max_lat - max_lat) / pixel_height)
        
        # Ensure we don't go out of bounds
        if x_offset < 0 or y_offset < 0:
            continue
        if x_offset + w > output_width or y_offset + h > output_height:
            continue
        
        # Place the tile (simple overwrite, no blending)
        output[y_offset:y_offset+h, x_offset:x_offset+w] = img
    
    return output


def load_geodata():
    """Loads all geojson tile boundaries into a GeoDataFrame."""
    print("Loading geospatial data for all tiles...")
    geojson_files = glob.glob('data/rareplanes/train/geojson_aircraft_tiled/*.geojson')
    tile_data = []

    for f in tqdm(geojson_files, desc="Parsing GeoJSON files"):
        try:
            with open(f, 'r') as gj:
                data = json.load(gj)
            
            # Extract all coordinates and create a bounding box for the tile
            all_coords = []
            if not data['features']:
                continue
            
            for feature in data['features']:
                geom_type = feature['geometry']['type']
                coords = feature['geometry']['coordinates']
                if geom_type == 'Polygon':
                    all_coords.extend(coords[0])
                elif geom_type == 'MultiPolygon':
                    for poly in coords:
                        all_coords.extend(poly[0])

            if not all_coords:
                continue

            lons, lats = zip(*all_coords)
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            
            tile_data.append({
                'filename': Path(f).stem,
                'geometry': box(min_lon, min_lat, max_lon, max_lat),
                'min_lon': min_lon,
                'max_lon': max_lon
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping corrupt or invalid geojson file: {f}, error: {e}")
            continue

    gdf = gpd.GeoDataFrame(tile_data, crs="EPSG:4326")
    print(f"Loaded {len(gdf)} tile boundaries.")
    return gdf

def find_horizontal_sequence(gdf, seq_length=3):
    """
    Finds a sequence of horizontally adjacent tiles with more robust logic.
    """
    print(f"Searching for a sequence of {seq_length} adjacent tiles...")
    # Create a spatial index for faster searching
    sindex = gdf.sindex

    for index, start_tile in tqdm(gdf.iterrows(), total=gdf.shape[0], desc="Finding tile sequence"):
        sequence = [start_tile]
        current_tile = start_tile
        
        try:
            for _ in range(seq_length - 1):
                # Define a narrow search area just to the right of the current tile
                min_lon, min_lat, max_lon, max_lat = current_tile.geometry.bounds
                search_width = (max_lon - min_lon) * 0.1 # 10% of tile width
                search_box = box(max_lon - search_width, min_lat, max_lon + search_width, max_lat)
                
                # Use the spatial index to find possible intersecting candidates
                possible_matches_idx = list(sindex.intersection(search_box.bounds))
                candidates = gdf.iloc[possible_matches_idx]
                
                # Filter for candidates that are actually to the right and intersect
                next_tile_candidates = candidates[
                    (candidates.geometry.intersects(search_box)) &
                    (candidates['min_lon'] > current_tile['min_lon']) &
                    (candidates.index != current_tile.name)
                ]
                
                if next_tile_candidates.empty:
                    break
                
                # Find the closest one to the right
                next_tile = next_tile_candidates.sort_values('min_lon').iloc[0]
                sequence.append(next_tile)
                current_tile = next_tile
            
            if len(sequence) == seq_length:
                print(f"Found a valid sequence starting with {sequence[0]['filename']}")
                return sequence
                
        except (IndexError, KeyError):
            continue
            
    return None

def create_neural_network_diagram():
    """Generates and saves a diagram of a simple neural network."""
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    ax.axis('off')

    # Layer positions
    layer_sizes = [4, 5, 5, 2]
    layer_x_coords = [0.1, 0.4, 0.7, 0.95]
    node_radius = 0.04

    # Draw nodes and connections
    node_positions = []
    for i, (size, x_coord) in enumerate(zip(layer_sizes, layer_x_coords)):
        layer_positions = []
        y_coords = np.linspace(0.1, 0.9, size)
        for y_coord in y_coords:
            circle = mpatches.Circle((x_coord, y_coord), node_radius, facecolor='skyblue', edgecolor='black', zorder=4)
            ax.add_patch(circle)
            layer_positions.append((x_coord, y_coord))
        node_positions.append(layer_positions)

    # Draw connections
    for i in range(len(layer_sizes) - 1):
        for start_node in node_positions[i]:
            for end_node in node_positions[i+1]:
                ax.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 'gray', zorder=1, alpha=0.5)

    # Add labels
    ax.text(0.1, 1.0, 'Input Layer\\n(e.g., Image Pixels)', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.55, 1.0, 'Hidden Layers\\n(Pattern Recognition)', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.95, 1.0, 'Output Layer\\n(e.g., Pose)', ha='center', fontsize=12, fontweight='bold')

    plt.suptitle("Anatomy of a Neural Network", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = 'chapters/4/images/neural_network_diagram.png'
    plt.savefig(save_path)
    print(f"Saved neural network diagram to {save_path}")
    plt.close(fig)


def find_adjacent_tiles(max_tiles=20):
    """
    Finds a set of truly adjacent tiles by analyzing their geographic bounds.
    Returns a list of tile paths that form a contiguous area.
    """
    print("Searching for adjacent tiles...")
    all_tiles = glob.glob('data/rareplanes/train/PS-RGB_tiled/*.png')
    
    # Read bounds for all tiles
    tiles_with_bounds = []
    for tile_path in tqdm(all_tiles[:500], desc="Reading tile metadata"):  # Limit search
        aux_path = tile_path + '.aux.xml'
        if not Path(aux_path).exists():
            continue
        
        gt = read_geotransform(aux_path)
        if gt is None:
            continue
        
        # Quick size check - assuming 512x512 tiles
        img_size = 512
        min_lon, max_lat = gt[0], gt[3]
        pixel_width, pixel_height = gt[1], -gt[5]
        max_lon = min_lon + img_size * pixel_width
        min_lat = max_lat - img_size * pixel_height
        
        tiles_with_bounds.append({
            'path': tile_path,
            'bounds': (min_lon, min_lat, max_lon, max_lat),
            'center': ((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)
        })
    
    if len(tiles_with_bounds) < 3:
        return []
    
    # Find a dense cluster of adjacent tiles
    # Start with a random tile and grow a contiguous region
    selected = [tiles_with_bounds[0]]
    remaining = tiles_with_bounds[1:]
    
    while len(selected) < max_tiles and remaining:
        # Find the tile from remaining that is closest to any selected tile
        best_tile = None
        best_dist = float('inf')
        
        for candidate in remaining:
            for sel_tile in selected:
                # Calculate center-to-center distance
                dx = candidate['center'][0] - sel_tile['center'][0]
                dy = candidate['center'][1] - sel_tile['center'][1]
                dist = (dx**2 + dy**2) ** 0.5
                
                if dist < best_dist:
                    best_dist = dist
                    best_tile = candidate
        
        if best_tile and best_dist < 0.01:  # Adjacent threshold in degrees
            selected.append(best_tile)
            remaining.remove(best_tile)
        else:
            break  # No more adjacent tiles
    
    print(f"Found {len(selected)} adjacent tiles")
    return [t['path'] for t in selected]


def create_flyover_gif():
    """
    Creates a smooth fly-over GIF by stitching truly adjacent georeferenced tiles.
    """
    print("Creating fly-over preview GIF...")

    # Find tiles that are actually adjacent to each other
    tile_paths = find_adjacent_tiles(max_tiles=15)
    
    if len(tile_paths) < 5:
        print(f"Not enough adjacent tiles found. Creating grid-based fly-over instead...")
        # Create a large grid of tiles and pan across it
        diverse_tiles = glob.glob('data/rareplanes/train/PS-RGB_tiled/*.png')[:24]  # 6x4 grid
        
        if len(diverse_tiles) < 12:
            print("Not enough tiles for visualization")
            return
        
        # Arrange tiles in a grid
        grid_cols, grid_rows = 6, 4
        tile_size = 512  # Original tile size
        grid_width = grid_cols * tile_size
        grid_height = grid_rows * tile_size
        
        print(f"Creating {grid_cols}x{grid_rows} tile grid ({grid_width}x{grid_height} pixels)...")
        grid_img = Image.new('RGB', (grid_width, grid_height))
        
        for idx, tile_path in enumerate(diverse_tiles[:grid_cols * grid_rows]):
            try:
                img = Image.open(tile_path).convert('RGB')
                col = idx % grid_cols
                row = idx // grid_cols
                x, y = col * tile_size, row * tile_size
                grid_img.paste(img, (x, y))
            except Exception as e:
                print(f"Error loading tile {tile_path}: {e}")
                continue
        
        # Create smooth panning animation across the grid
        # 16:9 aspect ratio
        frame_width = 1920
        frame_height = 1080
        frames = []
        
        # Pan horizontally across the grid
        step = 15  # Smaller step = smoother animation
        num_frames = (grid_width - frame_width) // step
        
        print(f"Creating {num_frames} frames for smooth fly-over...")
        for i in tqdm(range(num_frames), desc="Generating frames"):
            x_offset = i * step
            # Center vertically or pan slightly
            y_offset = min(i * 2, grid_height - frame_height)  # Slight vertical pan
            
            box = (x_offset, y_offset, x_offset + frame_width, y_offset + frame_height)
            frame = grid_img.crop(box)
            # Scale down to reasonable size for GIF (720p width at 16:9)
            frame = frame.resize((720, 405), Image.Resampling.LANCZOS)  # 16:9 ratio
            frames.append(frame)
        
        if not frames:
            print("Failed to generate frames")
            return
        
        save_path = 'chapters/4/images/rareplanes_flyover.gif'
        frames[0].save(
            save_path, 
            save_all=True, 
            append_images=frames[1:], 
            duration=50,  # 50ms = 20 FPS for smooth animation
            loop=0
        )
        print(f"Saved grid fly-over GIF with {len(frames)} frames to {save_path}")
        return
    
    print(f"Found {len(tile_paths)} adjacent tiles. Stitching...")
    
    # Stitch the adjacent tiles
    stitched = stitch_tiles_by_coordinates(tile_paths)
    
    if stitched is None or stitched.size == 0:
        print("Stitching failed.")
        return
    
    # Convert to PIL Image
    stitched_rgb = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
    stitched_img = Image.fromarray(stitched_rgb)
    
    # Crop to remove excessive black borders
    # Find the bounding box of non-black pixels
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        stitched_img = stitched_img.crop((x, y, x + w, y + h))
        print(f"Cropped mosaic to {w}x{h} (removed black borders)")
    
    # Create animated fly-over
    frames = []
    width, height = stitched_img.size
    print(f"Final mosaic size: {width}x{height}")
    
    if width < 800:
        print("Mosaic too narrow. Using full image.")
        frames = [stitched_img.resize((600, 400), Image.Resampling.LANCZOS)]
    else:
        frame_width = min(1000, width - 50)
        frame_height = int(frame_width * 0.6)
        step = max(30, width // 40)  # Slower animation
        
        for x_offset in tqdm(range(0, width - frame_width, step), desc="Creating frames"):
            y_offset = max(0, (height - frame_height) // 2)
            y_end = min(y_offset + frame_height, height)
            
            box = (x_offset, y_offset, x_offset + frame_width, y_end)
            frame = stitched_img.crop(box)
            frame = frame.resize((600, 400), Image.Resampling.LANCZOS)
            frames.append(frame)
    
    if not frames:
        print("Failed to generate frames.")
        return

    save_path = 'chapters/4/images/rareplanes_flyover.gif'
    frames[0].save(
        save_path, 
        save_all=True, 
        append_images=frames[1:], 
        duration=150,  # Slower: 150ms per frame
        loop=0
    )
    print(f"Saved fly-over GIF with {len(frames)} frames to {save_path}")


def create_sample_image_visualization():
    """Saves a high-quality sample image from the dataset for context."""
    image_path = 'data/rareplanes/train/PS-RGB_tiled/100_1040010029990A00_tile_319.png'
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.imshow(img_rgb)
        ax.axis('off')
        ax.set_title('Sample Image from the RarePlanes Dataset', fontsize=16, fontweight='bold')
        
        save_path = 'chapters/4/images/rareplanes_sample.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved sample image to {save_path}")
        plt.close(fig)

    except (FileNotFoundError, Exception) as e:
        print(f"Could not create sample image: {e}")

def create_cnn_filter_diagram():
    """Generates a diagram explaining how a CNN filter (kernel) works, now in color."""
    image_path = 'data/rareplanes/train/PS-RGB_tiled/1_104005000FDC8D00_tile_114.png'
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (256, 256))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    except (FileNotFoundError, Exception) as e:
        print(f"Error loading RarePlanes image: {e}. Using a placeholder.")
        img_rgb = np.full((256, 256, 3), 128, dtype=np.uint8)
        img_gray = np.zeros((256, 256), dtype=np.uint8)
        img_gray[64:192, 64:192] = 128
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        cv2.circle(img_gray, (128, 128), 32, 255, -1)

    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    filtered_img = cv2.filter2D(img_gray, -1, kernel)

    fig = plt.figure(figsize=(15, 7), facecolor='white')

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_rgb)
    ax1.set_title('1. Original Color Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    patch_size = 32
    patch_start = (70, 80)
    rect = mpatches.Rectangle(patch_start, patch_size, patch_size, linewidth=2, edgecolor='cyan', facecolor='none', zorder=10)
    ax1.add_patch(rect)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.axis('off')
    ax2.set_title('2. Filter Applied to All Channels', fontsize=14, fontweight='bold')
    
    img_patch = img_rgb[patch_start[1]:patch_start[1]+patch_size, patch_start[0]:patch_start[0]+patch_size]
    axins_patch = inset_axes(ax2, width="50%", height="50%", loc='center')
    axins_patch.imshow(img_patch)
    axins_patch.set_title('Color Image Patch')
    axins_patch.axis('off')
    ax2.text(0.5, 0.2, "A 3D filter processes R, G, and B\\nchannels to produce one value.", 
             ha='center', va='center', style='italic', fontsize=10)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(filtered_img, cmap='gray')
    ax3.set_title('3. Resulting Feature Map', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    rect_out = mpatches.Rectangle(patch_start, patch_size, patch_size, linewidth=2, edgecolor='cyan', facecolor='none', zorder=10)
    ax3.add_patch(rect_out)

    con = ConnectionPatch(xyA=(patch_start[0] + patch_size, patch_start[1] + patch_size/2), xyB=(0, 1), 
                          coordsA=ax1.transData, coordsB=ax2.transAxes,
                          arrowstyle="->", shrinkB=5, color='red', lw=2)
    fig.add_artist(con)

    con2 = ConnectionPatch(xyA=(1, 0.5), xyB=(patch_start[0], patch_start[1] + patch_size/2), 
                           coordsA=ax2.transAxes, coordsB=ax3.transData,
                           arrowstyle="->", shrinkB=5, color='red', lw=2)
    fig.add_artist(con2)

    plt.suptitle("How a CNN Filter Works on a Color Image", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    save_path = 'chapters/4/images/cnn_filter_diagram_color.png'
    plt.savefig(save_path)
    print(f"Saved CNN filter diagram to {save_path}")
    plt.close(fig)

def create_training_loop_diagram():
    """Generates a diagram of the deep learning training loop."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    plt.suptitle("The Deep Learning Training Loop", fontsize=20, fontweight='bold')

    # Define positions and properties for the boxes
    positions = {
        'Show': (2.5, 6),
        'Guess': (7.5, 6),
        'Compare': (7.5, 2),
        'Adjust': (2.5, 2)
    }
    
    box_style = dict(boxstyle='round,pad=0.5', fc='lightcyan', ec='black', lw=1)
    arrow_style = dict(arrowstyle='simple,head_length=.5,head_width=.3,tail_width=.1', 
                       fc='gray', ec='none', connectionstyle='arc3,rad=-0.3')

    # Create boxes
    for name, (x, y) in positions.items():
        if name == 'Show':
            text = "1. Show\\n(Feed image to CNN)"
        elif name == 'Guess':
            text = "2. Guess\\n(CNN outputs a pose)"
        elif name == 'Compare':
            text = "3. Compare & Correct\\n(Measure error vs. true pose)"
        else: # Adjust
            text = "4. Adjust\\n(Update CNN's internals)"
        ax.text(x, y, text, ha='center', va='center', fontsize=12, bbox=box_style)

    # Create arrows
    ax.add_patch(mpatches.FancyArrowPatch(positions['Show'], positions['Guess'], **arrow_style))
    ax.add_patch(mpatches.FancyArrowPatch(positions['Guess'], positions['Compare'], **arrow_style))
    ax.add_patch(mpatches.FancyArrowPatch(positions['Compare'], positions['Adjust'], **arrow_style))
    ax.add_patch(mpatches.FancyArrowPatch(positions['Adjust'], positions['Show'], **arrow_style))
    
    ax.text(5, 0.5, "Repeat millions of times...", ha='center', fontsize=14, style='italic')

    save_path = 'chapters/4/images/training_loop_diagram.png'
    plt.savefig(save_path)
    print(f"Saved training loop diagram to {save_path}")
    plt.close(fig)


def main():
    print("Generating visualizations for Chapter 4...")
    create_sample_image_visualization()
    create_neural_network_diagram()
    create_cnn_filter_diagram()
    create_training_loop_diagram()
    create_flyover_gif()
    print("Done.")

if __name__ == '__main__':
    main()
