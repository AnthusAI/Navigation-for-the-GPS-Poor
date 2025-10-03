# Chapter 1 Visualizations

This directory contains visualizations for Chapter 1: Simple Visual Odometry with OpenCV.

## Generated Images

### Original KITTI Frames
- `frame_000.png` - First frame from KITTI sequence 00
- `frame_001.png` - Second frame from KITTI sequence 00  
- `frame_010.png` - Frame 10 from KITTI sequence 00

### Feature Detection & Matching
- `features_detected.png` - Shows 200 detected corner features (green circles) on the first frame
- `feature_matches.png` - Shows 50 matched features between consecutive frames (colored lines connecting same points)

### Trajectory & Analysis
- `trajectory.png` - 3D and top-down view of estimated vs. ground truth camera trajectory
- `error_analysis.png` - Position and rotation errors over time
- `statistics.txt` - Numerical statistics about the visual odometry performance

## Attribution

### KITTI Images
The original frames (`frame_*.png`) are from the KITTI Vision Benchmark Suite.

**Source**: [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)

**Citation**:
```
@INPROCEEDINGS{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}
```

### Derivative Works
The annotated images (`features_detected.png`, `feature_matches.png`) and analysis plots (`trajectory.png`, `error_analysis.png`) are derivative works created using OpenCV and our custom visual odometry implementation. These visualizations are generated to illustrate the concepts explained in Chapter 1.

## Regenerating Visualizations

To regenerate these visualizations:

```bash
# From project root, with conda environment activated
conda activate navigation-gps-poor
python chapters/1/generate_visualizations.py
```

This will:
1. Load KITTI sequence 00 data
2. Detect and match features
3. Run visual odometry
4. Generate all visualization images
5. Save them to this directory

## Usage in Article

These images are referenced in `chapters/1/index.md` to help readers understand:
- What the KITTI dataset looks like
- How feature detection identifies corners and edges
- How features are matched between frames
- What the resulting trajectory looks like
- How errors accumulate over time



