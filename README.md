# Navigation for the GPS Poor

Do you need to teach devices to find their way around without using GPS?  Good news!  There are ways to do that with machine learning, even for the GPU Poor.  Your little robots can navigate using computer vision and sensor fusion, even without GPS or GPUs.

Let's talk about practical navigation methods using computer vision, sensor fusion, and machine learning.  We'll play with working code that demonstrates everything that we talk about.

## Chapters

### [Chapter 1: Simple Visual Odometry with OpenCV](chapters/1/)
Learn the fundamentals of visual odometry using feature matching and camera pose estimation.

![Feature Matching Example](chapters/1/images/sequence_00_features.gif)
*Feature matching between consecutive KITTI frames - matching features are connected with lines*

- **Dataset**: KITTI Visual Odometry
- **Key Skills**: Feature detection, essential matrix, trajectory estimation
- **Difficulty**: Beginner

### [Chapter 2: Stereo Visual Odometry](chapters/2/)
Solve the scale ambiguity problem from Chapter 1 using a second camera to perceive depth.
- **Dataset**: KITTI Visual Odometry (Stereo Pairs)
- **Key Skills**: Stereo correspondence, disparity, 3D triangulation, scaled trajectory
- **Difficulty**: Intermediate

### [Chapter 3: SLAM Fundamentals](chapters/3/) *(Coming Soon)*
Build a basic Simultaneous Localization and Mapping system.
- **Dataset**: TUM RGB-D Dataset
- **Key Skills**: Loop closure, bundle adjustment, map representation
- **Difficulty**: Intermediate

### [Chapter 4: Deep Learning for Visual Navigation](chapters/4/) *(Coming Soon)*
Implement learning-based navigation using neural networks.
- **Dataset**: Custom + Synthetic
- **Key Skills**: CNN pose estimation, reinforcement learning
- **Difficulty**: Advanced

### [Chapter 5: Sensor Fusion](chapters/5/) *(Coming Soon)*
Combine multiple sensors for robust navigation.
- **Dataset**: EuRoC MAV Dataset
- **Key Skills**: IMU integration, Kalman filtering, calibration
- **Difficulty**: Advanced

### [Chapter 6: Advanced SLAM Techniques](chapters/6/) *(Coming Soon)*
Explore state-of-the-art SLAM methods and long-term autonomy.
- **Dataset**: Various benchmarks
- **Key Skills**: Graph SLAM, direct methods, semantic mapping
- **Difficulty**: Expert

## Getting Started

### Prerequisites
- Python 3.11+ (recommended)
- Anaconda or Miniconda
- Basic computer vision knowledge

### Environment Setup

**Option 1: Automatic Setup (Recommended)**
```bash
# Clone the repository
git clone <repository-url>
cd Navigation-for-the-GPS-Poor

# Run setup script
./setup_environment.sh

# Activate environment
conda activate navigation-gps-poor
```

**Option 2: Manual Setup**
```bash
# Create conda environment
conda create -n navigation-gps-poor python=3.11 -y

# Activate environment
conda activate navigation-gps-poor

# Install dependencies
conda install opencv numpy matplotlib scipy jupyter requests tqdm pillow pytest -y
```

**Option 3: From Environment File**
```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate navigation-gps-poor
```

### Verification
```bash
# Run tests to verify installation
python -m pytest tests/

# Test modules
python -c "from src.datasets import KITTIDatasetFetcher; print('✅ All modules working!')"
```

## Project Structure

```
├── README.md                    # This file
├── AGENTS.md                   # Project planning and context
├── LICENSE                     # MIT license
├── environment.yml             # Conda environment specification
├── setup_environment.sh        # Automatic setup script
├── pyproject.toml             # Python project configuration
├── requirements.txt           # Alternative pip requirements
├── chapters/                  # Individual tutorial chapters
│   └── 1/                    # Chapter 1: Visual Odometry
│       ├── index.md          # Chapter content
│       └── demo.ipynb        # Practical implementation
├── src/                      # Shared source code (used by all chapters)
│   ├── datasets.py          # Dataset fetching utilities
│   ├── utils.py             # Computer vision utilities
│   ├── feature_matching.py  # Feature detection and matching
│   ├── pose_estimation.py   # Camera pose estimation
│   └── visual_odometry.py   # Visual odometry implementation
└── tests/                   # Comprehensive test suite
    ├── test_datasets.py     # Dataset utility tests
    └── test_feature_matching.py  # Feature matching tests
```

## Contributing

This is an educational project. Suggestions for improvements, additional datasets, or new techniques are welcome!

## License

MIT License - Feel free to use this content for learning and teaching.
