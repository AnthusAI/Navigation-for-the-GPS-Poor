# Navigation for the GPS Poor - Agent Planning Document

## Project Overview
This project is a comprehensive tutorial series teaching computer vision and navigation techniques for scenarios where GPS is unavailable or unreliable. The project is structured as a multi-chapter guide with practical implementations and datasets.

## Project Structure
- `README.md` - Main project introduction and navigation
- `AGENTS.md` - This file containing overall planning and agent context
- `environment.yml` - Conda environment specification
- `setup_environment.sh` - Automatic environment setup script
- `chapters/N/` - Individual chapter directories
  - `index.md` - Chapter article content
  - `demo.ipynb` - Jupyter notebook with practical implementation
- `src/` - Shared source code used by all chapters
  - `datasets.py` - Dataset fetching utilities (KITTI, TUM, etc.)
  - `utils.py` - Common computer vision functions
  - `feature_matching.py` - Feature detection and matching
  - `pose_estimation.py` - Camera pose estimation
  - `visual_odometry.py` - Visual odometry implementation
- `tests/` - Comprehensive test suite for all modules

## Chapter Plan

### Chapter 1: Simple Visual Odometry with OpenCV
**Goal**: Implement basic visual odometry using feature matching
**Dataset**: KITTI Visual Odometry Dataset
**Key Concepts**:
- Feature detection and matching (ORB, SIFT)
- Essential matrix estimation
- Camera pose recovery
- Trajectory visualization

**Implementation Tasks**:
- [x] Dataset fetching from KITTI
- [x] Feature detection pipeline
- [x] Motion estimation
- [x] Trajectory plotting
- [x] Error analysis

### Chapter 2: Stereo Visual Odometry
**Goal**: Solve the scale ambiguity problem using a second camera to perceive depth.
**Dataset**: KITTI Visual Odometry Dataset (Stereo Pairs)
**Key Concepts**:
- Stereo correspondence and disparity
- Triangulation for 3D depth perception
- 3D-to-2D motion estimation (PnP algorithm)
- Eliminating the need for ground truth scale

**Implementation Tasks**:
- [x] Load stereo image pairs from KITTI
- [x] Match features between left/right frames to find disparity
- [x] Triangulate 3D points from stereo matches
- [x] Track 3D points across time to estimate camera motion
- [x] Build and visualize the scaled trajectory

### Chapter 3: SLAM Fundamentals
**Goal**: Build a basic SLAM system
**Dataset**: TUM RGB-D Dataset
**Key Concepts**:
- [x] Simultaneous Localization and Mapping
- [x] Loop closure detection
- [x] Bundle adjustment basics
- [x] Map representation

### Chapter 4: Deep Learning for Visual Navigation
**Goal**: Implement learning-based navigation
**Dataset**: Custom dataset + synthetic data
**Key Concepts**:
- CNN-based pose estimation
- Reinforcement learning for navigation
- Transfer learning techniques

### Chapter 5: Sensor Fusion
**Goal**: Combine multiple sensors for robust navigation
**Dataset**: EuRoC MAV Dataset
**Key Concepts**:
- IMU integration
- Kalman filtering
- Multi-sensor calibration
- Robust estimation

### Chapter 6: Advanced SLAM Techniques
**Goal**: Implement state-of-the-art SLAM methods
**Dataset**: Various benchmark datasets
**Key Concepts**:
- Graph-based SLAM
- Direct methods
- Semantic SLAM
- Long-term autonomy

## Environment Setup (CRITICAL for Agents)

**ALWAYS activate the project environment before working:**
```bash
conda activate navigation-gps-poor
```

**If environment doesn't exist, set it up:**
```bash
# Option 1: Automatic setup (recommended)
./setup_environment.sh

# Option 2: From environment file
conda env create -f environment.yml

# Option 3: Manual setup
conda create -n navigation-gps-poor python=3.11 -y
conda activate navigation-gps-poor
conda install opencv numpy matplotlib scipy jupyter requests tqdm pillow pytest -y
```

**Verify setup:**
```bash
# Test all modules work
python -c "from src.datasets import KITTIDatasetFetcher; print('✅ Environment ready')"

# Run test suite
python -m pytest tests/
```

## Technical Requirements
- Python 3.11+ (managed via conda environment)
- OpenCV 4.10+
- NumPy, SciPy, Matplotlib
- Jupyter Notebooks
- PyTorch (for deep learning chapters - install when needed)
- pytest (for testing)

## Current Session Accomplishments
1. ✅ Establish project structure with proper src/ and tests/ directories
2. ✅ Create AGENTS.md with overall plan and environment setup
3. ✅ Set up Chapter 1 structure with comprehensive content
4. ✅ Implement KITTI dataset fetching utility with full functionality
5. ✅ Create initial Chapter 1 content and demo notebook
6. ✅ Build comprehensive test suite (23 tests passing)
7. ✅ Set up dedicated conda environment with all dependencies
8. ✅ Create environment setup scripts and documentation

## Code Architecture
- **Modular Design**: All functionality split into testable modules in `src/`
- **Shared Codebase**: All chapters use the same source code, building upon each other
- **Comprehensive Testing**: Full test coverage for all modules
- **Environment Isolation**: Dedicated conda environment prevents dependency conflicts

## Key Modules Created
- `src/datasets.py` - KITTI dataset fetching with download, parsing, and loading
- `src/feature_matching.py` - ORB/SIFT feature detection and matching
- `src/pose_estimation.py` - Essential matrix and PnP pose estimation
- `src/visual_odometry.py` - Complete monocular VO implementation
- `src/utils.py` - Common computer vision utilities and plotting

## Notes for Future Sessions
- **ALWAYS activate environment first**: `conda activate navigation-gps-poor`
- Each chapter builds on the shared codebase in `src/`
- Run tests before making changes: `python -m pytest tests/`
- Emphasize practical implementation over theory
- Include error analysis and troubleshooting sections
- Consider computational requirements for each chapter
- Update notebook imports to use new `src/` modules
