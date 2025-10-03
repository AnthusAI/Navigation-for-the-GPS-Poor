#!/bin/bash
# Setup script for Navigation for the GPS Poor project

echo "Setting up Navigation for the GPS Poor project environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'navigation-gps-poor'..."
conda create -n navigation-gps-poor python=3.11 -y

# Activate environment and install dependencies
echo "📚 Installing dependencies..."
conda activate navigation-gps-poor
conda install opencv numpy matplotlib scipy jupyter requests tqdm pillow pytest -y

echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "    conda activate navigation-gps-poor"
echo ""
echo "To run tests:"
echo "    python -m pytest tests/"
echo ""
echo "To start Jupyter notebook:"
echo "    jupyter notebook chapters/1/demo.ipynb"
