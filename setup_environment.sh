#!/bin/bash
# Setup script for Navigation for the GPS Poor project

echo "Setting up Navigation for the GPS Poor project environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment from file
echo "📦 Creating conda environment 'navigation-gps-poor' from environment.yml..."
echo "This might take a few minutes."
conda env create -f environment.yml

# Check if the environment was created successfully
if [ $? -ne 0 ]; then
    echo "❌ Failed to create conda environment. Please check for errors above."
    exit 1
fi

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
