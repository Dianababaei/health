#!/bin/bash

# Generate all synthetic datasets
# This script runs the dataset generator to create train/val/test splits
# and multi-day datasets with circadian patterns

echo "=========================================="
echo "Synthetic Dataset Generation"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import numpy, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required packages (numpy, pandas) not installed"
    echo "Run: pip install -r requirements.txt"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Create output directory
mkdir -p data/synthetic

# Run the dataset generator
echo "Running dataset generator..."
cd src/data
python3 dataset_generator.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Dataset generation complete!"
    echo "=========================================="
    echo ""
    echo "Generated files:"
    echo "  - data/synthetic/train.csv"
    echo "  - data/synthetic/val.csv"
    echo "  - data/synthetic/test.csv"
    echo "  - data/synthetic/multiday_*.csv"
    echo "  - data/synthetic/dataset_metadata.json"
    echo ""
else
    echo ""
    echo "❌ Dataset generation failed"
    exit 1
fi
