# Scripts Directory

This directory contains command-line scripts for the Artemis Health system.

## Dataset Generation Scripts

### generate_datasets.py

Main script for generating simulated cattle sensor datasets with ground truth labels.

**Usage:**

```bash
# Generate all three dataset time scales
python scripts/generate_datasets.py

# Generate specific dataset
python scripts/generate_datasets.py --dataset short
python scripts/generate_datasets.py --dataset medium
python scripts/generate_datasets.py --dataset long --duration 180

# Specify output directory and seed
python scripts/generate_datasets.py --output data/simulated --seed 42

# Show help
python scripts/generate_datasets.py --help
```

**Outputs:**
- `data/simulated/short_term_7d.csv` - 7-day dataset
- `data/simulated/medium_term_30d.csv` - 30-day dataset
- `data/simulated/long_term_90d.csv` - 90-day dataset (or 180d)
- `data/simulated/metadata_*.json` - Metadata for each dataset
- `data/simulated/splits/` - Train/validation/test splits

### test_dataset_generation.py

Test script that generates smaller sample datasets to verify the system works.

**Usage:**

```bash
python scripts/test_dataset_generation.py
```

Runs three tests:
1. 1-day short-term test
2. 3-day medium-term test
3. 30-day test with estrus cycles

### quick_test.py

Quick integration test that verifies all components import and work together.

**Usage:**

```bash
python scripts/quick_test.py
```

Runs in seconds and confirms the system is operational.

## Requirements

All scripts require the dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Documentation

See `docs/dataset_documentation.md` for complete documentation on:
- Dataset format and structure
- Ground truth labels
- Health event scenarios
- Usage examples
