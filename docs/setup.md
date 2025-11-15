# Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd goodreads-topic-modeling
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 4. Install Additional Dependencies (if needed)

Some stages may require additional dependencies:

```bash
# For NLP preprocessing (spaCy model)
python -m spacy download en_core_web_sm

# For data audit (if using Makefile)
# Install make if not already available
```

### 5. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Check key packages
python -c "import pandas, spacy; print('Dependencies OK')"
```

## Data Setup

### Raw Data

Place Goodreads JSON files in `data/raw/`:
- `goodreads_books_*.json.gz`
- `goodreads_reviews_*.json.gz`
- `goodreads_*.json.gz` (other metadata files)

### Processed Data

Processed datasets will be created in `data/processed/` during pipeline execution.

## Quick Start

### Run a Single Stage

```bash
# Example: Run data quality pipeline
cd src/02_data_quality
python pipeline_runner.py
```

### Run Full Pipeline

Execute stages sequentially:
1. Data Integration
2. Data Quality
3. Text Preprocessing
4. Review Extraction
5. Topic Modeling
6. Shelf Normalization
7. Corpus Analysis

See main `README.md` for stage-specific commands.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **Missing Dependencies**: Run `pip install -r requirements.txt` again
3. **spaCy Model Missing**: Run `python -m spacy download en_core_web_sm`
4. **Path Errors**: Ensure you're running commands from project root

### Getting Help

- Check stage-specific README files in `src/[stage]/README.md`
- Review logs in `outputs/logs/`
- See `docs/replication_guide.md` for dataset adaptation

