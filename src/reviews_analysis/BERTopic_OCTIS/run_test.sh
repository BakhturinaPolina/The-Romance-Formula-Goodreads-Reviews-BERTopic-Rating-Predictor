#!/bin/bash
# Test runner script for BERTopic+OCTIS pipeline
# Uses test dataset for quick verification

set -e

# Project root
PROJECT_ROOT="/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research"
VENV_PATH="$PROJECT_ROOT/romance-novel-nlp-research/.venv"

# Activate virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"

echo "=========================================="
echo "BERTopic+OCTIS Test Runner"
echo "=========================================="
echo "Virtual environment: $VENV_PATH"
echo "Python: $(which python)"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Step 1: Create test dataset (10K sentences)
echo "Step 1: Creating test dataset (10K sentences)..."
python src/reviews_analysis/BERTopic_OCTIS/sample_test_dataset.py \
    --n_samples 10000 \
    --output data/interim/review_sentences_test_10k.parquet \
    --stratify pop_tier \
    --preserve-reviews

echo ""
echo "Step 1 complete!"
echo ""

# Step 2: Run BERTopic+OCTIS (with test dataset)
echo "Step 2: Running BERTopic+OCTIS optimization..."
echo "Note: This will use the test dataset (10K sentences)"
echo ""

# For testing, modify bertopic_plus_octis.py to use test dataset
# Or create a test version that loads from the test parquet
python src/reviews_analysis/BERTopic_OCTIS/bertopic_plus_octis.py

echo ""
echo "Test run complete!"
echo "Check outputs in: data/interim/octis_reviews/"

