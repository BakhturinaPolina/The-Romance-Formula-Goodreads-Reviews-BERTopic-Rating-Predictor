# BERTopic + OCTIS Pipeline for Reviews Corpus

This directory contains the adapted BERTopic+OCTIS pipeline for topic modeling on Goodreads reviews.

## Overview

The pipeline has been adapted from the original novels corpus to work with reader reviews. Key adaptations:

- **Raw text for embeddings**: Uses unpreprocessed text (critical for sentence transformers)
- **Book-level analysis**: Uses `work_id` as labels for book-level topic analysis
- **Quality correlation**: Preserves `pop_tier` for correlation analysis with book ratings
- **GPU/CPU fallback**: Automatically detects and uses GPU when available
- **Single model**: Uses `all-mpnet-base-v2` for initial testing

## Files

### Core Scripts
- **`bertopic_plus_octis.py`** - Main optimization script (adapted)
- **`optimizer.py`** - Bayesian Optimization class (no changes needed)
- **`load_raw_sentences.py`** - Loads raw (unpreprocessed) sentences from reviews
- **`sample_test_dataset.py`** - Creates test datasets for quick testing

### Utilities
- **`topic_npy_to_json.py`** - Converts topic arrays to JSON
- **`restart_script.py`** - Auto-restart script for crashes

### Documentation
- **`ADAPTATION_PLAN.md`** - Detailed adaptation plan
- **`ADAPTATION_SUMMARY.md`** - Summary of completed adaptations
- **`README.md`** - This file

## Quick Start

### 1. Create Test Dataset

```bash
cd /home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research
source romance-novel-nlp-research/.venv/bin/activate

# Sample 10K sentences for testing
python src/reviews_analysis/BERTopic_OCTIS/sample_test_dataset.py \
    --n_samples 10000 \
    --output data/interim/review_sentences_test_10k.parquet \
    --stratify pop_tier \
    --preserve-reviews
```

### 2. Run Optimization

```bash
# The script will automatically detect and use the test dataset if it exists
python src/reviews_analysis/BERTopic_OCTIS/bertopic_plus_octis.py
```

Or use the test runner:

```bash
./src/reviews_analysis/BERTopic_OCTIS/run_test.sh
```

## Important Notes

### Raw Text for Embeddings

**CRITICAL**: The pipeline uses **raw, unpreprocessed text** for sentence transformer embeddings. This is important because:

- Transformer models are trained on raw text
- Preprocessing (lowercasing, normalization) can hurt embedding quality
- The current parquet file has cleaned text, so for production you should use `load_raw_sentences_from_reviews()` which loads from source reviews

### Test vs Production Mode

- **Test mode**: If `data/interim/review_sentences_test_10k.parquet` exists, the script will use it (may have cleaned text)
- **Production mode**: If test file doesn't exist, loads raw sentences from source reviews (truly raw text)

### Virtual Environment

All scripts should be run in:
```
/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research/romance-novel-nlp-research/.venv
```

## Outputs

The pipeline creates:

1. **OCTIS Corpus**: `data/interim/octis_reviews/corpus.tsv`
   - Format: `sentence_text \t partition \t work_id`
   - Uses `work_id` as label for book-level topic analysis

2. **Embeddings**: `data/interim/octis_reviews/embeddings_all-mpnet-base-v2.pkl`
   - Pre-calculated embeddings for faster optimization

3. **Optimization Results**: `data/interim/octis_reviews/optimization_results/all-mpnet-base-v2/result.json`
   - Best hyperparameters found by Bayesian Optimization

## Data Structure

### Input (from `load_raw_sentences_from_reviews()`)
- `sentence_text` - **Raw, unpreprocessed** sentence text
- `review_id` - Source review ID
- `work_id` - Book/work ID (used as OCTIS label)
- `pop_tier` - Quality tier (thrash/mid/top) - preserved for correlation analysis
- `rating` - Review rating
- `sentence_index` - Position within review
- `n_sentences_in_review` - Total sentences in review

### OCTIS Format
- Sentence text (raw)
- Partition: `train`
- Label: `work_id` (book-level grouping)

## Configuration

### Embedding Model
Currently: `all-mpnet-base-v2`
- Can be changed in `bertopic_plus_octis.py` line ~441

### Test Dataset Size
For initial testing, the script can:
1. Use test parquet file if it exists (created by `sample_test_dataset.py`)
2. Or load from source with `max_sentences` parameter

## GPU/CPU Support

The pipeline automatically:
- Detects GPU availability for UMAP/HDBSCAN (cuML)
- Falls back to CPU versions if GPU not available
- Uses GPU for embeddings if PyTorch CUDA is available
- Logs which device is being used

## Next Steps

1. ✅ Data loading adapted
2. ✅ OCTIS format conversion adapted
3. ✅ GPU/CPU fallback added
4. ✅ Test dataset sampling script created
5. ⏳ Test with small dataset (10K sentences)
6. ⏳ Verify embeddings calculation
7. ⏳ Test optimization loop
8. ⏳ Scale to larger datasets

## Troubleshooting

### Import Errors
If you get import errors, make sure you're in the project root and the venv is activated:
```bash
cd /home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research
source romance-novel-nlp-research/.venv/bin/activate
```

### GPU Not Detected
The script will automatically fall back to CPU. Check logs for messages like:
- `✓ GPU libraries (cuML) available` - GPU will be used
- `⚠ GPU libraries (cuML) not available` - CPU will be used

### Memory Issues
For large datasets, start with a small test dataset (10K sentences) first. The full dataset has 8.6M sentences and requires significant memory.

## References

- Original pipeline: `romantic_novels_project_code/src/stage03_modeling/`
- BERTopic documentation: https://maartengr.github.io/BERTopic/
- OCTIS documentation: https://github.com/MIND-Lab/OCTIS

