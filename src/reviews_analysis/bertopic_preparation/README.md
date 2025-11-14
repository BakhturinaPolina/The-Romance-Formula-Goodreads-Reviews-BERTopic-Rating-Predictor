# BERTopic Preparation Stage

This directory contains all code and documentation for preparing the review dataset for BERTopic topic modeling.

## Overview

This stage converts review-level data into sentence-level data suitable for BERTopic analysis. The process:
1. Loads joined reviews and books data (6,000 books dataset)
2. Splits reviews into sentences using spaCy
3. Cleans and normalizes sentence text
4. Saves sentence-level dataset as parquet file

## Files

### Core Scripts
- **`prepare_bertopic_input.py`** - Main script that performs sentence splitting and dataset creation
- **`data_loading.py`** - Data loading utilities (located in parent directory `src/reviews_analysis/`)

### Execution Scripts
- **`run_bertopic_prep.sh`** - Wrapper script to run the preparation with proper logging and venv
- **`monitor_bertopic_prep.sh`** - Continuous monitoring script (updates every 30 seconds)
- **`check_bertopic_status.sh`** - Quick status check script

### Documentation
- **`DATA_MAPPING_ISSUE.md`** - Documentation of work_id vs book_id mapping solution
- **`README.md`** - This file

## Usage

### Running the Preparation

```bash
# From project root
cd /home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research

# Run the preparation script
./src/reviews_analysis/bertopic_preparation/run_bertopic_prep.sh

# Or run directly with Python
./romance-novel-nlp-research/.venv/bin/python src/reviews_analysis/bertopic_preparation/prepare_bertopic_input.py
```

### Monitoring Progress

**Option 1: Continuous monitoring (in separate terminal)**
```bash
./src/reviews_analysis/bertopic_preparation/monitor_bertopic_prep.sh
```

**Option 2: Quick status check**
```bash
./src/reviews_analysis/bertopic_preparation/check_bertopic_status.sh
```

**Option 3: View log directly**
```bash
tail -f /tmp/bertopic_prep_monitor.log
```

## Output

The script produces:
- **Output file**: `data/processed/review_sentences_for_bertopic.parquet`
- **Log file**: `/tmp/bertopic_prep_monitor.log`
- **Temporary chunks**: `data/processed/review_sentences_temp/` (cleaned up after completion)

## Results Summary

From the latest run (2025-11-14):
- **Total sentences**: 8,671,667
- **Unique reviews**: 965,418
- **Unique books**: 5,998 (out of 6,000)
- **Average sentences per review**: 8.98
- **Processing rate**: ~108 reviews/sec
- **Execution time**: ~152 minutes (~2.5 hours)
- **Output file size**: 508 MB

### Pop Tier Distribution
- **mid**: 4,193,576 sentences (48.4%)
- **top**: 3,429,717 sentences (39.5%)
- **thrash**: 1,048,374 sentences (12.1%)

## Data Structure

The output parquet file contains the following columns:
- `sentence_id` - Unique sentence identifier
- `sentence_text` - The sentence text (cleaned, lowercase, normalized whitespace)
- `review_id` - Source review ID
- `work_id` - Book/work ID
- `pop_tier` - Quality tier (thrash/mid/top)
- `rating` - Review rating (if available)
- `sentence_index` - Position of sentence within its review (0-based)
- `n_sentences_in_review` - Total number of sentences in the source review

## Technical Details

### Processing Strategy
- Processes reviews in chunks of 5,000 for better progress tracking
- Uses spaCy batch processing (`nlp.pipe()`) for efficient sentence splitting
- Saves chunks incrementally to prevent data loss
- Combines all chunks at the end into final dataset

### spaCy Configuration
- Model: `en_core_web_sm`
- Disabled components: `['tagger', 'parser', 'ner', 'lemmatizer', 'attribute_ruler']`
- Active components: `['tok2vec', 'sentencizer']`
- Optimized for sentence splitting only (faster processing)

### Text Cleaning
- Removes extra newlines
- Normalizes whitespace
- Converts to lowercase
- Filters out empty sentences (min length: 10 characters)

## Dependencies

- Python 3.12+
- pandas >= 2.3.0
- spacy >= 3.8.0
- en_core_web_sm spaCy model
- pyarrow >= 15.0.0 (for parquet)

## Related Files

- **Data loading**: `src/reviews_analysis/data_loading.py`
- **Input data**: 
  - `data/processed/romance_subdataset_6000.csv` (books)
  - `data/processed/romance_reviews_english_subdataset_6000.csv` (reviews)
- **Output data**: `data/processed/review_sentences_for_bertopic.parquet`

## Notes

- The script handles duplicate review_ids by removing duplicates (found ~0.40% duplicates)
- Reviews are mapped from `book_id` to `work_id` using the mapping in `data_loading.py`
- Temporary chunk files are automatically cleaned up after successful completion

