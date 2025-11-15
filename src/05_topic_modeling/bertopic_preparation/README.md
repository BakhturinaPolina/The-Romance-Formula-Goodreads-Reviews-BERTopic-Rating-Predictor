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
- **`venv_utils.py`** - Virtual environment utilities for venv verification

### Execution Scripts (Python)
- **`run_preparation.py`** - Wrapper script to run the preparation with proper logging and venv
- **`monitor_preparation.py`** - Continuous monitoring script (updates every 30 seconds)
- **`check_status.py`** - Quick status check script
- **`run_test.py`** - Test runner for BERTopic+OCTIS pipeline

### Documentation
- **`DATA_MAPPING_ISSUE.md`** - Documentation of work_id vs book_id mapping solution
- **`README.md`** - This file

## ⚠️ Virtual Environment Requirement

**ALL scripts MUST be run using the virtual environment at `romance-novel-nlp-research/.venv`.**

All Python scripts automatically verify they're using the correct venv. Always use:

```bash
romance-novel-nlp-research/.venv/bin/python3 script.py
```

## Usage

### Running the Preparation

**Recommended (uses venv automatically):**
```bash
cd src/05_topic_modeling
romance-novel-nlp-research/.venv/bin/python3 bertopic_preparation/run_preparation.py
```

**Or run directly:**
```bash
romance-novel-nlp-research/.venv/bin/python3 bertopic_preparation/prepare_bertopic_input.py
```

### Monitoring Progress

**Option 1: Continuous monitoring (in separate terminal)**
```bash
romance-novel-nlp-research/.venv/bin/python3 bertopic_preparation/monitor_preparation.py
```

**Option 2: Quick status check**
```bash
romance-novel-nlp-research/.venv/bin/python3 bertopic_preparation/check_status.py
```

### Running Tests

```bash
romance-novel-nlp-research/.venv/bin/python3 bertopic_preparation/run_test.py
```

## Input Files

- `data/processed/romance_subdataset_6000.csv` - Books dataset (6,000 books)
- `data/processed/romance_reviews_english.csv` - English reviews dataset
- `data/processed/romance_books_main_final.csv` - Main dataset (for book_id mapping)

## Output Files

- `data/processed/review_sentences_for_bertopic.parquet` - Sentence-level dataset for BERTopic
- `data/processed/review_sentences_temp/chunk_*.parquet` - Temporary chunk files (during processing)
- `/tmp/bertopic_prep_monitor.log` - Processing log file

## Process Details

### Sentence Splitting
- Uses spaCy `en_core_web_sm` model for sentence segmentation
- Processes reviews in chunks of 20,000 reviews
- Saves incremental progress to chunk files

### Data Mapping
- Reviews are mapped from `book_id` to `work_id` using the mapping in `utils/data_loading.py`
- Only reviews for books in the 6,000 book dataset are included

### Progress Tracking
- Progress is logged to `/tmp/bertopic_prep_monitor.log`
- Monitor script displays:
  - Process status (running/stopped)
  - CPU and memory usage
  - Latest progress lines
  - Key statistics (chunk progress, processing rate, ETA)

## Dependencies

- pandas
- spacy (with `en_core_web_sm` model)
- pyarrow (for parquet files)
- Standard library: pathlib, logging, typing

## Troubleshooting

### Process Not Running
- Check if process crashed: `tail -50 /tmp/bertopic_prep_monitor.log`
- Verify input files exist
- Check disk space

### Slow Processing
- Sentence splitting is CPU-intensive
- Processing rate: ~100-200 reviews/sec (depends on review length)
- Full dataset (1.2M reviews) takes several hours

### Memory Issues
- Process uses ~2-4GB RAM
- Chunk files are saved incrementally to prevent data loss
- If process crashes, chunk files can be recovered from `data/processed/review_sentences_temp/`
