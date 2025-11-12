# Review Extraction Module

This module extracts English-language reviews for romance books from the Goodreads reviews dataset.

## Overview

The review extraction process:
1. Loads book IDs from the main romance books dataset
2. Filters reviews from `goodreads_reviews_romance.json.gz` matching those book IDs
3. Detects and keeps only English reviews using language detection
4. Outputs a CSV file with columns: `review_id`, `review_text`, `rating`, `book_id`

## Files

- **`extract_reviews.py`**: Main extraction script that processes reviews line-by-line
- **`monitor_extraction.py`**: Real-time monitoring script for tracking extraction progress
- **`monitor.sh`**: Quick wrapper script for monitoring
- **`estimate_time.py`**: Script to estimate remaining time for extraction completion
- **`review_dataset.py`**: Utility script to review Goodreads dataset structure
- **`MONITORING.md`**: Detailed monitoring guide
- **`REVIEW_EXTRACTION_PAUSE_INFO.md`**: Historical pause/resume information

## Quick Start

### Running Extraction

```bash
# Activate virtual environment
source romance-novel-nlp-research/.venv/bin/activate

# Run extraction (will run in foreground)
python3 src/review_extraction/extract_reviews.py

# Or run in background
nohup python3 src/review_extraction/extract_reviews.py > /tmp/extract_reviews.log 2>&1 &
```

### Monitoring Progress

```bash
# Quick monitoring with wrapper script (auto-detects PID and log file)
./src/review_extraction/monitor.sh [PID]

# Or use Python script directly
python3 src/review_extraction/monitor_extraction.py \
    --pid <PID> \
    --log-file logs/extract_reviews_YYYYMMDD_HHMMSS.log \
    --output-file data/processed/romance_reviews_english.csv \
    --interval 10
```

### Estimating Time Remaining

```bash
# Auto-detects latest log file
python3 src/review_extraction/estimate_time.py

# Or specify log file
python3 src/review_extraction/estimate_time.py --log-file logs/extract_reviews_YYYYMMDD_HHMMSS.log
```

## Input/Output

### Input Files
- **Books CSV**: `anna_archive_romance_pipeline/data/processed/romance_books_main_final.csv`
  - Must contain column `book_id_list_en` with list of book IDs
- **Reviews File**: `data/raw/goodreads_reviews_romance.json.gz`
  - Gzipped JSONL file with one review per line
  - Total reviews: ~3.6 million

### Output File
- **Output CSV**: `data/processed/romance_reviews_english.csv`
  - Columns: `review_id`, `review_text`, `rating`, `book_id`
  - Only English reviews matching the book IDs

## Processing Details

### Language Detection
- Uses `langdetect` library for language detection
- Only reviews detected as English (`en`) are kept
- Reviews shorter than 10 characters are skipped

### Progress Logging
- Progress logged every 5,000 reviews processed
- Logs include:
  - Total reviews processed
  - Processing rate (reviews/sec)
  - Matched reviews count and percentage
  - English reviews count and percentage
  - Reviews written to CSV
  - Elapsed time

### Performance
- Processing rate: ~100-110 reviews/second
- Total processing time: ~9-10 hours for full dataset
- Memory efficient: Processes line-by-line without loading entire file

## Monitoring Features

The monitoring script provides:
1. **Process Status**: PID, state, CPU, memory usage, runtime
2. **Progress Statistics**: From log file parsing
   - Reviews processed
   - Processing rate
   - Matched/English/Written counts and percentages
   - Delta since last check
3. **Output File Info**: Size, line count, number of reviews
4. **Recent Log Entries**: Last 5 log entries with color coding

## Time Estimation

The `estimate_time.py` script calculates:
- Remaining reviews to process
- Time remaining estimates (optimistic, current rate, conservative)
- Estimated completion times
- Progress percentage

Based on:
- Total reviews in file: 3,565,378
- Current processing rate from log file
- Current progress from log file

## Log Files

Log files are created in `logs/` directory with format:
```
extract_reviews_YYYYMMDD_HHMMSS.log
```

Each log file contains:
- Initialization information
- Book ID loading progress
- Review extraction progress (every 5,000 reviews)
- Final statistics and completion message

## Troubleshooting

### Process Not Found
- Process may have completed or crashed
- Check log file for completion message
- Verify process exists: `ps -p <PID>`

### No Progress in Log
- Process may be stuck or waiting
- Check process state: `ps -p <PID> -o state`
- Check recent log entries for errors

### Log File Not Found
- Script will try to auto-detect latest log file
- Or specify manually with `--log-file` option

### Slow Processing
- Language detection is CPU-intensive
- Processing rate may vary based on system load
- Typical rate: 100-110 reviews/second

## Dependencies

- `pandas`: For reading CSV files
- `langdetect`: For language detection
- Standard library: `gzip`, `json`, `csv`, `logging`, `pathlib`

## Notes

- The script processes reviews sequentially (cannot be parallelized easily due to language detection)
- Output file is overwritten on each run (no append mode)
- The script does not support resume from interruption (would need to restart)
- For very large datasets, consider running in background with monitoring

