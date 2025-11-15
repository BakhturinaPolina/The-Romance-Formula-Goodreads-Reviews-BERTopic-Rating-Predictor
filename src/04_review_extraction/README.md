# Stage 04: Review Extraction

## Purpose

This stage extracts and filters English-language reviews from the Goodreads reviews dataset.

## ⚠️ Virtual Environment Requirement

**ALL scripts MUST be run using the virtual environment at `romance-novel-nlp-research/.venv`.**

The wrapper scripts in `scripts/` (`monitor.sh`, `run_extraction.sh`) automatically use this venv via shared `venv_setup.sh`. If running Python scripts directly, always use:

```bash
romance-novel-nlp-research/.venv/bin/python3 script.py
```

**Never use system Python or other virtual environments - always use `romance-novel-nlp-research/.venv` first.**

For full project-wide rules, see `.cursor/rules/venv-requirement.mdc`.

## Structure

```
04_review_extraction/
├── core/                    # Main extraction logic
│   └── extract_reviews.py  # Main extraction script
├── monitoring/              # Monitoring and time estimation
│   ├── monitor_extraction.py  # Real-time monitoring
│   ├── estimate_time.py      # Time estimation utility
│   └── log_parser.py          # Shared log parsing utilities
├── scripts/                 # Wrapper scripts for running and monitoring
│   ├── run_extraction.sh     # Extraction wrapper (uses venv)
│   ├── monitor.sh            # Monitoring wrapper (uses venv)
│   └── venv_setup.sh         # Shared venv setup (sourced by wrappers)
└── utils/                   # Utility scripts
    └── review_dataset.py    # Dataset review utility
```

## Input Files

- `data/raw/goodreads_reviews_*.json.gz` - Raw review data
- `data/processed/*.csv` - Book metadata with book IDs

## Output Files

- `data/processed/*_reviews_english.csv` - Filtered English reviews
- `logs/extract_reviews_*.log` - Extraction logs

## How to Run

### Main Extraction

**Recommended (uses venv automatically):**
```bash
cd src/04_review_extraction
./scripts/run_extraction.sh
```

Or directly with venv:
```bash
cd src/04_review_extraction
romance-novel-nlp-research/.venv/bin/python3 core/extract_reviews.py
```

Or using module syntax (with venv):
```bash
romance-novel-nlp-research/.venv/bin/python3 -m src.04_review_extraction.core.extract_reviews
```

### Monitoring

#### Quick Start (Recommended)

```bash
# Monitor with default PID (155101)
./scripts/monitor.sh

# Monitor with custom PID
./scripts/monitor.sh 12345
```

#### Using Python Scripts Directly (with venv)

```bash
# Real-time monitoring
romance-novel-nlp-research/.venv/bin/python3 monitoring/monitor_extraction.py --pid 155101 --interval 10

# Time estimation
romance-novel-nlp-research/.venv/bin/python3 monitoring/estimate_time.py
```

#### Monitoring Features

The monitoring tools provide:

1. **Process Status**
   - PID, State (Running/Stopped/Sleeping)
   - CPU and Memory usage
   - Total runtime

2. **Progress Statistics** (from log file)
   - Reviews processed
   - Processing rate (reviews/sec)
   - Matched reviews count and percentage
   - English reviews count and percentage
   - Reviews written to CSV
   - Elapsed time
   - Delta since last check

3. **Output File Info**
   - File size
   - Total lines (including header)
   - Number of reviews extracted

4. **Recent Log Entries**
   - Last 5 log entries
   - Color-coded by log level (ERROR=red, WARNING=yellow, INFO=cyan)

#### Process States

- **R** = Running (actively processing)
- **S** = Sleeping (waiting for I/O, normal)
- **T** = Stopped/Paused (can be resumed with `kill -CONT PID`)
- **Z** = Zombie (process terminated but not cleaned up)

#### Monitoring Tips

- Default update interval is 10 seconds
- Press `Ctrl+C` to stop monitoring
- Scripts auto-detect the latest log file if not specified
- Progress updates appear in the log every 5,000 reviews processed

## Dependencies

- pandas
- langdetect
- gzip (standard library)
- json (standard library)

## Example Usage

```bash
# Run extraction (takes several hours for full dataset)
./scripts/run_extraction.sh

# Or with venv directly
romance-novel-nlp-research/.venv/bin/python3 core/extract_reviews.py

# Monitor progress (uses venv automatically)
./scripts/monitor.sh

# Estimate remaining time (with venv)
romance-novel-nlp-research/.venv/bin/python3 monitoring/estimate_time.py
```

## Key Features

- Language detection and filtering
- Book ID matching
- Real-time progress monitoring
- Time estimation
- Resume functionality (skips already extracted reviews)

## Troubleshooting

### Process not found
- Process may have completed or crashed
- Check if process exists: `ps -p PID`
- Check log file for completion message

### No progress in log
- Process may be stuck or waiting
- Check process state: `ps -p PID -o state`
- Check recent log entries for errors

### Log file not found
- Scripts will try to auto-detect latest log file
- Or specify manually with `--log-file` option
