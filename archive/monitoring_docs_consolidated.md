# Review Extraction Monitoring

## Quick Start

### Option 1: Use the wrapper script (easiest)

```bash
# Monitor with default PID (155101)
./src/review_extraction/monitor.sh

# Monitor with custom PID
./src/review_extraction/monitor.sh 12345
```

### Option 2: Use Python script directly

```bash
# Basic usage (auto-detects log file)
python3 src/review_extraction/monitor_extraction.py

# Custom PID and interval
python3 src/review_extraction/monitor_extraction.py --pid 155101 --interval 5

# Full options
python3 src/review_extraction/monitor_extraction.py \
    --pid 155101 \
    --log-file logs/extract_reviews_20251111_190005.log \
    --output-file data/processed/romance_reviews_english.csv \
    --interval 10
```

## What the Monitor Shows

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

## Process States

- **R** = Running (actively processing)
- **S** = Sleeping (waiting for I/O, normal)
- **T** = Stopped/Paused (can be resumed with `kill -CONT PID`)
- **Z** = Zombie (process terminated but not cleaned up)

## Tips

- Default update interval is 10 seconds
- Press `Ctrl+C` to stop monitoring
- The script auto-detects the latest log file if not specified
- Progress updates appear in the log every 5,000 reviews processed

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
- Script will try to auto-detect latest log file
- Or specify manually with `--log-file` option

