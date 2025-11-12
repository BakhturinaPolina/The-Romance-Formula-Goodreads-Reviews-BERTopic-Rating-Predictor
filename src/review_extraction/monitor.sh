#!/bin/bash
# Quick wrapper script to monitor review extraction

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default PID (can be overridden)
PID=${1:-155101}

# Auto-detect log file
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE=$(ls -t "$LOG_DIR"/extract_reviews_*.log 2>/dev/null | head -1)

# Output file
OUTPUT_FILE="$PROJECT_ROOT/data/processed/romance_reviews_english.csv"

# Run monitor
python3 "$SCRIPT_DIR/monitor_extraction.py" \
    --pid "$PID" \
    --log-file "${LOG_FILE:-$LOG_DIR/extract_reviews_20251111_190005.log}" \
    --output-file "$OUTPUT_FILE" \
    --interval 10

