#!/bin/bash
# Quick wrapper script to monitor review extraction
# ALWAYS uses the virtual environment at romance-novel-nlp-research/.venv

# Source shared venv setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv_setup.sh" || exit 1

# Default PID (can be overridden)
PID=${1:-155101}

# Auto-detect log file
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE=$(ls -t "$LOG_DIR"/extract_reviews_*.log 2>/dev/null | head -1)

# Output file
OUTPUT_FILE="$PROJECT_ROOT/data/processed/romance_reviews_english.csv"

# Run monitor using venv Python
"$VENV_PYTHON" "$SCRIPT_DIR/../monitoring/monitor_extraction.py" \
    --pid "$PID" \
    --log-file "${LOG_FILE:-$LOG_DIR/extract_reviews_20251111_190005.log}" \
    --output-file "$OUTPUT_FILE" \
    --interval 10

