#!/bin/bash
# Wrapper script to run prepare_bertopic_input.py with proper logging and venv

VENV_PYTHON="/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research/romance-novel-nlp-research/.venv/bin/python"
SCRIPT_PATH="src/reviews_analysis/bertopic_preparation/prepare_bertopic_input.py"
LOG_FILE="/tmp/bertopic_prep_monitor.log"
PROJECT_ROOT="/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research"

cd "$PROJECT_ROOT" || exit 1

echo "=========================================="
echo "Starting BERTopic Preparation Script"
echo "=========================================="
echo "Python: $VENV_PYTHON"
echo "Script: $SCRIPT_PATH"
echo "Log file: $LOG_FILE"
echo "Project root: $PROJECT_ROOT"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Run the script and redirect both stdout and stderr to log file
"$VENV_PYTHON" "$SCRIPT_PATH" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "Script finished with exit code: $EXIT_CODE"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

exit $EXIT_CODE

