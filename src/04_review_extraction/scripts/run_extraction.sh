#!/bin/bash
# Entry point wrapper for review extraction
# ALWAYS uses the virtual environment at romance-novel-nlp-research/.venv

# Source shared venv setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv_setup.sh" || exit 1

# Run extraction using venv Python
cd "$PROJECT_ROOT" || exit 1
"$VENV_PYTHON" "$SCRIPT_DIR/../core/extract_reviews.py" "$@"

