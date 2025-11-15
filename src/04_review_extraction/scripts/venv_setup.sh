#!/bin/bash
# Shared virtual environment setup for review extraction scripts
# This script sets up VENV_PYTHON and PROJECT_ROOT variables
# Source this script in other bash scripts to avoid code duplication

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/romance-novel-nlp-research/.venv/bin/python3"

# Verify venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at $VENV_PYTHON" >&2
    exit 1
fi

# Export variables for use in sourcing scripts
export PROJECT_ROOT
export VENV_PYTHON
export SCRIPT_DIR

