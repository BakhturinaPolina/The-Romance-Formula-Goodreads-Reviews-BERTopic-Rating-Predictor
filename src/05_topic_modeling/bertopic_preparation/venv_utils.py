"""
Virtual environment utilities for topic modeling scripts.

This module provides functions to verify and get paths for the project's
virtual environment, ensuring all scripts use the correct venv.
"""

import sys
from pathlib import Path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Go up from bertopic_preparation/ to project root
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent.parent


def get_venv_python() -> Path:
    """
    Get the path to the virtual environment Python executable.
    
    Returns:
        Path to venv python3 executable
        
    Raises:
        FileNotFoundError: If venv doesn't exist
    """
    project_root = get_project_root()
    venv_python = project_root / "romance-novel-nlp-research" / ".venv" / "bin" / "python3"
    
    if not venv_python.exists():
        raise FileNotFoundError(
            f"Virtual environment not found at {venv_python}\n"
            f"All scripts MUST use the virtual environment at romance-novel-nlp-research/.venv"
        )
    
    return venv_python


def verify_venv() -> None:
    """
    Verify that the current Python is using the project's virtual environment.
    
    Raises:
        SystemExit: If not using the correct venv
    """
    current_python = Path(sys.executable).resolve()
    expected_python = get_venv_python().resolve()
    
    if current_python != expected_python:
        print("⚠️  WARNING: Not using project virtual environment!", file=sys.stderr)
        print(f"   Current Python: {current_python}", file=sys.stderr)
        print(f"   Expected Python: {expected_python}", file=sys.stderr)
        print("", file=sys.stderr)
        print("   Use one of these instead:", file=sys.stderr)
        print(f"   - {expected_python} <script.py>", file=sys.stderr)
        print("", file=sys.stderr)
        sys.exit(1)

