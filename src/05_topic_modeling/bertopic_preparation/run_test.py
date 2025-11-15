#!/usr/bin/env python3
"""
Test runner script for BERTopic+OCTIS pipeline.

Uses test dataset for quick verification.
"""

import sys
import subprocess
from pathlib import Path

from venv_utils import get_project_root, get_venv_python, verify_venv

# Verify we're using the correct venv
verify_venv()

project_root = get_project_root()
venv_python = get_venv_python()
bertopic_octis_dir = project_root / "src" / "05_topic_modeling" / "BERTopic_OCTIS"

# Change to project root
import os
os.chdir(project_root)

print("=" * 50)
print("BERTopic+OCTIS Test Runner")
print("=" * 50)
print(f"Virtual environment: {venv_python}")
print(f"Python: {venv_python}")
print()

# Step 1: Create test dataset (10K sentences)
print("Step 1: Creating test dataset (10K sentences)...")
sample_script = bertopic_octis_dir / "sample_test_dataset.py"
result = subprocess.run(
    [
        str(venv_python),
        str(sample_script),
        "--n_samples", "10000",
        "--output", "data/interim/review_sentences_test_10k.parquet",
        "--stratify", "pop_tier",
        "--preserve-reviews"
    ],
    check=False
)

if result.returncode != 0:
    print(f"Error creating test dataset (exit code: {result.returncode})")
    sys.exit(result.returncode)

print()
print("Step 1 complete!")
print()

# Step 2: Run BERTopic+OCTIS (with test dataset)
print("Step 2: Running BERTopic+OCTIS optimization...")
print("Note: This will use the test dataset (10K sentences)")
print()

bertopic_script = bertopic_octis_dir / "bertopic_plus_octis.py"
result = subprocess.run(
    [str(venv_python), str(bertopic_script)],
    check=False
)

if result.returncode != 0:
    print(f"Error running BERTopic+OCTIS (exit code: {result.returncode})")
    sys.exit(result.returncode)

print()
print("Test run complete!")
print("Check outputs in: data/intermediate/octis_reviews/")

