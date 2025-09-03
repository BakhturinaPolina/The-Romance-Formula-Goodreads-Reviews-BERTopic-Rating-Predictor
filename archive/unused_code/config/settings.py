"""
Configuration settings for Romance Novel NLP Research Project.

This module contains essential configuration parameters for data exploration
and basic project setup. Analysis-specific configurations will be added
as we progress through data exploration and determine requirements.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create essential directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  INTERMEDIATE_DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Target romance subgenres for analysis (research objective)
TARGET_SUBGENRES = [
    "contemporary romance",
    "historical romance", 
    "paranormal romance",
    "romantic suspense",
    "romantic fantasy",
    "science fiction romance"
]

# Data file paths (known dataset structure)
DATA_FILES = {
    "books": RAW_DATA_DIR / "goodreads_books_romance.json.gz",
    "reviews": RAW_DATA_DIR / "goodreads_reviews_romance.json.gz",
    "interactions": RAW_DATA_DIR / "goodreads_interactions_romance.json.gz",
    "genres": RAW_DATA_DIR / "goodreads_book_genres_initial.json.gz",
    "authors": RAW_DATA_DIR / "goodreads_book_authors.json.gz",
    "series": RAW_DATA_DIR / "goodreads_book_series.json.gz",
    "works": RAW_DATA_DIR / "goodreads_book_works.json.gz",
    "reviews_dedup": RAW_DATA_DIR / "goodreads_reviews_dedup.json.gz",
    "reviews_spoiler": RAW_DATA_DIR / "goodreads_reviews_spoiler.json.gz"
}

# Output file paths
OUTPUT_FILES = {
    "json_structure_report": OUTPUT_DIR / "reports" / "json_structure_analysis.html",
    "data_quality_report": OUTPUT_DIR / "reports" / "data_quality_report.html",
    "exploration_summary": OUTPUT_DIR / "reports" / "exploration_summary.txt"
}

# Basic data processing settings
DATA_PROCESSING = {
    "chunk_size": 1000,    # Conservative chunk size for initial exploration
    "max_workers": 2,      # Conservative parallel processing
    "sample_size": None,   # Will be determined during exploration
}

# Logging settings
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "romance_research.log",
    "max_file_size": "10MB",
    "backup_count": 5,
}

# Development and debugging
DEBUG = {
    "verbose": True,
    "save_intermediate": True,
    "test_mode": False,  # Set to True for testing with small samples
}

# TODO: Add analysis-specific configurations after data exploration
# - Genre classification keywords (after seeing actual data)
# - Text preprocessing parameters (after examining text content)
# - Quality thresholds (after understanding data distribution)
# - Model parameters (after determining analysis approach)