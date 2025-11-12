# Anna's Archive Book Matcher

This module provides automated book matching between your romance novels dataset and Anna's Archive collections to extract MD5 hashes for batch downloading.

## Features

- **DuckDB-based Analysis**: High-performance SQL queries on Anna's Archive datasets
- **Multi-source Matching**: Searches across Elasticsearch, AAC, and MariaDB datasets
- **Fuzzy Matching**: Handles title/author variations and formatting differences
- **Batch Processing**: Processes large datasets efficiently
- **MD5 Extraction**: Automatically extracts download hashes for matched books

## Dataset Sources

Based on the [Anna's Archive Data Science Starter Kit](https://github.com/RArtutos/Data-science-starter-kit-Enhance/):

1. **Elasticsearch Dataset**: Main book collection with metadata
2. **AAC Dataset**: Additional book records and metadata
3. **MariaDB Dataset**: Archive metadata and file information

## Usage

```python
from anna_archive_matcher import BookMatcher

# Initialize matcher
matcher = BookMatcher()

# Load your romance books dataset
romance_books = matcher.load_romance_dataset()

# Find matches in Anna's Archive
matches = matcher.find_matches(romance_books)

# Extract MD5 hashes for download
md5_hashes = matcher.extract_md5_hashes(matches)
```

## Directory Structure

```
anna_archive_matcher/
├── data/
│   ├── elasticsearch/    # Original .gz files
│   ├── elasticsearchF/   # Processed .parquet files
│   ├── aac/             # AAC .zst files
│   ├── aacF/            # Processed AAC .parquet files
│   ├── mariadb/         # MariaDB .gz files
│   └── mariadbF/        # Processed MariaDB .parquet files
├── notebooks/           # Analysis notebooks
├── core/               # Core matching logic
└── utils/              # Utility functions
```
