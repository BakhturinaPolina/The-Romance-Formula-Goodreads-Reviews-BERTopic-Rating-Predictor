# External Data Extraction Module

This module provides tools for extracting structured data from external datasets, specifically designed for romance book data extraction from Hugging Face datasets.

## Overview

The `extract_romance_books.py` tool loads the `AlekseyKorshuk/romance-books` dataset from Hugging Face and extracts `(author, title)` pairs using layered heuristics with confidence scoring.

## Features

- **Layered Extraction Heuristics**: Multiple extraction methods with fallback strategies
- **Confidence Scoring**: Each extraction includes a confidence score (0.0-1.0)
- **Multiple Output Formats**: CSV and JSONL support
- **Optional spaCy Integration**: Enhanced author detection using Named Entity Recognition
- **URL Fallback**: Extracts titles from URLs when text headers are noisy
- **Comprehensive Reporting**: Summary statistics and low-confidence examples

## Extraction Methods (in order of preference)

1. **Title by Author**: `"Book Title by Author Name"`
2. **Dash Split**: `"Author Name — Book Title"` or `"Book Title — Author Name"`
3. **Quoted Title**: `"Book Title" by Author Name`
4. **Leading Name**: `"Author Name Book Title"`
5. **URL Fallback**: Extract title from URL path
6. **spaCy NER**: Use Named Entity Recognition for author detection (optional)

## Installation

The tool requires the `datasets` library, which is already included in the project requirements:

```bash
pip install datasets
```

For enhanced author detection with spaCy (optional):

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```bash
# Extract to CSV with 1000 samples
python -m src.external_data_extraction.extract_romance_books --out romance_pairs.csv --limit 1000

# Extract all data to JSONL
python -m src.external_data_extraction.extract_romance_books --out romance_pairs.jsonl

# Use spaCy for enhanced author detection
python -m src.external_data_extraction.extract_romance_books --out romance_pairs.csv --use-spacy
```

### Command Line Options

- `--split`: Dataset split to use (default: "train")
- `--limit`: Maximum number of rows to process (0 = all, default: 0)
- `--out`: Output file path (required)
- `--fmt`: Force output format ("csv" or "jsonl", auto-detected by extension if not specified)
- `--use-spacy`: Enable spaCy PERSON NER fallback for author detection

### Output Format

The tool outputs the following fields:

- `idx`: Row index in the dataset
- `author`: Extracted author name (if found)
- `title`: Extracted book title (if found)
- `confidence`: Confidence score (0.0-1.0)
- `method`: Extraction method used
- `source_url`: Original URL from the dataset
- `raw_header`: Raw header text that was processed

### Example Output

```csv
idx,author,title,confidence,method,source_url,raw_header
0,"Jane Smith","The Romance Novel",0.92,title_by_author,https://example.com/book,"The Romance Novel by Jane Smith"
1,"John Doe","Love Story",0.85,dash_author_left,https://example.com/love,"John Doe — Love Story"
```

## Confidence Scoring

- **0.9+**: High confidence (e.g., "Title by Author" pattern)
- **0.8-0.9**: Good confidence (e.g., dash-separated patterns)
- **0.6-0.8**: Moderate confidence (e.g., leading name patterns)
- **0.4-0.6**: Low confidence (e.g., URL fallback, partial matches)
- **<0.4**: Very low confidence (e.g., header snippets)

## Performance

- Processes ~1000 rows per second on typical hardware
- Memory usage scales linearly with dataset size
- Progress reporting every 1000 rows for large datasets

## Error Handling

- Gracefully handles missing or malformed data
- Provides detailed error reporting for low-confidence extractions
- Continues processing even if individual rows fail

## Integration with Project

This tool is designed to work with the existing romance novel NLP research project structure:

- Outputs are saved to the specified path (typically in `data/processed/` or `organized_outputs/`)
- Follows the project's coding patterns and error handling conventions
- Can be integrated into larger data processing pipelines

## Examples

### Extract a small sample for testing

```bash
python -m src.external_data_extraction.extract_romance_books --out data/processed/romance_sample.csv --limit 100
```

### Full dataset extraction with spaCy

```bash
python -m src.external_data_extraction.extract_romance_books --out organized_outputs/datasets/romance_full.jsonl --use-spacy
```

### Process specific dataset split

```bash
python -m src.external_data_extraction.extract_romance_books --split validation --out data/processed/romance_validation.csv
```

## Troubleshooting

### Common Issues

1. **Hugging Face authentication**: You may need to accept terms for gated datasets
2. **Memory issues**: Use `--limit` to process smaller batches
3. **spaCy model missing**: Install with `python -m spacy download en_core_web_sm`

### Low Confidence Results

The tool reports low-confidence extractions at the end. These typically indicate:
- Unusual header formats not covered by heuristics
- Missing or corrupted data
- Non-English content
- Very short or noisy headers

Review these cases to improve extraction heuristics if needed.
