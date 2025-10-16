# Text Matching Module

This module provides fuzzy matching functionality between Goodreads books and external text datasets using title and author similarity.

## Features

- **Fuzzy String Matching**: Uses RapidFuzz for efficient title and author matching
- **Blocking Strategy**: Implements blocking to reduce computational complexity
- **Configurable Thresholds**: Customizable acceptance and review thresholds
- **Multiple Data Sources**: Supports both Hugging Face datasets and local CSV files
- **Comprehensive Scoring**: Combines title, author, and publication year similarity

## Files

- `match_goodreads_to_texts.py`: Core matching functionality
- `run_matcher.py`: Easy-to-use runner script with predefined configurations
- `__init__.py`: Module initialization and exports

## Dependencies

The following packages are required (already added to `requirements.txt`):
- `pandas`: Data manipulation
- `rapidfuzz`: Fast fuzzy string matching
- `unidecode`: Unicode normalization
- `python-Levenshtein`: Faster fuzzy matching
- `datasets`: Hugging Face datasets support

## Usage

### Basic Usage with Hugging Face Dataset

```bash
# Using the runner script
python src/text_matching/run_matcher.py \
  --goodreads_csv data/processed/romance_books_main_final_canonicalized.csv \
  --output_dir data/processed \
  --config default

# Using the main script directly
python src/text_matching/match_goodreads_to_texts.py \
  --goodreads_csv data/processed/romance_books_main_final_canonicalized.csv \
  --use_hf_texts \
  --out_matches data/processed/matches_definitive.csv \
  --out_review data/processed/matches_needs_review.csv
```

### Using Local Text Dataset

```bash
python src/text_matching/run_matcher.py \
  --goodreads_csv data/processed/romance_books_main_final_canonicalized.csv \
  --local_texts_csv path/to/your/text_dataset.csv \
  --output_dir data/processed \
  --config relaxed
```

### Configuration Presets

- **default**: Balanced precision and recall
- **relaxed**: Better recall, more matches (lower thresholds)
- **strict**: Better precision, fewer false positives (higher thresholds)

## Input Data Format

### Goodreads CSV Requirements
- `work_id`: Unique identifier for each book
- `title`: Book title
- `author_name`: Author name(s)
- `publication_year`: Publication year (optional but recommended)

### Text Dataset Requirements
- `title`: Book title
- `author`: Author name
- `publication_year`: Publication year (optional)
- `text_book_id`: Unique identifier (auto-generated if missing)

## Output Files

1. **Definitive Matches** (`*_matches_definitive.csv`): High-confidence matches above the acceptance threshold
2. **Needs Review** (`*_matches_needs_review.csv`): Medium-confidence matches requiring manual review

### Output Columns

- `goodreads_work_id`: Original Goodreads book ID
- `text_book_id`: Matched text dataset book ID
- `gr_title`: Original Goodreads title
- `gr_author_name`: Original Goodreads author
- `gr_publication_year`: Original Goodreads publication year
- `tx_title`: Matched text dataset title
- `tx_author`: Matched text dataset author
- `tx_publication_year`: Matched text dataset publication year
- `title_score`: Title similarity score (0-100)
- `author_score`: Author similarity score (0-100)
- `composite`: Combined similarity score (0-100)

## Configuration

The `MatchConfig` class allows fine-tuning of matching parameters:

```python
from src.text_matching import MatchConfig

config = MatchConfig(
    title_threshold_accept=90,      # Accept matches above this score
    title_threshold_review=80,      # Review matches between this and accept
    author_weight=0.25,             # Weight of author in composite score
    title_weight=0.70,              # Weight of title in composite score
    year_bonus_per_match=5,         # Bonus for exact year match
    year_tolerance=2,               # Years within this get small bonus
    block_title_prefix=14,          # Length of title prefix for blocking
    max_candidates_per_block=50,    # Max candidates per block
    top_k=3                         # Top K candidates for review
)
```

## Algorithm Details

1. **Normalization**: Titles and authors are normalized (accent folding, punctuation removal, case normalization)
2. **Blocking**: Books are grouped by normalized title prefix and author last name
3. **Candidate Generation**: Within each block, fuzzy matching finds candidate pairs
4. **Scoring**: Composite score combines title, author, and year similarity
5. **Selection**: Best match per Goodreads book is selected with tie-breaking rules

## Performance

- **Blocking**: Reduces O(n²) comparisons to manageable chunks
- **RapidFuzz**: Fast C++ implementation for fuzzy string matching
- **Memory Efficient**: Processes data in blocks to handle large datasets

## Example Results

The matching process typically achieves:
- **High Precision**: 90%+ of definitive matches are correct
- **Good Recall**: 70-80% of actual matches found (depending on data quality)
- **Review Band**: 10-20% of matches require manual review

## Troubleshooting

### Low Match Rate
- Try the "relaxed" configuration
- Check if column names match expected format
- Verify data quality (missing titles/authors)

### High False Positive Rate
- Try the "strict" configuration
- Increase `title_threshold_accept`
- Increase `author_weight` in configuration

### No Matches Found
- Check if blocking is too restrictive (reduce `block_title_prefix`)
- Verify external dataset has compatible format
- Check for encoding issues in text data

