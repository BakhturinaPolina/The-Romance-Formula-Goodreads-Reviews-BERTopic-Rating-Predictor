# Stage 05: Prepare Reviews Corpus for BERTopic

## Purpose

This stage prepares the review dataset for BERTopic topic modeling by converting review-level data into sentence-level data suitable for BERTopic analysis.

## ⚠️ Virtual Environment Requirement

**ALL scripts MUST be run using the virtual environment at `romance-novel-nlp-research/.venv`.**

All Python scripts automatically verify they're using the correct venv. Always use:

```bash
romance-novel-nlp-research/.venv/bin/python3 script.py
```

For full project-wide rules, see `.cursor/rules/venv-requirement.mdc`.

## Structure

```
05_prepare_reviews_corpus_for_BERTopic/
├── core/                          # Main preparation logic
│   ├── __init__.py
│   └── prepare_bertopic_input.py  # Main preparation script
├── monitoring/                    # Monitoring and status checking
│   ├── __init__.py
│   ├── monitor_preparation.py    # Real-time monitoring
│   └── check_status.py           # Quick status check
├── scripts/                       # Execution and test scripts
│   ├── __init__.py
│   ├── run_preparation.py        # Preparation wrapper (uses venv)
│   ├── run_test.py               # Test runner (uses venv)
│   └── venv_utils.py             # Virtual environment utilities
├── utils/                         # Utility modules
│   ├── data_loading.py           # Data loading utilities
│   └── checks_coverage.py        # Coverage checking utilities
├── notebooks/                      # Jupyter notebooks
│   └── eda_reviews.ipynb        # Exploratory data analysis
├── __init__.py                    # Module exports
├── README.md                      # This file
└── README_SCIENTIFIC.md           # Scientific documentation
```

## Input Files

- `data/processed/romance_subdataset_6000.csv` - Books dataset (6,000 books)
- `data/processed/romance_reviews_english.csv` - English reviews dataset
- `data/processed/romance_books_main_final.csv` - Main dataset (for book_id mapping)

## Output Files

- `data/processed/review_sentences_for_bertopic.parquet` - Sentence-level dataset for BERTopic
- `data/processed/review_sentences_temp/chunk_*.parquet` - Temporary chunk files (during processing)
- `/tmp/bertopic_prep_monitor.log` - Processing log file

## How to Run

### Running the Preparation

**Recommended (uses venv automatically):**
```bash
cd src/05_prepare_reviews_corpus_for_BERTopic
romance-novel-nlp-research/.venv/bin/python3 scripts/run_preparation.py
```

**Or run directly:**
```bash
romance-novel-nlp-research/.venv/bin/python3 core/prepare_bertopic_input.py
```

### Monitoring Progress

**Option 1: Continuous monitoring (in separate terminal)**
```bash
romance-novel-nlp-research/.venv/bin/python3 monitoring/monitor_preparation.py
```

**Option 2: Quick status check**
```bash
romance-novel-nlp-research/.venv/bin/python3 monitoring/check_status.py
```

### Running Tests

```bash
romance-novel-nlp-research/.venv/bin/python3 scripts/run_test.py
```

## Process Details

### Sentence Splitting
- Uses spaCy `en_core_web_sm` model for sentence segmentation
- Processes reviews in chunks of 20,000 reviews
- Saves incremental progress to chunk files

### Data Mapping

**Important**: The books dataset uses `work_id` as the identifier, while reviews use `book_id`. In Goodreads:
- **work_id**: Represents a work (abstract book concept)
- **book_id**: Represents a specific edition of that work

A single work can have multiple editions (paperback, hardcover, ebook, etc.), each with its own `book_id`.

**Solution**: The mapping is automatically handled by `utils/data_loading.py`:
- `load_book_id_to_work_id_mapping()` creates a reverse mapping from the main dataset
- `load_joined_reviews()` uses this mapping to convert `book_id` → `work_id` before joining
- Only reviews for books in the 6,000 book dataset are included

**Results**:
- ✅ **969,675 reviews** successfully joined
- ✅ **5,998 out of 6,000 books (99.97%)** have reviews
- ✅ All reviews mapped correctly using `book_id_list_en` from main dataset

**Technical Details**:
1. Main dataset contains `book_id_list_en` column with Python list strings like `['3462', '6338758', ...]`
2. Each `work_id` maps to multiple `book_id` values (editions)
3. A reverse dictionary `{book_id: work_id}` is created for all editions
4. Reviews are mapped using this dictionary before joining on `work_id`

### Progress Tracking
- Progress is logged to `/tmp/bertopic_prep_monitor.log`
- Monitor script displays:
  - Process status (running/stopped)
  - CPU and memory usage
  - Latest progress lines
  - Key statistics (chunk progress, processing rate, ETA)

## Dependencies

- pandas
- spacy (with `en_core_web_sm` model)
- pyarrow (for parquet files)
- Standard library: pathlib, logging, typing

## Troubleshooting

### Process Not Running
- Check if process crashed: `tail -50 /tmp/bertopic_prep_monitor.log`
- Verify input files exist
- Check disk space

### Slow Processing
- Sentence splitting is CPU-intensive
- Processing rate: ~100-200 reviews/sec (depends on review length)
- Full dataset (1.2M reviews) takes several hours

### Memory Issues
- Process uses ~2-4GB RAM
- Chunk files are saved incrementally to prevent data loss
- If process crashes, chunk files can be recovered from `data/processed/review_sentences_temp/`
