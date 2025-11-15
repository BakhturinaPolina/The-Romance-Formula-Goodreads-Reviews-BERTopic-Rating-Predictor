# Stage 01: Data Integration

## Purpose

This stage handles CSV building, data integration from multiple sources, external data extraction, and representative subdataset sampling. It transforms raw Goodreads JSON files into structured CSV datasets suitable for downstream analysis.

## Input Files

- `data/raw/goodreads_*.json.gz` - Raw Goodreads JSON files (read-only)
- External datasets (if using external extraction, e.g., Hugging Face romance-books dataset)

## Output Files

- `data/processed/*.csv` - Integrated and processed datasets
- `data/processed/*_subdataset_*.csv` - Representative samples (e.g., `romance_subdataset_6000.csv`)

## File Descriptions

### Core CSV Building (`csv_building/`)

#### `csv_building/final_csv_builder.py`
**Purpose**: Main CSV builder class that processes raw Goodreads JSON files and creates structured CSV datasets.

**Key Features**:
- Aggregates multiple editions into work-level records
- Filters books by language (English only by default)
- Handles null values and data type conversions
- Computes aggregated statistics (median pages, weighted ratings, etc.)
- Strips brackets from titles and normalizes text
- Comprehensive logging and data quality tracking

**Main Class**: `OptimizedFinalCSVBuilder`
- `build_final_csv_optimized()`: Main method to build CSV from raw JSON files
- Supports both full dataset processing and sample processing for testing

#### `csv_building/run_builder.py`
**Purpose**: Interactive command-line runner for the CSV builder.

**Features**:
- Provides user-friendly CLI interface
- Allows choosing between sample processing (for testing) or full dataset processing
- Displays data quality summary after processing
- Shows dataset statistics (shape, publication year range, null counts, etc.)

**Usage**:
```bash
python csv_building/run_builder.py
# Then choose: 1 for sample, 2 for full dataset
```

### Subdataset Sampling (`subdataset_sampling/`)

#### `subdataset_sampling/create_subdataset_6000.py`
**Purpose**: Creates a representative 6,000 book subdataset with balanced representation across popularity tiers.

**Key Features**:
- Stratified sampling across three popularity tiers (thrash, mid, top)
- Preserves demographic characteristics (genre groups, page count quartiles, publication years)
- Removes comics/graphic novels
- Applies genre canonicalization (merges historical fiction/historical romance, etc.)
- Uses quantile-based tier boundaries (25th and 75th percentiles of weighted average rating)
- Ensures equal representation: 2,000 books per tier

**Main Function**: `create_subdataset_6000(input_csv_path, output_csv_path)`

#### `subdataset_sampling/run_subdataset_sampling.py`
**Purpose**: Command-line runner for subdataset sampling.

**Features**:
- Simple interface to create the 6,000 book subdataset
- Accepts optional command-line arguments for input/output paths
- Displays tier distribution after sampling
- Default paths: `data/processed/romance_books_main_final.csv` → `data/processed/romance_subdataset_6000.csv`

**Usage**:
```bash
python subdataset_sampling/run_subdataset_sampling.py [input_csv] [output_csv]
```

### External Data Extraction (`external_data_extraction/`)

#### `external_data_extraction/extract_romance_books.py`
**Purpose**: Extracts author/title pairs from the AlekseyKorshuk/romance-books Hugging Face dataset using layered heuristics.

**Key Features**:
- Loads dataset from Hugging Face using `datasets` library
- Multiple extraction strategies with confidence scoring:
  - "by" clause parsing (e.g., "Title by Author")
  - Dash-separated parsing (e.g., "Title - Author")
  - Quoted title extraction
  - Leading name detection
  - URL fallback parsing
- Optional spaCy integration for enhanced extraction
- Outputs CSV or JSONL format
- Includes confidence scores and extraction method metadata

**Usage**:
```bash
python external_data_extraction/extract_romance_books.py --out output.csv --limit 1000
python external_data_extraction/extract_romance_books.py --out output.jsonl --use-spacy
```

#### `external_data_extraction/bookrix_extractor.py`
**Purpose**: Specialized extractor for BookRix URLs that achieves high accuracy (99.9%) by parsing URL patterns.

**Key Features**:
- Designed for BookRix URL pattern: `https://www.bookrix.com/_ebook-<author-slug>-<title-slug>/`
- Extracts author and title from URL slugs
- Handles URL encoding and apostrophe patterns (e.g., `039` → `'`)
- Humanizes slugs with proper title casing
- Heuristic-based author/title boundary detection
- Can process entire Hugging Face dataset splits

**Main Functions**:
- `extract_from_url(url)`: Extracts author/title from single URL
- `process_dataset(split, output_dir)`: Processes entire dataset split

#### `external_data_extraction/book_matcher.py`
**Purpose**: Matches books from external datasets (like BookRix) to existing Goodreads metadata using exact and fuzzy matching techniques.

**Key Features**:
- Exact matching on author+title combinations
- Fuzzy matching using RapidFuzz library:
  - Author similarity matching
  - Title similarity matching
  - Combined similarity scoring
- Configurable similarity thresholds
- Weighted scoring (author weight vs. title weight)
- Preprocessing and normalization for better matching
- Returns match results with confidence scores and match types

**Main Class**: `BookMatcher`
- `match_all()`: Matches all books from external dataset
- `match_one()`: Matches single book
- Supports filtering by confidence threshold

### Module Initialization

#### `__init__.py`
**Purpose**: Module initialization file that exports the main subdataset sampling function.

**Exports**: `create_subdataset_6000`

## How to Run

### CSV Building

**Step 1**: Build main CSV from raw Goodreads JSON files
```bash
cd src/01_data_integration
python csv_building/run_builder.py
```

This will:
1. Prompt you to choose sample or full dataset processing
2. Process raw JSON files from `data/raw/`
3. Create aggregated work-level CSV in `data/processed/`
4. Display data quality summary

### Subdataset Sampling

**Step 2**: Create representative 6,000 book subdataset
```bash
cd src/01_data_integration
python subdataset_sampling/run_subdataset_sampling.py
```

Or with custom paths:
```bash
python subdataset_sampling/run_subdataset_sampling.py \
    data/processed/romance_books_main_final.csv \
    data/processed/romance_subdataset_6000.csv
```

### External Data Extraction

**Optional**: Extract romance books from external sources

```bash
# Extract from Hugging Face dataset
python external_data_extraction/extract_romance_books.py --out output.csv --limit 1000

# Extract with spaCy enhancement
python external_data_extraction/extract_romance_books.py --out output.jsonl --use-spacy

# Extract from BookRix URLs
python -c "from external_data_extraction.bookrix_extractor import process_dataset; process_dataset('train', 'data/processed')"
```

## Dependencies

### Core Dependencies
- `pandas` - Data manipulation and CSV handling
- `json` / `gzip` - Standard library for reading compressed JSON files
- `numpy` - Numerical operations
- `scipy` - Statistical functions (chi-square tests for sampling validation)

### External Data Extraction
- `datasets` - Hugging Face datasets library
- `rapidfuzz` - Fast fuzzy string matching (for book_matcher.py)
- `spacy` - Optional NLP library for enhanced extraction

## Data Processing Pipeline

### Typical Workflow

1. **Raw Data → Main CSV**
   ```
   data/raw/goodreads_*.json.gz 
   → run_builder.py 
   → data/processed/romance_books_main_final.csv
   ```

2. **Main CSV → Subdataset**
   ```
   data/processed/romance_books_main_final.csv 
   → run_subdataset_sampling.py 
   → data/processed/romance_subdataset_6000.csv
   ```

3. **External Data Integration** (optional)
   ```
   Hugging Face dataset 
   → extract_romance_books.py / bookrix_extractor.py 
   → book_matcher.py 
   → Matched records for enrichment
   ```

## Key Features

### Work-Level Aggregation
- Multiple editions of the same work are aggregated into a single record
- Computes median values for numeric fields (pages, ratings)
- Weighted averages for ratings based on review counts
- Preserves all unique metadata (genres, series info, etc.)

### Language Filtering
- Filters to English-language books only
- Uses regex pattern matching on language codes (eng, en, en-US, etc.)

### Representative Sampling
- Stratified sampling across popularity tiers
- Preserves demographic characteristics:
  - Genre groups (paranormal, historical, fantasy, mystery, young adult, other)
  - Page count quartiles
  - Publication year ranges
  - Series vs. standalone distribution

### Data Quality
- Enhanced null value handling
- Type validation and conversion
- Comprehensive logging of data transformations
- Quality metrics tracking

## Example Usage

### Complete Pipeline

```bash
# 1. Build main CSV (start with sample for testing)
cd src/01_data_integration
python csv_building/run_builder.py
# Choose option 1, enter sample size (e.g., 100)

# 2. Verify output
ls -lh data/processed/*.csv

# 3. Create subdataset (requires full main CSV)
python subdataset_sampling/run_subdataset_sampling.py

# 4. Verify subdataset
python -c "import pandas as pd; df = pd.read_csv('data/processed/romance_subdataset_6000.csv'); print(df['pop_tier'].value_counts())"
```

### External Data Integration

```bash
# Extract from external source
python external_data_extraction/extract_romance_books.py --out data/processed/external_romance.csv --limit 5000

# Match to Goodreads metadata
python -c "
from external_data_extraction.book_matcher import BookMatcher
matcher = BookMatcher(
    'data/processed/romance_books_main_final.csv',
    'data/processed/external_romance.csv'
)
matches = matcher.match_all(threshold=0.8)
print(f'Found {len(matches)} matches')
"
```

## Output Schema

### Main CSV Columns
- `work_id` - Unique work identifier
- `title` - Book title (brackets stripped)
- `author_name` - Primary author name
- `author_id` - Author identifier
- `publication_year` - Year of publication
- `num_pages_median` - Median page count across editions
- `average_rating_weighted_mean` - Weighted average rating
- `ratings_count_sum` - Total ratings across editions
- `genres_str` - Comma-separated genre list
- `series_id` - Series identifier (or 'stand_alone')
- `description` - Book description
- `language_code` - Language code (filtered to English)
- And more...

### Subdataset CSV
- Same schema as main CSV
- Additional column: `pop_tier` - Popularity tier ('thrash', 'mid', 'top')
- Exactly 6,000 records (2,000 per tier)
