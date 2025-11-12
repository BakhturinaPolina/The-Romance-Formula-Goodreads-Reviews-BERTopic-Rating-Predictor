# Romance Novel NLP Research Project

## Overview

This research project explores the relationship between thematic characteristics of romance novels and their popularity among readers using Natural Language Processing (NLP) techniques. The study focuses on analyzing multiple forms of reader engagement to understand how thematic elements influence reader interest, satisfaction, and overall reception.

## Research Objectives

### Primary Research Question
How do thematic characteristics of romance novels relate to their popularity among readers?

### Specific Objectives
1. **Topic Modeling**: Extract themes from romance novels across different subgenres
2. **Review Analysis**: Analyze reader reviews to identify key themes and preferences
3. **Correlation Analysis**: Examine relationships between book themes and popularity metrics
4. **Comparative Analysis**: Compare author-intended themes with reader-perceived themes

## Dataset

The project uses Goodreads metadata including:
- **Books**: 348MB of romance book metadata
- **Reviews**: 1.2GB of romance-specific reviews
- **Interactions**: 2.1GB of user-book interactions
- **Genres**: 23MB of genre classifications
- **Authors**: 17MB of author information
- **Series**: 27MB of series data
- **Works**: 72MB of work-level metadata

### Data Processing Approach

The dataset uses **work-level aggregation** to handle multiple editions of the same book:

- **Individual Edition Fields**: Removed from final dataset to avoid confusion
  - `average_rating` (individual edition rating)
  - `ratings_count` (individual edition ratings count)
  - `text_reviews_count` (individual edition reviews count)

- **Work-Level Aggregated Fields**: Used for all analysis
  - `average_rating_weighted_mean` (weighted average across all editions)
  - `ratings_count_sum` (total ratings across all editions)
  - `text_reviews_count_sum` (total reviews across all editions)

This approach ensures fair comparison between books regardless of how many editions they have.

## Current Project Status

### ✅ **Completed**
- **Data Exploration**: Comprehensive understanding of all data sources
- **Data Processing**: Clean, analysis-ready datasets created
- **Data Cleaning**: Titles normalized, descriptions cleaned, series structured
- **Quality Validation**: Data quality assessed and documented
- **Data Quality Pipeline**: 6-step pipeline implemented and validated
- **Complete Data Cleaning Pipeline**: Full pipeline from raw data to final analysis
- **NLP Text Preprocessing**: HTML cleaning, text normalization, genre categorization
- **EDA Analysis**: Comprehensive exploratory data analysis with statistical insights

### 🔄 **In Progress**
- **Data Audit**: Statistical analysis and data exploration pipeline
- **Shelf Normalization**: Processing messy shelf tags into normalized forms
- **Review Extraction**: Extracting English reviews from Goodreads dataset (~9-10 hours processing time)

## Project Structure

### Current Clean Repository Structure
```
romance-novel-nlp-research/
├── src/                                          # Source code modules
│   ├── csv_building/                             # CSV generation and data processing
│   │   ├── final_csv_builder.py                  # Main CSV builder with work-level aggregation
│   │   └── run_builder.py                        # Builder runner script
│   ├── data_quality/                             # Complete 6-step data quality pipeline
│   │   ├── pipeline_runner.py                    # Main pipeline runner
│   │   ├── step1_missing_values_cleaning.py      # Step 1: Missing Values Cleaning
│   │   ├── step2_duplicate_detection.py          # Step 2: Duplicate Detection
│   │   ├── step3_data_type_validation.py         # Step 3: Data Type Validation
│   │   ├── step4_outlier_detection.py            # Step 4: Outlier Detection
│   │   ├── step4_outlier_treatment.py            # Step 4: Outlier Treatment
│   │   ├── step5_data_type_optimization.py       # Step 5: Data Type Optimization
│   │   ├── step6_final_quality_validation.py     # Step 6: Final Quality Validation
│   │   ├── comprehensive_data_analysis.py        # Comprehensive data analysis
│   │   └── comprehensive_data_cleaner.py         # Comprehensive data cleaning
│   ├── nlp_preprocessing/                        # NLP text preprocessing
│   │   ├── text_preprocessor.py                  # Main text preprocessor
│   │   ├── run_preprocessor.py                   # Preprocessor runner
│   │   └── test_preprocessor.py                  # Preprocessor tests
│   ├── data_audit/                               # Statistical analysis and data exploration
│   │   ├── core/                                 # Core audit functionality
│   │   │   └── data_auditor.py                   # Main audit script with CSN methodology
│   │   ├── parsing/                              # Data parsing utilities
│   │   │   └── list_parser.py                    # List parsing script
│   │   ├── notebooks/                            # Interactive analysis notebooks
│   │   └── utils/                                # Utility scripts
│   ├── shelf_normalization/                      # Shelf tag normalization pipeline
│   │   ├── core/                                 # Core normalization logic
│   │   │   └── shelf_normalize.py                # Main pipeline script
│   │   ├── bridge/                               # Integration with other pipeline steps
│   │   │   └── bridge_audit_normalize.py         # Bridge Step 1 → Step 2
│   │   └── diagnostics/                          # Quality assurance and validation
│   │       ├── diagnostics_explore.py            # Deep-dive diagnostics
│   │       └── validate_bridge.py                # Bridge output validation
│   └── review_extraction/                        # Review extraction and processing
│       ├── extract_reviews.py                    # Main extraction script
│       ├── monitor_extraction.py                 # Real-time monitoring
│       ├── estimate_time.py                      # Time estimation utility
│       ├── review_dataset.py                     # Dataset review utility
│       └── README.md                             # Module documentation
├── data/
│   ├── raw/                                      # Original Goodreads JSON files (9 files)
│   ├── intermediate/                              # Temporary processing outputs
│   └── processed/                                # Current datasets (9 files)
├── organized_outputs/                            # All organized outputs
│   ├── datasets/                                 # All dataset versions
│   ├── logs/                                     # All pipeline execution logs
│   ├── reports/                                  # All analysis reports
│   └── visualizations/                           # Publication-ready plots
└── archive/                                      # Archived development history
```

## Source Code Modules

### CSV Building (`src/csv_building/`)
**Purpose**: Create clean, analysis-ready romance novel datasets from raw Goodreads data

**Key Features**:
- **Work-Level Aggregation**: Handles multiple editions efficiently
- **High Performance**: Processes full dataset in ~15 minutes
- **Data Quality**: Comprehensive validation and reporting
- **Clean Architecture**: Single responsibility, well-documented

**Main Scripts**:
- `final_csv_builder.py`: Main CSV builder with enhanced null handling
- `run_builder.py`: Interactive builder runner with sample/full dataset options

### Data Quality (`src/data_quality/`)
**Purpose**: Implement comprehensive data quality assurance pipeline

**Key Features**:
- **6-Step Pipeline**: Complete data cleaning and validation workflow
- **Quality Gates**: Automated quality threshold validation
- **Data Type Optimization**: Memory efficiency improvements
- **Comprehensive Reporting**: Detailed quality metrics and certification

**Main Scripts**:
- `pipeline_runner.py`: Executes all 6 steps in sequence
- `step1_missing_values_cleaning.py`: Missing value analysis and treatment
- `step2_duplicate_detection.py`: Duplicate detection and resolution
- `step3_data_type_validation.py`: Data type validation and optimization
- `step4_outlier_detection.py`: Statistical outlier detection
- `step4_outlier_treatment.py`: Outlier treatment strategies
- `step5_data_type_optimization.py`: Memory optimization and persistence
- `step6_final_quality_validation.py`: Final quality certification

### NLP Preprocessing (`src/nlp_preprocessing/`)
**Purpose**: Text cleaning and normalization for NLP analysis

**Key Features**:
- **HTML Cleaning**: Remove HTML tags and entities
- **Text Normalization**: Standardize text formatting
- **Genre Categorization**: Map genres to standard categories
- **Shelf Standardization**: Normalize popular shelf tags

**Main Scripts**:
- `text_preprocessor.py`: Main text preprocessing pipeline
- `run_preprocessor.py`: Preprocessor runner script
- `test_preprocessor.py`: Testing and validation utilities

### Data Audit (`src/data_audit/`)
**Purpose**: Statistical analysis and data exploration using rigorous methodologies

**Key Features**:
- **Schema Validation**: Comprehensive column presence and data type validation
- **Heavy-Tail Analysis**: Clauset-Shalizi-Newman (2009) power-law fitting methodology
- **Overdispersion Testing**: Dean-Lawless and Cameron-Trivedi formal statistical tests
- **List Parsing**: Robust parsing of list-like fields with fallback strategies
- **Interactive Analysis**: Jupyter notebooks for exploratory data analysis
- **Statistical Reporting**: HTML reports with comprehensive analysis results

**Main Scripts**:
- `core/data_auditor.py`: Main audit script implementing CSN methodology
- `parsing/list_parser.py`: Multi-strategy parsing of list-like fields
- `utils/diff_bridge_runs.py`: Bridge run comparison utilities

**Statistical Methodologies**:
- **Heavy-Tail Analysis**: Power-law distribution detection and characterization
- **Overdispersion Tests**: Formal statistical tests for count data
- **Schema Validation**: Expected column checking and data type analysis

### Shelf Normalization (`src/shelf_normalization/`)
**Purpose**: Transform messy shelf tags into normalized, canonical forms

**Key Features**:
- **Canonicalization**: Deterministic normalization of shelf strings
- **Segmentation**: Conservative CamelCase and concatenation splitting
- **Alias Detection**: Multi-metric identification of potential shelf aliases
- **Non-Content Filtering**: Exclusion of non-content shelf categories
- **Quality Assurance**: Comprehensive validation and diagnostics

**Main Scripts**:
- `core/shelf_normalize.py`: Main normalization pipeline
- `bridge/bridge_audit_normalize.py`: Integration with audit outputs
- `diagnostics/diagnostics_explore.py`: Deep-dive quality analysis
- `diagnostics/validate_bridge.py`: Bridge output validation

**Algorithm Details**:
- **Canonicalization**: Unicode normalization, separator standardization, case folding
- **Segmentation**: CamelCase detection with guard conditions
- **Alias Detection**: Jaro-Winkler similarity, edit distance, character n-grams

### Review Extraction (`src/review_extraction/`)
**Purpose**: Extract English-language reviews for romance books from Goodreads dataset

**Key Features**:
- **Book ID Filtering**: Matches reviews to books from main dataset
- **Language Detection**: Automatically detects and filters English reviews
- **Real-Time Monitoring**: Live progress tracking with process status
- **Time Estimation**: Estimates remaining processing time
- **Progress Logging**: Detailed logging every 5,000 reviews

**Main Scripts**:
- `extract_reviews.py`: Main extraction script (processes ~3.6M reviews)
- `monitor_extraction.py`: Real-time monitoring with process stats
- `estimate_time.py`: Calculate remaining time estimates
- `review_dataset.py`: Utility to review dataset structure
- `monitor.sh`: Quick wrapper for monitoring

**Processing Details**:
- **Input**: `goodreads_reviews_romance.json.gz` (~1.2GB, ~3.6M reviews)
- **Output**: `romance_reviews_english.csv` (review_id, review_text, rating, book_id)
- **Processing Rate**: ~100-110 reviews/second
- **Total Time**: ~9-10 hours for full dataset
- **Language Detection**: Uses `langdetect` library

**Usage**:
```bash
# Run extraction
python3 src/review_extraction/extract_reviews.py

# Monitor progress
./src/review_extraction/monitor.sh [PID]

# Estimate time remaining
python3 src/review_extraction/estimate_time.py
```

### Dataset Versions Available
- **Complete Dataset**: All 30 columns with full metadata
- **Core Research Dataset**: 23 essential columns for efficient analysis
- **False Duplicates Dataset**: 21,105 records with similar titles by different authors
- **Step-by-Step Datasets**: 7 datasets showing pipeline progression
- **Specialized Versions**: 9 different dataset versions for specific research needs

## Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment with required packages
- Access to processed datasets

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd romance-novel-nlp-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Analysis

#### CSV Building
```bash
# Run the CSV builder
cd src/csv_building
python run_builder.py

# Choose processing mode:
# 1. Test with sample (recommended for first run)
# 2. Process full dataset
```

#### Data Quality Pipeline
```bash
# Run complete 6-step pipeline
cd src/data_quality
python pipeline_runner.py

# Or run individual steps:
python step1_missing_values_cleaning.py
python step2_duplicate_detection.py
python step3_data_type_validation.py
python step4_outlier_detection.py
python step4_outlier_treatment.py
python step5_data_type_optimization.py
python step6_final_quality_validation.py
```

#### NLP Text Preprocessing
```bash
# Run text preprocessing
cd src/nlp_preprocessing
python run_preprocessor.py

# Test preprocessing on sample data
python test_preprocessor.py
```

#### Data Audit
```bash
# Run complete audit pipeline
cd src/data_audit
make audit

# Or run individual components:
python core/data_auditor.py --data-path ../../data/processed/romance_books_main_final.csv
python parsing/list_parser.py --data-path ../../data/processed/romance_books_main_final.csv
```

#### Shelf Normalization
```bash
# Run complete normalization pipeline
cd src/shelf_normalization
make pipeline

# Or run individual steps:
make normalize    # Step 1: Normalize shelves
make bridge       # Step 2: Bridge with parsed data
make diagnostics  # Step 3: Run diagnostics
make validate     # Step 4: Validate outputs
```

#### Review Extraction
```bash
# Run extraction (will take ~9-10 hours)
cd src/review_extraction
python3 extract_reviews.py

# Monitor progress in real-time
./monitor.sh [PID]

# Estimate remaining time
python3 estimate_time.py

# Review dataset structure
python3 review_dataset.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Data Source**: UCSD Goodreads Book Graph

## Quick non-Docker usage (member API)

```bash
# 1. Setup
cd ~/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt requests pandas python-dotenv

# 2. Auth
export ANNAS_SECRET_KEY="<your_member_secret>"

# 3. Batch-download when MD5s already exist
python src/book_download/batch_md5_download.py \
  --csv data/processed/test_books_with_md5_final.csv \
  --out organized_outputs/epub_downloads

# 4. Batch-download without MD5s (title/author only)
#    (search_md5() falls back to member-API search)
python - <<'PY'
import csv, pathlib, os
from src.book_download.anna_api_client import AnnaAPIClient
api = AnnaAPIClient()
root = pathlib.Path('organized_outputs/epub_downloads'); root.mkdir(parents=True, exist_ok=True)
with open('data/processed/sample_books_for_download.csv') as f:
    for row in csv.DictReader(f):
        md5 = (row.get('md5') or '').strip()
        if not md5:
            md5 = api.search_md5(row['title'], row['author_name'])
        if not md5:
            print('no match:', row['title']); continue
        api.download_book(md5, download_dir=str(root))
PY
```

This path avoids Elasticsearch/MariaDB and heavy services – ideal for quick experiments.