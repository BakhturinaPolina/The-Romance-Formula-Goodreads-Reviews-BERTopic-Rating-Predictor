# Reviews Analysis Module

This module contains the complete pipeline for topic modeling on Goodreads reviews of romance novels.

## Directory Structure

```
reviews_analysis/
├── README.md                    # This file - overview of the module
├── __init__.py                  # Module exports
│
├── core/                        # Core utilities (optional - currently in root)
│   ├── data_loading.py          # Data loading and joining functions
│   └── checks_coverage.py       # Review coverage verification
│
├── notebooks/                   # Jupyter notebooks
│   └── eda_reviews.ipynb        # Exploratory data analysis
│
├── bertopic_preparation/        # Phase 4: Sentence-level dataset preparation
│   ├── README.md                # Detailed documentation
│   ├── prepare_bertopic_input.py # Main preparation script
│   ├── run_bertopic_prep.sh     # Execution script
│   ├── monitor_bertopic_prep.sh # Monitoring script
│   └── check_bertopic_status.sh # Status check script
│
├── BERTopic_OCTIS/              # Phase 5: Topic modeling pipeline
│   ├── README.md                # Detailed documentation
│   ├── bertopic_plus_octis.py   # Main modeling script
│   ├── optimizer.py             # Bayesian optimization
│   ├── load_raw_sentences.py    # Data loading for modeling
│   └── ...                      # Additional utilities
│
└── docs/                        # Documentation
    └── archive/                 # Archived planning documents
        ├── plan.md              # Original project plan
        ├── background.md        # Research background
        └── TASKS.md             # Task breakdown
```

## Pipeline Overview

The pipeline follows these phases:

1. **Data Loading** (`data_loading.py`)
   - Load books and reviews datasets
   - Join reviews to books using work_id mapping
   - Handle book_id → work_id conversion

2. **Coverage Verification** (`checks_coverage.py`)
   - Compute review counts per book
   - Generate coverage statistics
   - Identify books with missing reviews

3. **Exploratory Data Analysis** (`notebooks/eda_reviews.ipynb`)
   - Review count distributions
   - Review length analysis
   - Ratings analysis
   - Lexical patterns

4. **BERTopic Preparation** (`bertopic_preparation/`)
   - Split reviews into sentences using spaCy
   - Create sentence-level dataset
   - Clean and normalize text
   - Output: `data/processed/review_sentences_for_bertopic.parquet`

5. **Topic Modeling** (`BERTopic_OCTIS/`)
   - Train BERTopic models on sentence-level data
   - Optimize hyperparameters using Bayesian optimization
   - Evaluate with OCTIS metrics
   - Map topics back to reviews and books

## Quick Start

### 1. Load and Verify Data

```python
from reviews_analysis.data_loading import load_joined_reviews
from reviews_analysis.checks_coverage import compute_review_counts, summarize_coverage

# Load data
joined_df = load_joined_reviews()

# Check coverage
counts = compute_review_counts(joined_df)
summarize_coverage(counts)
```

### 2. Prepare BERTopic Input

```bash
cd /home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research
./src/reviews_analysis/bertopic_preparation/run_bertopic_prep.sh
```

See `bertopic_preparation/README.md` for detailed instructions.

### 3. Run Topic Modeling

```bash
cd /home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research
python src/reviews_analysis/BERTopic_OCTIS/bertopic_plus_octis.py
```

See `BERTopic_OCTIS/README.md` for detailed instructions.

## Key Files

- **`data_loading.py`**: Core data loading utilities
- **`checks_coverage.py`**: Coverage verification functions
- **`notebooks/eda_reviews.ipynb`**: Exploratory data analysis
- **`bertopic_preparation/prepare_bertopic_input.py`**: Sentence splitting and dataset creation
- **`BERTopic_OCTIS/bertopic_plus_octis.py`**: Main topic modeling script

## Data Flow

```
Books CSV → data_loading.py → Joined DataFrame
Reviews CSV ↗

Joined DataFrame → checks_coverage.py → Coverage Report
                → eda_reviews.ipynb → EDA Visualizations
                → prepare_bertopic_input.py → Sentence Dataset (Parquet)

Sentence Dataset → bertopic_plus_octis.py → Topic Model + Assignments
```

## Output Locations

- **Coverage tables**: `data/interim/review_coverage_by_book.csv`
- **Sentence dataset**: `data/processed/review_sentences_for_bertopic.parquet`
- **EDA figures**: `reports/reviews_eda/`
- **Model artifacts**: `data/` and `reports/` (see subdirectory READMEs)

## Documentation

- **`bertopic_preparation/README.md`**: Detailed documentation for sentence preparation
- **`BERTopic_OCTIS/README.md`**: Detailed documentation for topic modeling
- **`docs/archive/`**: Archived planning documents (for reference)

## Dependencies

- Python 3.12+
- pandas >= 2.3.0
- spacy >= 3.8.0
- BERTopic, OCTIS
- See individual subdirectory READMEs for specific requirements

