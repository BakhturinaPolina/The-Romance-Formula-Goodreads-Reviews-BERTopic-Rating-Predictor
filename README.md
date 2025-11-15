# Goodreads Metadata and Reviews Topic Modeling

A research pipeline for analyzing Goodreads book metadata and reader reviews using topic modeling techniques.

## Overview

This project processes Goodreads book metadata and reader reviews to extract topics and analyze patterns in reader engagement. The pipeline handles data integration, quality assurance, text preprocessing, review extraction, topic modeling, and corpus analysis.

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd goodreads-topic-modeling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

The pipeline consists of 8 stages:

1. **Data Integration** (`src/01_data_integration/`)
   ```bash
   cd src/01_data_integration
   python run_builder.py
   ```

2. **Data Quality** (`src/02_data_quality/`)
   ```bash
   cd src/02_data_quality
   python pipeline_runner.py
   ```

3. **Text Preprocessing** (`src/03_text_preprocessing/`)
   ```bash
   cd src/03_text_preprocessing
   python run_preprocessor.py
   ```

4. **Review Extraction** (`src/04_review_extraction/`)
   ```bash
   cd src/04_review_extraction
   python extract_reviews.py
   ```

5. **Prepare Reviews Corpus for BERTopic** (`src/05_prepare_reviews_for_BERTopic/`)
   ```bash
   cd src/05_prepare_reviews_for_BERTopic
   # See stage README for detailed instructions
   ```

6. **Topic Modeling** (`src/06_topic_modeling/`)
   ```bash
   cd src/06_topic_modeling
   # See stage README for detailed instructions
   ```

7. **Shelf Normalization** (`src/07_shelf_normalization/`)
   ```bash
   cd src/07_shelf_normalization
   make pipeline
   ```

8. **Corpus Analysis** (`src/08_corpus_analysis/`)
   ```bash
   cd src/08_corpus_analysis
   python generate_corpus_statistics.py
   ```

## Project Structure

```
goodreads-topic-modeling/
├── data/                    # Data storage
│   ├── raw/                 # Original Goodreads JSON files (read-only)
│   ├── intermediate/        # Temporary processing outputs
│   └── processed/          # Final cleaned datasets
├── src/                     # Source code (organized by research stage)
│   ├── 01_data_integration/
│   ├── 02_data_quality/
│   ├── 03_text_preprocessing/
│   ├── 04_review_extraction/
│   ├── 05_prepare_reviews_for_BERTopic/
│   ├── 06_topic_modeling/
│   ├── 07_shelf_normalization/
│   └── 08_corpus_analysis/
├── outputs/                 # All research outputs
│   ├── datasets/            # Processed datasets
│   ├── reports/             # Analysis reports
│   ├── visualizations/     # Plots and charts
│   └── logs/                # Execution logs
├── docs/                     # Project documentation
│   ├── setup.md             # Detailed setup instructions
│   └── replication_guide.md # How to use with other datasets
└── notebooks/               # Jupyter notebooks for exploration
```

## Documentation

- **Scientific README**: See `README_SCIENTIFIC.md` for research objectives, methodology, and results
- **Setup Guide**: See `docs/setup.md` for detailed setup instructions
- **Replication Guide**: See `docs/replication_guide.md` for adapting the pipeline to other datasets
- **Stage Documentation**: Each stage directory contains `README.md` and `README_SCIENTIFIC.md`

## Dataset

The project uses Goodreads metadata including:
- Book metadata (titles, authors, publication years, ratings, genres)
- Reader reviews (text reviews with ratings)
- User-book interactions
- Series and work-level information

All data is stored in `data/raw/` as compressed JSON files. Processed datasets are saved to `data/processed/`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Data Source**: UCSD Goodreads Book Graph
