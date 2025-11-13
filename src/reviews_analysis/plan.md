# Plan: Review-Based Topic Modeling Pipeline

## Overview

Build a complete pipeline for topic modeling on Goodreads reviews of 6,000 romance novels, adapting the existing BERTopic+OCTIS workflow from novel texts to review-based documents.

## Project Structure

All work happens in:
- **Main repo**: `/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research`
- **Code location**: `src/reviews_analysis/`
- **Virtual environment**: `/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research/romance-novel-nlp-research`

## Phase 1: Data Loading and Linking

**Objective**: Establish reliable data loading and joining between books and reviews.

**Tasks**:
1. Create `src/reviews_analysis/data_loading.py` with:
   - `load_books()`: Load `data/processed/romance_subdataset_6000.csv`
   - `load_reviews()`: Load `data/processed/romance_reviews_english.csv`
   - `load_joined_reviews()`: Join reviews to books using `book_id` (reviews) ↔ `work_id` (books)
   - Schema validation and ID mapping documentation

**Outputs**:
- Modular data loading functions
- Documentation of join keys and any ID mismatches

## Phase 2: Coverage Verification

**Objective**: Verify and report review coverage for all 6,000 books.

**Tasks**:
1. Create `src/reviews_analysis/checks_coverage.py`:
   - For each book in `romance_subdataset_6000.csv`, compute:
     - `n_reviews_english`: Count of English reviews
     - Optionally `n_reviews_total` if accessible from logs
   - Join counts back to book dataset, preserving `pop_tier`
   - Generate summary statistics:
     - Books with 0 reviews
     - Distribution of review counts overall and by `pop_tier`
   - Write coverage table: `data/interim/review_coverage_by_book.csv`

**Outputs**:
- Coverage table with review counts per book
- Summary report (printed/logged) of coverage statistics

## Phase 3: Exploratory Data Analysis

**Objective**: Comprehensive EDA on review data, globally and by quality group.

**Tasks**:
1. Create `src/reviews_analysis/eda_reviews.ipynb` (or `.py`):
   - Load joined reviews+books table
   - Compute and visualize:
     - Distribution of English reviews per book (overall and by `pop_tier`)
     - Distribution of review lengths (characters/tokens)
     - Ratings distribution (if available), overall and by `pop_tier`
   - Basic lexical analysis:
     - Frequent words/phrases per group (after cleaning/stopword removal)
   - Save figures to `reports/reviews_eda/`

**Outputs**:
- EDA notebook with visualizations
- Figures saved to `reports/reviews_eda/`
- Summary statistics and insights

## Phase 4: Prepare BERTopic Input

**Objective**: Aggregate reviews per book into single documents for BERTopic.

**Tasks**:
1. Create `src/reviews_analysis/prepare_bertopic_input.py`:
   - Input: English reviews joined to books with `pop_tier`
   - Aggregate reviews per book:
     - Concatenate all English reviews for each book
     - Optionally cap total length if needed (document max length considerations)
   - Output columns:
     - `book_id` (or `work_id`)
     - `pop_tier` (trash/middle/top)
     - `concat_reviews_text` (aggregated review text)
   - Save: `data/processed/review_documents_per_book.csv` (or `.parquet`)

**Outputs**:
- Per-book review documents dataset
- Documentation of aggregation strategy (concatenation, length caps, etc.)

## Phase 5: Adapt BERTopic + OCTIS for Reviews

**Objective**: Adapt existing BERTopic+OCTIS workflow for review-based documents.

**Tasks**:
1. Study existing code:
   - `/home/polina/Documents/goodreads_romance_research_cursor/romantic_novels_project_code/src/stage03_modeling`
   - Understand: embedding models, UMAP/HDBSCAN settings, vectorizer configs, OCTIS integration

2. Create `src/reviews_analysis/run_bertopic_reviews.py`:
   - Reuse best configurations from `stage03_modeling` as starting point
   - Adjust hyperparameters for shorter, noisier review texts:
     - Consider different embedding models (e.g., models better for short texts)
     - Adjust `min_topic_size` (likely smaller than 127 for reviews)
     - Adjust UMAP `n_neighbors` and HDBSCAN `min_cluster_size`
     - Consider different vectorizer settings (e.g., `min_df`, `ngram_range`)
   - Train at least one **global** BERTopic model on:
     - `data/processed/review_documents_per_book.csv`
   - Optionally: models per `pop_tier` group (trash, middle, top)
   - Integrate OCTIS to compute:
     - Topic coherence (e.g., C_V)
     - Topic diversity
   - Save:
     - Model artifacts
     - Topic-word lists
     - Topic-document matrices
     - OCTIS metrics

3. Create `src/reviews_analysis/config/`:
   - Configuration files for review-based BERTopic & OCTIS settings
   - Separate configs for global vs. per-group models if applicable

**Outputs**:
- Trained BERTopic models (global, optionally per-group)
- Topic-word lists and topic-document assignments
- OCTIS evaluation metrics
- Model artifacts saved to clearly named folders under `data/` and `reports/`

## Phase 6: Documentation

**Objective**: Document the complete pipeline and key decisions.

**Tasks**:
1. Create `src/reviews_analysis/README.md`:
   - Pipeline overview (from input files to BERTopic/OCTIS results)
   - Step-by-step execution guide (which scripts/notebooks, in what order)
   - Key decisions documented:
     - Minimum review count per book (if any filtering applied)
     - Text cleaning rules
     - Aggregation strategy (how reviews combined per book)
     - Hyperparameter adjustments for reviews vs. novels
   - Output file locations and formats

2. Optional: Use `task-researcher`'s `research_topic` to generate methods section on:
   - Topic modeling on review text
   - Tradeoffs in aggregating many short reviews into one document per entity
   - Interpretation of topics across trash/middle/top groups

**Outputs**:
- Complete README with pipeline documentation
- Optional research-backed methods section

## Key Paths Reference

- **Reviews**: `data/processed/romance_reviews_english.csv`
- **Books**: `data/processed/romance_subdataset_6000.csv`
- **Existing modeling code**: `/home/polina/Documents/goodreads_romance_research_cursor/romantic_novels_project_code/src/stage03_modeling`
- **Review extraction logs**: `logs/`
- **Output locations**:
  - Coverage tables: `data/interim/`
  - Prepared documents: `data/processed/`
  - EDA figures: `reports/reviews_eda/`
  - Model artifacts: `data/` and `reports/` (clearly named subfolders)

## Dependencies

- Python virtual environment: `romance-novel-nlp-research`
- Existing packages: BERTopic, OCTIS, pandas, numpy, scikit-learn, etc.
- MCP server: `task-researcher` for planning and research

## Success Metrics

- ✅ Coverage report shows review availability for all 6,000 books
- ✅ EDA visualizations saved and insights documented
- ✅ Per-book review documents prepared and saved
- ✅ At least one global BERTopic model trained on reviews
- ✅ OCTIS metrics computed and saved
- ✅ Clear documentation of adaptations for review-based modeling
- ✅ Comparison of topics across trash/middle/top groups (if per-group models created)

