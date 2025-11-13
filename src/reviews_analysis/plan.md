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

**Objective**: Split reviews into sentences and create sentence-level dataset for BERTopic (matching the original novels pipeline approach).

**Tasks**:
1. Create `src/reviews_analysis/prepare_bertopic_input.py`:
   - Input: English reviews joined to books with `pop_tier`
   - Split reviews into sentences using spaCy (same approach as original code)
   - Create sentence-level dataset where each row = one sentence
   - Output columns:
     - `sentence_id`: Unique identifier for each sentence
     - `sentence_text`: The sentence text (cleaned)
     - `review_id`: ID of the source review
     - `work_id`: ID of the book (work)
     - `pop_tier`: Quality tier of the book (trash/middle/top)
     - `rating`: Review rating (if available)
     - `sentence_index`: Index of sentence within its review
     - `n_sentences_in_review`: Total sentences in source review
   - Save: `data/processed/review_sentences_for_bertopic.parquet`

**Outputs**:
- Sentence-level dataset (one row per sentence)
- Metadata preserved for mapping topics back: sentence → review → book
- Documentation of sentence splitting strategy (spaCy model, min length, cleaning rules)

## Phase 5: Adapt BERTopic + OCTIS for Reviews

**Objective**: Adapt existing BERTopic+OCTIS workflow for review-based documents.

**Tasks**:
1. Study existing code:
   - `/home/polina/Documents/goodreads_romance_research_cursor/romantic_novels_project_code/src/stage03_modeling`
   - Understand: embedding models, UMAP/HDBSCAN settings, vectorizer configs, OCTIS integration

2. Create `src/reviews_analysis/run_bertopic_reviews.py`:
   - Reuse best configurations from `stage03_modeling` as starting point
   - **Input**: Sentence-level dataset from Phase 4 (`review_sentences_for_bertopic.parquet`)
   - Adjust hyperparameters for sentence-level analysis (similar to original novels code):
     - Consider different embedding models (e.g., models better for short texts)
     - Adjust `min_topic_size` (likely smaller than 127 for sentences)
     - Adjust UMAP `n_neighbors` and HDBSCAN `min_cluster_size` for sentence-level clustering
     - Consider different vectorizer settings (e.g., `min_df`, `ngram_range`)
   - Train at least one **global** BERTopic model on sentences
   - **Map topics back to reviews and books**:
     - Create topic-sentence assignments (from BERTopic output)
     - Aggregate sentence topics → review-level topic distributions
     - Aggregate review topics → book-level topic distributions
     - Preserve metadata (review_id, work_id, pop_tier) throughout mapping
   - Optionally: models per `pop_tier` group (trash, middle, top)
   - Integrate OCTIS to compute:
     - Topic coherence (e.g., C_V)
     - Topic diversity
   - Save:
     - Model artifacts
     - Topic-word lists
     - Topic-sentence assignments
     - Topic-review assignments (aggregated)
     - Topic-book assignments (aggregated)
     - OCTIS metrics

3. Create `src/reviews_analysis/config/`:
   - Configuration files for review-based BERTopic & OCTIS settings
   - Separate configs for global vs. per-group models if applicable

**Outputs**:
- Trained BERTopic models (global, optionally per-group) trained on sentences
- Topic-word lists and topic-sentence assignments
- Topic-review assignments (aggregated from sentences)
- Topic-book assignments (aggregated from reviews)
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
     - Sentence splitting strategy (spaCy model, min length, cleaning)
     - Topic mapping strategy (sentence → review → book aggregation)
     - Hyperparameter adjustments for sentence-level analysis vs. novel-level
   - Output file locations and formats

2. Optional: Use `task-researcher`'s `research_topic` to generate methods section on:
   - Topic modeling on review text using sentence-level analysis
   - Tradeoffs in sentence-level vs. document-level topic modeling
   - Aggregation strategies for mapping sentence topics to reviews and books
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
  - Sentence dataset: `data/processed/review_sentences_for_bertopic.parquet`
  - EDA figures: `reports/reviews_eda/`
  - Model artifacts: `data/` and `reports/` (clearly named subfolders)

## Dependencies

- Python virtual environment: `romance-novel-nlp-research`
- Existing packages: BERTopic, OCTIS, pandas, numpy, scikit-learn, etc.
- MCP server: `task-researcher` for planning and research

## Success Metrics

- ✅ Coverage report shows review availability for all 6,000 books
- ✅ EDA visualizations saved and insights documented
- ✅ Sentence-level dataset prepared and saved (one row per sentence)
- ✅ At least one global BERTopic model trained on sentences
- ✅ Topic assignments mapped back to reviews and books
- ✅ OCTIS metrics computed and saved
- ✅ Clear documentation of sentence-level approach and mapping strategy
- ✅ Comparison of topics across trash/middle/top groups (if per-group models created)

