# Corpus Construction & EDA Pipeline Review Report

**Repository:** `/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research`  
**Date:** November 2025  
**Purpose:** Comprehensive review of corpus building, preprocessing, and EDA work for both novel metadata and reader reviews

---

## Overview

This repository implements a research pipeline for analyzing romance novels using Goodreads metadata and reader reviews. The project constructs two main corpora: (1) a curated dataset of 6,000 romance novels with metadata (title, author, publication year, ratings, genres, etc.), and (2) a corpus of English-language reader reviews extracted from Goodreads for these novels. The repository contains well-structured code for data ingestion, cleaning, preprocessing, subsampling, and exploratory data analysis (EDA), with clear separation between raw data, intermediate processing steps, and final research-ready datasets.

---

## Corpus 1 – Goodreads Metadata for Romantic Novels

### Data Sources & Storage

**Raw Data Location:** `data/raw/`
- `goodreads_books_romance.json.gz` – Main book metadata (348MB)
- `goodreads_book_works.json.gz` – Work-level metadata (72MB)
- `goodreads_book_authors.json.gz` – Author information (17MB)
- `goodreads_book_series.json.gz` – Series data (27MB)
- `goodreads_book_genres_initial.json.gz` – Genre classifications (23MB)
- `goodreads_reviews_romance.json.gz` – Review data (1.2GB, ~3.6M reviews)
- `goodreads_interactions_romance.json.gz` – User-book interactions (2.1GB)

**Processed Data Location:** `data/processed/`
- `romance_books_main_final.csv` – Main processed dataset (source for subsampling)
- `romance_books_main_final_canonicalized.csv` – Main dataset with canonicalized genres
- `romance_subdataset_6000.csv` – Final 6,000-book research corpus

**Source Code:** `src/csv_building/final_csv_builder.py` (main builder), `src/data_quality/` (cleaning pipeline)

### Corpus Building & Preprocessing Steps

The corpus construction pipeline follows this sequence:

1. **Raw Data Ingestion** (`src/csv_building/final_csv_builder.py`)
   - Loads books data from `goodreads_books_romance.json.gz` into pandas DataFrame
   - Loads supplementary data: works, authors, series, genres (all as JSON dictionaries)
   - Processes data line-by-line from gzipped JSONL files

2. **Language Filtering**
   - Filters to English editions only using regex pattern: `^(eng|en(?:-[A-Za-z]+)?)$` (case-insensitive)
   - Applied in `final_csv_builder.py` line 377: `books_df['is_english'] = books_df['language_code'].apply(self.is_english_language)`

3. **Work-Level Aggregation**
   - Groups multiple editions by `work_id` to handle books with multiple editions
   - Aggregates metrics across editions:
     - `ratings_count_sum`: Sum of ratings across all editions
     - `text_reviews_count_sum`: Sum of text reviews across all editions
     - `average_rating_weighted_mean`: Weighted average rating (weighted by ratings_count)
     - `num_pages_median`: Median page count across editions
   - Creates `book_id_list_en`: Sorted list of book IDs (by descending ratings_count, then book_id)
   - Preserves longest description and union of popular shelves

4. **Title Cleaning**
   - Strips content within brackets/parentheses: `\s*\([^)]*\)|\s*\[[^\]]*\]`
   - Fallback title selection: `works.original_title` → `works.best_book_title` → `works.title` → `edition.title` → `'Untitled'`

5. **Publication Year Logic**
   - Prefers `original_publication_year` from works data (if 1800–2030)
   - Fallback: median of publication years from English editions
   - Filters to 2000–2020 range (applied in `build_final_csv_optimized`, line 492)

6. **Author Selection**
   - Selects primary author based on: (1) frequency across editions, (2) highest ratings_count, (3) lexicographically first if tie
   - Joins with authors data for `author_name`, `author_average_rating`, `author_ratings_count`

7. **Series Information**
   - Extracts series IDs from books data
   - Joins with series data for `series_title`, `series_works_count`
   - Sets `series_id = 'stand_alone'` for non-series books

8. **Genre Aggregation**
   - Aggregates genres from all books in a work (union of genre names)
   - Stored as comma-separated string in `genres` column

9. **Data Quality Pipeline** (`src/data_quality/comprehensive_data_cleaner.py`)
   - **Step 1:** Missing values cleaning (removes books with missing descriptions, pages)
   - **Step 2:** Duplicate detection
   - **Step 3:** Data type validation
   - **Step 4:** Outlier detection and treatment (publication year outliers, negative work counts)
   - **Step 5:** Data type optimization (memory efficiency)
   - **Step 6:** Final quality validation
   - **Filters applied:**
     - Remove books outside 2000–2017 range (122 books removed, per `comprehensive_data_cleaner.py`)
     - Remove books with missing descriptions (6,172 removed)
     - Remove books with missing pages (35,908 removed)
     - Apply ratings/reviews cuts (optional, configurable: default `ratings_min=10`, `reviews_min=2`)
     - Remove low-rating authors (optional)

10. **Null Value Handling** (`final_csv_builder.py`, `safe_numeric_conversion` method)
    - Handles empty strings before numeric conversion
    - Replaces common non-numeric values ('Unknown', 'N/A', 'TBD', '0') with None
    - Tracks conversion issues in quality metrics

**Output:** `romance_books_main_final.csv` (location: `data/processed/` or referenced from `anna_archive_romance_pipeline/data/processed/`)

### Subsampling to 6,000 Novels

**Implementation:** `src/subdataset_sampling/create_subdataset_6000.py`

**Selection Criteria:**
- **Target size:** Exactly 6,000 books (2,000 per popularity tier)
- **Tier definition:** Based on `average_rating_weighted_mean` quantiles:
  - **Thrash:** Bottom 25% (< Q1 ≈ 3.71)
  - **Mid:** Middle 50% (Q1 to Q3 ≈ 3.71 to 4.15)
  - **Top:** Top 25% (> Q3 ≈ 4.15)
- **Engagement priority:** Within each tier, books ranked by:
  1. `ratings_count_sum` (descending)
  2. `text_reviews_count_sum` (descending)
  3. `author_ratings_count` (descending)

**Stratification Strategy:**
- **Primary stratification:** By `(publication_year, genre_group)` within each tier
  - Genre groups: `paranormal`, `historical`, `fantasy`, `mystery`, `young_adult`, `other`
  - Quotas calculated proportionally from full dataset distribution
- **Secondary stratification:** Within each cell, preserves `(series_flag, pages_bin)` composition
  - `pages_bin`: Quartiles of `num_pages_median` (Q1, Q2, Q3, Q4)
  - `series_flag`: Binary (1 = series, 0 = standalone)

**Additional Processing:**
- Removes comics/graphic books: `~meta['genres_str'].str.contains('comics|graphic', case=False, na=False)`
- Genre canonicalization: Merges variations (e.g., 'sci fi' → 'science fiction', 'historical fiction' → 'historical romance')
- Backfilling: If quotas don't reach 2,000 per tier, backfills in engagement order
- Random seed: `RANDOM_SEED = 42` (for reproducibility, though selection is deterministic based on engagement)

**Validation:**
- Tier balance check (exactly 2,000 per tier)
- Representativeness validation: Compares distributions of publication year, genre_group, series_flag, pages_bin between full dataset and sample
- Statistical validation: Chi-square test for genre distribution
- Engagement quality check: Compares median engagement metrics by tier

**Output:** `data/processed/romance_subdataset_6000.csv` (columns: `work_id`, `title`, `author_id`, `author_name`, `publication_year`, `num_pages_median`, `genres_str`, `series_id`, `series_title`, `ratings_count_sum`, `text_reviews_count_sum`, `average_rating_weighted_mean`, `pop_tier`)

### What's Clear vs. Unclear

**Well-Documented:**
- ✅ Subsampling strategy fully documented in `src/subdataset_sampling/README.md` and code
- ✅ CSV building pipeline clearly implemented with logging
- ✅ Data quality pipeline has 6-step structure with clear validation
- ✅ Work-level aggregation logic is explicit in code
- ✅ Genre canonicalization rules are visible in code

**Missing or Ambiguous:**
- ⚠️ **Source of `romance_books_main_final.csv`:** The file is referenced in multiple places (including `extract_reviews.py` line 29 pointing to `anna_archive_romance_pipeline/data/processed/`), but the exact path where it's generated in this repository is not clearly documented. The `final_csv_builder.py` generates files with timestamps (e.g., `final_books_2000_2020_en_enhanced_*.csv`), but it's unclear if `romance_books_main_final.csv` is a renamed version or comes from a different pipeline.
- ⚠️ **Data quality pipeline execution:** While the code exists in `src/data_quality/`, it's not clear from the repository whether `romance_books_main_final.csv` was created with or without running the comprehensive data cleaner. The cleaner appears to be a separate step that may or may not have been applied.
- ⚠️ **Publication year range inconsistency:** The CSV builder filters to 2000–2020 (line 492), but the data cleaner removes books outside 2000–2017 (per documentation). It's unclear which range was actually applied to the final dataset.
- ⚠️ **Genre normalization timing:** Genre canonicalization happens in the subsampling script, but it's unclear if the main dataset also has canonicalized genres or if this is only applied during subsampling.

---

## Corpus 2 – Readers' Reviews

### Review Corpus Construction

**Processed Reviews Files:**
- `data/processed/romance_reviews_english.csv` – Full English reviews for all books in main dataset
- `data/processed/romance_reviews_english_subdataset_6000.csv` – English reviews for 6,000-book subdataset

**Extraction Implementation:** `src/review_extraction/extract_reviews.py`

**Extraction Process:**
1. **Book ID Loading:**
   - **Mode 1 (direct):** Reads `book_id_list_en` column from `romance_books_main_final.csv` (parsed as Python list using `ast.literal_eval`)
   - **Mode 2 (work_id mapping):** For subdataset, loads `work_id` from subdataset CSV, then maps to `book_id_list_en` from main dataset
   - Collects all unique book IDs into a set

2. **Review Filtering:**
   - Reads `goodreads_reviews_romance.json.gz` line-by-line (JSONL format, ~3.6M reviews)
   - Filters reviews where `book_id` matches the loaded book ID set
   - Skips reviews shorter than 10 characters (`MIN_REVIEW_LENGTH = 10`)

3. **Language Detection:**
   - Uses `langdetect` library (`detect()` function)
   - Keeps only reviews detected as English (`detected_lang == 'en'`)
   - Handles `LangDetectException` by skipping the review

4. **Output:**
   - CSV with columns: `review_id`, `review_text`, `rating`, `book_id`
   - Supports resume functionality: checks existing `review_id`s in output file and skips them

**Performance:**
- Processing rate: ~100–110 reviews/second
- Total processing time: ~9–10 hours for full dataset
- Progress logging: Every 5,000 reviews processed
- Logs saved to: `logs/extract_reviews_YYYYMMDD_HHMMSS.log`

**Monitoring Tools:**
- `src/review_extraction/monitor_extraction.py` – Real-time progress monitoring
- `src/review_extraction/estimate_time.py` – Time remaining estimation
- `src/review_extraction/monitor.sh` – Quick wrapper script

### Preprocessing & Cleaning

**Sentence-Level Processing:** `src/reviews_analysis/prepare_bertopic_input.py`

1. **Sentence Splitting:**
   - Uses spaCy model (`en_core_web_sm`) with optimized pipeline (only tokenizer and sentence segmenter enabled)
   - Batch processing with `nlp.pipe()` for efficiency (default `spacy_batch_size=1000`)
   - Minimum sentence length: 10 characters (configurable)

2. **Text Cleaning** (`clean_sentence_text` function):
   - Removes extra newlines: `' '.join(str(x).split('\n'))`
   - Normalizes whitespace: `' '.join(str(x).split()).strip()`
   - Converts to lowercase: `.lower()`
   - Removes empty sentences after cleaning

3. **Metadata Preservation:**
   - Each sentence row includes: `sentence_id`, `sentence_text`, `review_id`, `work_id`, `pop_tier`, `rating`, `sentence_index`, `n_sentences_in_review`
   - Links sentences back to source review and book for downstream analysis

**Output:** Sentence-level parquet files in `data/processed/review_sentences_temp/` (chunked for memory efficiency)

**Note:** The code exists but it's unclear from the repository whether sentence-level preprocessing has been fully executed. The `review_sentences_temp/` directory contains 103 parquet chunks, suggesting partial or complete execution.

### EDA on Reviews

**Implementation:** `src/reviews_analysis/eda_reviews.ipynb` (Jupyter notebook)

**Analysis Performed:**

1. **Review Count Distributions:**
   - Overall distribution (histogram)
   - Distribution by quality tier (thrash/mid/top)
   - Output: `reports/reviews_eda/review_count_distribution.png`

2. **Review Length Distributions:**
   - Character length: Mean 747.8, Median 348.0, Range 10–20,024
   - Token length: Mean 137.7, Median 65.0, Range 1–3,789
   - Distributions overall and by tier
   - Outputs: `review_length_chars_distribution.png`, `review_length_tokens_distribution.png`

3. **Ratings Distributions:**
   - Rating distributions (1–5 stars) overall and by tier
   - Output: `ratings_distribution.png`

4. **Review Metrics Boxplots:**
   - Side-by-side boxplots comparing review counts per book, character length, and token count across tiers
   - Output: `review_metrics_boxplots.png`

5. **Top Words by Tier:**
   - Horizontal bar charts showing top 15 words by frequency for each tier
   - Stopword handling: "story" removed from stopwords; "characters" and "plot" retained as content words
   - Output: `top_words_by_tier.png`

6. **Summary Statistics:**
   - Tabular summary of key metrics aggregated by tier
   - Output: `reports/reviews_eda/summary_statistics.csv`

**Key Findings (from `PHASE_3_EDA_COMPLETION_SUMMARY.md`):**
- **Total reviews:** 969,675 reviews from 5,998 books (99.97% coverage)
- **Trash tier:** 121,343 reviews, mean 60.7/book, mean rating 3.35
- **Middle tier:** 480,126 reviews, mean 240.1/book, mean rating 3.89
- **Top tier:** 368,206 reviews, mean 184.2/book, mean rating 4.27
- **Pattern:** Review volume increases from trash → middle (4×), then decreases slightly middle → top; ratings increase monotonically trash → middle → top

**Output Location:** `reports/reviews_eda/` (6 PNG files + 1 CSV)

### What's Clear vs. Unclear

**Well-Documented:**
- ✅ Review extraction process fully documented in `src/review_extraction/README.md`
- ✅ EDA notebook clearly shows all analysis steps
- ✅ Review coverage statistics are explicit (5,998/6,000 books have reviews)
- ✅ Monitoring and logging tools are well-implemented

**Missing or Ambiguous:**
- ⚠️ **Review preprocessing completeness:** While `prepare_bertopic_input.py` exists and `review_sentences_temp/` contains 103 parquet chunks, it's unclear if sentence-level preprocessing has been completed for the full review corpus or only partially executed.
- ⚠️ **Review text cleaning details:** The EDA notebook computes token counts using simple word splitting (`str.split().str.len()`), but it's unclear if more sophisticated tokenization (e.g., spaCy tokenization) was applied for the actual analysis or only for BERTopic preparation.
- ⚠️ **HTML/emoji handling:** The review extraction script outputs raw `review_text`, but there's no explicit documentation of HTML tag removal or emoji handling in the review corpus. The sentence-level preprocessing does basic cleaning, but it's unclear if this was applied to the full corpus or only for BERTopic input.
- ⚠️ **Review coverage details:** While `data/interim/review_coverage_by_book.csv` exists, it's not clear what this file contains or how it was generated. The EDA notebook references `generate_coverage_table()` from `checks_coverage.py`, but the exact coverage analysis methodology is not fully documented in the repository.

---

## Potential Follow-Up Coding Tasks

### Corpus Building & Documentation

1. **Clarify main dataset generation path:** Create a single entry point script that documents the exact sequence of steps to generate `romance_books_main_final.csv` from raw data, including whether the comprehensive data cleaner is applied and in what order.

2. **Consolidate publication year filtering:** Resolve the inconsistency between 2000–2020 (CSV builder) and 2000–2017 (data cleaner) and document which range was actually applied to the final dataset.

3. **Document genre canonicalization:** Clarify whether genre canonicalization is applied to the main dataset or only during subsampling, and create a separate script if needed to canonicalize the main dataset consistently.

4. **Create corpus construction pipeline script:** Build a single CLI entry point (e.g., `python scripts/build_corpus.py --from-raw --apply-cleaning --create-subset`) that orchestrates the entire pipeline from raw JSON files to the 6,000-book subset with clear logging of each step.

### Preprocessing & Reproducibility

5. **Standardize review preprocessing:** Create a unified review preprocessing script that applies consistent cleaning (HTML removal, emoji handling, tokenization) to the full review corpus and saves a cleaned version, not just for BERTopic input.

6. **Document sentence-level preprocessing status:** Add a script or notebook cell that verifies whether sentence-level preprocessing has been completed for the full corpus and provides statistics on coverage.

7. **Create preprocessing configuration file:** Move hard-coded parameters (e.g., `MIN_REVIEW_LENGTH=10`, `spacy_batch_size=1000`, publication year ranges) into a YAML configuration file for easier reproducibility and experimentation.

8. **Add data validation checkpoints:** Create validation scripts that verify data integrity at each major step (after CSV building, after cleaning, after subsampling, after review extraction) with clear pass/fail criteria.

### EDA Coverage

9. **Expand EDA for main dataset:** Create EDA notebooks/scripts for the full `romance_books_main_final.csv` dataset (not just the 6,000-book subset) to document distributions of publication years, genres, ratings, page counts, etc., before subsampling.

10. **Review coverage analysis:** Document the `review_coverage_by_book.csv` generation process and create visualizations showing review coverage patterns (e.g., books with 0 reviews, distribution of review counts, coverage by tier/publication year).

11. **Cross-corpus validation:** Create analysis that validates the representativeness of the 6,000-book subset by comparing key metrics (genre distributions, publication year distributions, engagement metrics) between the full dataset and the subset with statistical tests.

---

## Summary

This repository contains a well-structured pipeline for constructing two research corpora: a 6,000-book romance novel metadata dataset and a corresponding corpus of English-language reader reviews. The subsampling strategy is clearly documented and implements a sophisticated stratification approach. The review extraction process is robust with good monitoring tools. EDA on reviews has been completed with comprehensive visualizations and summary statistics.

The main areas for improvement are: (1) clarifying the exact sequence of steps that produced `romance_books_main_final.csv`, (2) documenting whether and how the data quality pipeline was applied, (3) standardizing review preprocessing across the full corpus, and (4) creating a single entry point script for reproducing the entire corpus construction pipeline from raw data to final research datasets.

