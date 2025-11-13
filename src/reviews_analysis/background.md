# Background: Review-Based Topic Modeling Pipeline

## Research Context

This project extends the existing romance novel NLP research by building a **review-based topic modeling pipeline**. We have:

- **6,000 romance novels** with quality classifications (`pop_tier`: trash/middle/top)
- **English Goodreads reviews** extracted and processed for these books
- An existing **BERTopic + OCTIS workflow** designed for full novel texts that we need to adapt for shorter, aggregated review documents

## Key Data Sources

### Books Dataset
- **Location**: `data/processed/romance_subdataset_6000.csv`
- **Key columns**:
  - `work_id`: Unique book identifier (join key with reviews)
  - `pop_tier`: Quality classification (trash/middle/top)
  - `title`, `author_name`, `publication_year`, `num_pages_median`
  - `ratings_count_sum`, `text_reviews_count_sum`, `average_rating_weighted_mean`
- **Purpose**: Main dataset of 6,000 books with quality flags

### Reviews Dataset
- **Location**: `data/processed/romance_reviews_english.csv`
- **Key columns**:
  - `review_id`: Unique review identifier
  - `book_id`: Foreign key to books (maps to `work_id` in books dataset)
  - `review_text`: The actual review content (English, already filtered)
  - `rating`: Star rating (if available)
- **Purpose**: Processed English reviews ready for analysis

### Review Extraction Logs
- **Location**: `logs/` directory
- **Contains**: Meta-information about the review extraction process, timing, coverage statistics
- **Purpose**: Diagnostic information about review availability

## Existing Code to Reuse

### BERTopic + OCTIS Implementation
- **Location**: `/home/polina/Documents/goodreads_romance_research_cursor/romantic_novels_project_code/src/stage03_modeling`
- **Key components**:
  - `bertopic_runner.py`: Main BERTopic wrapper with OCTIS integration
  - `retrain_from_tables.py`: Model retraining with coherence optimization
  - Uses: UMAP, HDBSCAN, CountVectorizer, ClassTfidfTransformer
  - Evaluation: OCTIS metrics (coherence, diversity)
  - Hyperparameters: min_topic_size, n_neighbors, embedding models, etc.
- **Current use**: Designed for full novel chapter/sentence texts
- **Adaptation needed**: Adjust for shorter, noisier review-based documents

## Research Goals

1. **Coverage Verification**: Ensure every book in the 6,000-book dataset has been checked for English reviews, with clear reporting of books with zero or few reviews.

2. **Distribution Analysis**: Understand how reviews are distributed across the three quality groups (trash/middle/top).

3. **Comprehensive EDA**: Explore review data globally and by quality group:
   - Review counts per book
   - Review lengths (characters/tokens)
   - Ratings distribution
   - Lexical patterns

4. **Document Preparation**: Aggregate reviews per book into single documents suitable for BERTopic input.

5. **Topic Modeling Pipeline**: Adapt the existing BERTopic+OCTIS workflow to work on review-based documents instead of novel texts, with appropriate hyperparameter adjustments for shorter, noisier texts.

6. **Comparative Analysis**: Compare topics across trash/middle/top groups to understand how review language differs by book quality.

## Technical Challenges

- **Text Length**: Reviews are much shorter than novel chapters, requiring different embedding and clustering strategies
- **Noise**: Reviews may contain more informal language, typos, and varied writing styles
- **Aggregation**: Need to decide how to combine multiple reviews per book (simple concatenation vs. weighted approaches)
- **Coverage Gaps**: Some books may have very few or no reviews, requiring handling strategies
- **Hyperparameter Tuning**: Existing BERTopic settings (e.g., min_topic_size=127) may be too large for review-based documents

## Success Criteria

- Complete coverage report showing review availability for all 6,000 books
- Comprehensive EDA with visualizations saved to `reports/reviews_eda/`
- Per-book review documents prepared and saved to `data/processed/review_documents_per_book.csv`
- At least one global BERTopic model trained on review documents
- OCTIS metrics computed (coherence, diversity)
- Clear documentation of adaptations made for review-based modeling
- Comparison of topics across trash/middle/top groups

