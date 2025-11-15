# Stage 05: Prepare Reviews Corpus for BERTopic - Scientific Documentation

## Research Objectives

- Convert review-level data into sentence-level data for BERTopic analysis
- Create a corpus suitable for topic modeling on reader reviews
- Ensure data quality and coverage for downstream topic modeling

## Research Questions

1. How should reviews be segmented for optimal topic extraction?
2. What is the coverage of reviews across the 6,000 book dataset?
3. How does sentence-level analysis compare to review-level analysis for topic modeling?

## Hypotheses

- **H1**: Sentence-level segmentation provides more granular topic extraction than review-level
- **H2**: spaCy sentence segmentation effectively handles review text structure
- **H3**: Review coverage varies significantly across quality tiers (trash/middle/top)
- **H4**: Sentence-level data enables better topic coherence in BERTopic models

## Dataset

- **Input**: 
  - Review-level data: `romance_reviews_english.csv`
  - Book metadata: `romance_subdataset_6000.csv`
  - Book ID mapping: `romance_books_main_final.csv`
- **Output**: 
  - Sentence-level dataset: `review_sentences_for_bertopic.parquet`
  - Each row represents a single sentence with metadata linking to review and book
- **Format**: Parquet format for efficient storage and loading

## Methodology

- **Sentence Segmentation**: Uses spaCy `en_core_web_sm` model for sentence splitting
- **Data Mapping**: Maps reviews from `book_id` to `work_id` for joining with book metadata
- **Chunked Processing**: Processes reviews in chunks of 20,000 to manage memory
- **Incremental Saving**: Saves progress to chunk files to prevent data loss
- **Coverage Analysis**: Computes review counts per book and distribution by quality tier

## Tools

- **spaCy**: Natural language processing for sentence segmentation
- **pandas**: Data manipulation and joining
- **pyarrow**: Efficient parquet file I/O
- **tqdm**: Progress tracking

## Statistical Tools

- Review coverage statistics
- Distribution analysis by quality tier
- Sentence length distributions
- Processing rate metrics

## Results

- Sentence-level corpus ready for BERTopic analysis
- Review coverage statistics by book and quality tier
- Processing logs and performance metrics
- Data quality validation results

## Data Quality

- All sentences linked to source review and book
- Metadata preserved: work_id, book_id, review_id, sentence_index
- Only English reviews included
- Only reviews for books in 6,000 book dataset included

