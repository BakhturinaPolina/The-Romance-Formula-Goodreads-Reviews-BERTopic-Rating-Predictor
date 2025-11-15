# Stage 04: Review Extraction - Scientific Documentation

## Research Objectives

- Extract English-language reviews from Goodreads dataset
- Filter reviews matching books in the main dataset
- Prepare review data for topic modeling analysis

## Research Questions

1. How can English reviews be effectively identified and extracted?
2. What proportion of reviews match books in the main dataset?
3. How can review extraction be efficiently processed at scale?

## Hypotheses

- **H1**: Language detection accurately identifies English reviews
- **H2**: Review extraction enables topic modeling analysis
- **H3**: Efficient processing enables large-scale review analysis

## Dataset

- **Input**: Raw Goodreads reviews JSON file (~3.6M reviews, 1.2GB)
- **Output**: Filtered English reviews CSV (review_id, review_text, rating, book_id)
- **Processing**: Line-by-line processing with language detection

## Methodology

- **Language Detection**: Uses langdetect library for automatic language identification
- **Book ID Matching**: Matches reviews to books using book_id from main dataset
- **Incremental Processing**: Processes reviews line-by-line for memory efficiency
- **Progress Monitoring**: Real-time tracking of extraction progress

## Tools

- pandas: Data manipulation
- langdetect: Language detection
- gzip/json: Reading compressed JSON files

## Statistical Tools

- Language detection accuracy validation
- Review coverage statistics
- Processing time estimation

## Results

- English reviews extracted and filtered
- Review coverage statistics calculated
- Processing logs for reproducibility
- Time estimates for full dataset processing

