# Goodreads Metadata and Reviews Topic Modeling - Scientific Documentation

## Research Objectives

### Primary Research Question
How do thematic characteristics of books relate to their popularity among readers, as measured through Goodreads metadata and reader reviews?

### Specific Objectives
1. **Topic Extraction**: Extract themes from reader reviews using topic modeling techniques
2. **Review Analysis**: Analyze reader reviews to identify key themes and preferences
3. **Correlation Analysis**: Examine relationships between book characteristics and popularity metrics
4. **Comparative Analysis**: Compare metadata-based characteristics with reader-perceived themes

## Research Questions

1. What topics emerge from reader reviews of books in the dataset?
2. How do extracted topics relate to book popularity metrics (ratings, review counts)?
3. What patterns exist in reader engagement across different book characteristics?
4. How do reader-perceived themes (from reviews) compare to metadata-based classifications?

## Hypotheses

1. **H1**: Distinct topics will emerge from reader reviews that reflect reader concerns and preferences
2. **H2**: Books with higher popularity metrics will show different topic distributions than less popular books
3. **H3**: Reader review topics will correlate with metadata characteristics (genres, publication years, etc.)
4. **H4**: Topic modeling will reveal themes not captured by standard metadata classifications

## Dataset

### Data Sources
The project uses Goodreads metadata from the UCSD Goodreads Book Graph:

- **Book Metadata**: Titles, authors, publication years, ratings, genres, series information
- **Reader Reviews**: Text reviews with ratings (~3.6M reviews, 1.2GB compressed)
- **User Interactions**: User-book interactions and reading patterns
- **Work-Level Aggregation**: Multiple editions aggregated to work-level for fair comparison

### Data Processing Approach

The dataset uses **work-level aggregation** to handle multiple editions of the same book:

- **Individual Edition Fields**: Removed from final dataset
  - `average_rating` (individual edition rating)
  - `ratings_count` (individual edition ratings count)
  - `text_reviews_count` (individual edition reviews count)

- **Work-Level Aggregated Fields**: Used for all analysis
  - `average_rating_weighted_mean` (weighted average across all editions)
  - `ratings_count_sum` (total ratings across all editions)
  - `text_reviews_count_sum` (total reviews across all editions)

This approach ensures fair comparison between books regardless of how many editions they have.

### Sample Characteristics
- **Main Dataset**: Processed book metadata with comprehensive quality assurance
- **Review Dataset**: English-language reviews extracted and filtered
- **Subdataset**: Representative 6,000-book sample for topic modeling analysis

## Methodology and Methods

### Research Pipeline

The research follows an 8-stage pipeline:

1. **Data Integration**: CSV building, data integration, representative sampling
2. **Data Quality**: 6-step quality assurance pipeline (missing values, duplicates, data types, outliers, optimization, validation)
3. **Text Preprocessing**: HTML cleaning, text normalization, genre categorization
4. **Review Extraction**: Language detection, filtering, and extraction of English reviews
5. **Prepare Reviews Corpus for BERTopic**: Sentence splitting and corpus creation from reviews
6. **Topic Modeling**: BERTopic analysis with hyperparameter optimization using OCTIS framework
7. **Shelf Normalization**: Normalization of user-generated shelf tags
8. **Corpus Analysis**: Statistical analysis of corpus characteristics

### Topic Modeling Methodology

- **Model**: BERTopic with various embedding models
- **Framework**: OCTIS (Optimization and Comparative Topic Modeling Infrastructure for Scholars)
- **Hyperparameter Optimization**: Bayesian optimization using OCTIS framework
- **Evaluation Metrics**: Topic coherence, diversity, and quality metrics
- **Sentence-Level Analysis**: Reviews split into sentences for granular topic extraction

### Statistical Methods

- **Heavy-Tail Analysis**: Clauset-Shalizi-Newman (2009) power-law fitting methodology
- **Overdispersion Testing**: Dean-Lawless and Cameron-Trivedi formal statistical tests
- **Quality Assurance**: Comprehensive validation with automated quality gates
- **Representative Sampling**: Stratified sampling preserving key demographic characteristics

## Tools

### Programming Languages and Frameworks
- **Python 3.8+**: Primary programming language
- **pandas**: Data manipulation and analysis
- **spaCy**: Natural language processing and sentence splitting
- **BERTopic**: Topic modeling framework
- **OCTIS**: Topic modeling optimization framework

### Data Processing Tools
- **pandas**: Data manipulation
- **pyarrow**: Parquet file handling
- **langdetect**: Language detection for review filtering

### Statistical Analysis
- **scipy**: Statistical tests and distributions
- **numpy**: Numerical computations
- **Custom implementations**: Power-law fitting, overdispersion tests

## Results

### Data Quality
- Comprehensive 6-step quality assurance pipeline implemented
- Missing values, duplicates, and outliers identified and treated
- Data type optimization for memory efficiency
- Final quality validation and certification completed

### Review Processing
- ~3.6M reviews processed from Goodreads dataset
- English-language reviews extracted and filtered
- Sentence-level dataset created for topic modeling (~8.7M sentences)
- Processing rate: ~100-110 reviews/second

### Topic Modeling
- BERTopic models trained with multiple embedding configurations
- Hyperparameter optimization completed
- Topic extraction and analysis in progress
- Results stored in structured format for further analysis

### Corpus Characteristics
- Representative 6,000-book subdataset created
- Balanced across popularity tiers (top, mid, thrash)
- Key demographic characteristics preserved
- Ready for statistical analysis and topic modeling

## Current Status

### Completed
- Data integration and quality assurance pipeline
- Text preprocessing and normalization
- Review extraction and filtering
- Sentence-level dataset preparation
- Topic modeling infrastructure setup
- Hyperparameter optimization framework

### In Progress
- Topic modeling analysis and interpretation
- Statistical analysis of topic distributions
- Correlation analysis between topics and metadata

## Outputs

All research outputs are organized in the `outputs/` directory:
- **Datasets**: Processed datasets at various pipeline stages
- **Reports**: Analysis reports in JSON and Markdown formats
- **Visualizations**: Publication-ready plots and charts
- **Logs**: Execution logs for reproducibility

## Reproducibility

The pipeline is designed for reproducibility:
- All code organized by research stage
- Configuration files for key parameters
- Comprehensive logging of all processing steps
- Documentation for each stage of the pipeline

See `docs/replication_guide.md` for instructions on adapting the pipeline to other datasets.

