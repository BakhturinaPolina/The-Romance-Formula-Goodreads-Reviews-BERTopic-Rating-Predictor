# Processed Data Directory

This directory contains the final cleaned and processed datasets ready for romance novel NLP research analysis.

## Current Datasets

### Latest Processed Dataset
**File**: `final_books_2000_2020_en_enhanced_titles_20250902_001152.csv`  
**Size**: 119,678 romance novels  
**Generated**: September 2, 2025 at 00:11:52  
**Processing Time**: ~15 minutes  

### Dataset Characteristics
- **Publication Years**: 2000-2020
- **Language**: English editions only
- **Structure**: 19 columns (simplified from 24)
- **Quality**: 95%+ completeness for core fields
- **Format**: Clean, research-ready CSV

## Dataset Structure

### Core Columns (19 total)

| Column | Description | Type | Example | Completeness |
|--------|-------------|------|---------|--------------|
| `work_id` | Unique work identifier | int | 1809886 | 100% |
| `book_id_list_en` | List of English book IDs | list | ['1810549'] | 100% |
| `title` | Cleaned book title | str | "The Enchantment" | 99.99% |
| `publication_year` | Publication year | int | 2001 | 100% |
| `language_codes_en` | Language codes | list | ['eng'] | 100% |
| `num_pages_median` | Median page count | float | 320.0 | 85% |
| `description` | Book description | str | "Pam Binder captures..." | 94.8% |
| `popular_shelves` | Popular shelf tags | str | "romance,historical" | 90% |
| `genres` | Genre classifications | str | "romance,historical" | 75% |
| `author_id` | Primary author ID | str | "12345" | 100% |
| `author_name` | Author name | str | "Pam Binder" | 100% |
| `author_average_rating` | Author average rating | float | 4.2 | 95% |
| `author_ratings_count` | Author ratings count | int | 15000 | 95% |
| `series_id` | Series identifier | str | "67890" | 66.9% |
| `series_title` | Series name | str | "Highland Series" | 66.9% |
| `series_works_count` | Books in series | int | 5 | 66.9% |
| `ratings_count_sum` | Total ratings across editions | int | 2500 | 100% |
| `text_reviews_count_sum` | Total reviews across editions | int | 150 | 100% |
| `average_rating_weighted_mean` | Weighted average rating | float | 4.1 | 100% |

## Data Quality Metrics

### Overall Quality
- **Total Works**: 119,678 romance novels
- **Works with Titles**: 119,667 (99.99%)
- **Works with Descriptions**: 113,506 (94.8%)
- **Works with Series Data**: 80,108 (66.9%)
- **Works with Author Data**: 119,678 (100%)

### Quality Validation Results
✅ **All data quality validation checks passed**  
✅ **No null publication years**  
✅ **No null author IDs**  
✅ **No null work IDs**  
✅ **No empty book ID lists**  

### Data Completeness by Field
- **Core Identifiers**: 100% complete
- **Book Metadata**: 99.99% complete
- **Content Fields**: 94.8% complete
- **Author Information**: 100% complete
- **Series Data**: 66.9% complete (normal for standalone books)
- **Rating Metrics**: 100% complete

## Processing Details

### Data Sources
- **Primary**: `goodreads_books_romance.json.gz` (348MB)
- **Works**: `goodreads_book_works.json.gz` (72MB)
- **Authors**: `goodreads_book_authors.json.gz` (17MB)
- **Series**: `goodreads_book_series.json.gz` (27MB)
- **Genres**: `goodreads_book_genres_initial.json.gz` (23MB)

### Processing Pipeline
1. **Data Loading**: Efficient JSON.gz processing with progress tracking
2. **Language Filtering**: English editions only (197,342 out of 335,449)
3. **Work Grouping**: 135,759 unique works identified
4. **Data Aggregation**: Work-level metrics across multiple editions
5. **Quality Validation**: Comprehensive data quality checks
6. **Output Generation**: Clean, structured CSV with 19 columns

### Aggregation Strategy
- **Work-Level Focus**: Multiple editions aggregated by work_id
- **Weighted Averages**: Ratings weighted by edition popularity
- **Median Calculations**: Page counts and publication years
- **Sum Aggregation**: Total ratings and reviews across editions
- **Quality Preservation**: Best available data selected for each field

## Usage Guidelines

### Research Applications
- **Topic Modeling**: Clean descriptions for NLP analysis
- **Popularity Analysis**: Comprehensive rating and review metrics
- **Genre Studies**: Popular shelves and genre classifications
- **Author Analysis**: Author-level statistics and performance
- **Series Research**: Series structure and progression analysis

### Data Access
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/processed/final_books_2000_2020_en_enhanced_titles_20250902_001152.csv')

# Basic statistics
print(f"Dataset shape: {df.shape}")
print(f"Publication years: {df['publication_year'].min()} - {df['publication_year'].max()}")
print(f"Total works: {len(df):,}")
```

### Column Descriptions

#### Core Identifiers
- **`work_id`**: Unique identifier for each literary work (multiple editions may exist)
- **`book_id_list_en`**: List of English edition book IDs, sorted by popularity

#### Book Metadata
- **`title`**: Cleaned book title with brackets and formatting removed
- **`publication_year`**: Publication year (2000-2020 range)
- **`language_codes_en`**: Language codes for English editions
- **`num_pages_median`**: Median page count across all editions

#### Content
- **`description`**: Longest available book description
- **`popular_shelves`**: Popular shelf tags from Goodreads users
- **`genres`**: Genre classifications from Goodreads

#### Author Information
- **`author_id`**: Primary author identifier
- **`author_name`**: Author's name
- **`author_average_rating`**: Author's average rating across all books
- **`author_ratings_count`**: Total ratings for author's books

#### Series Data
- **`series_id`**: Series identifier (if part of a series)
- **`series_title`**: Series name
- **`series_works_count`**: Total number of works in the series

#### Popularity Metrics
- **`ratings_count_sum`**: Total ratings across all editions
- **`text_reviews_count_sum`**: Total text reviews across all editions
- **`average_rating_weighted_mean`**: Weighted average rating across editions

## Quality Assurance

### Validation Checks Applied
1. **Required Fields**: All core fields must be present
2. **Data Types**: Proper numeric and string formatting
3. **Value Ranges**: Publication years within 2000-2020
4. **Referential Integrity**: Author and series references valid
5. **Content Quality**: Non-empty descriptions and titles

### Quality Reports
Each dataset includes a quality report with:
- Processing statistics and timing
- Data quality metrics
- Validation results
- Error summaries (if any)

## File Naming Convention

### Format
```
final_books_2000_2020_en_enhanced_titles_[timestamp].csv
```

### Components
- **`final_books`**: Indicates processed, final dataset
- **`2000_2020`**: Publication year range
- **`en`**: English language filter
- **`enhanced_titles`**: Title processing applied
- **`[timestamp]`**: Generation timestamp (YYYYMMDD_HHMMSS)

### Examples
- `final_books_2000_2020_en_enhanced_titles_20250902_001152.csv` - Full dataset
- `final_books_2000_2020_en_enhanced_titles_sampled_100_20250901_235256.csv` - Sample dataset

## Archive and Versioning

### Previous Versions
- **24-column versions**: Archived due to redundant columns
- **Intermediate processing**: Removed for cleaner output
- **Debug information**: Consolidated into quality reports

### Current Approach
- **Single source of truth**: One clean, working dataset
- **Simplified structure**: Essential columns only
- **Quality focus**: Research-ready data
- **Performance optimized**: Fast processing and loading

## Maintenance

### Regular Updates
- **Quarterly processing**: Regenerate datasets with latest data
- **Quality monitoring**: Track data quality metrics over time
- **Performance optimization**: Improve processing efficiency
- **Documentation updates**: Keep this README current

### Backup Strategy
- **Version control**: All datasets tracked in git
- **Quality reports**: Comprehensive processing documentation
- **Archive preservation**: Previous versions maintained
- **Data integrity**: Checksums and validation results

---

**Last Updated**: September 2025  
**Dataset Version**: 1.0.0 (Simplified Structure)  
**Status**: Production Ready  
**Quality**: Validated and Research Ready
