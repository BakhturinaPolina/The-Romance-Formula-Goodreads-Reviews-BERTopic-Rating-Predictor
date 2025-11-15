# Final Cleaned and Preprocessed Romance Novel Dataset

## Overview
This directory contains the **final cleaned and preprocessed dataset** for the romance novel NLP research project. This dataset represents the culmination of a comprehensive 6-step data cleaning pipeline applied to raw Goodreads data.

## Dataset Information
- **Records**: 80,755 romance novels
- **Columns**: 30 features
- **Time Period**: 2000-2017 (publication years)
- **Language**: English editions only
- **Quality**: High-quality, research-ready dataset

## Files

### Complete Dataset (All 30 Columns)
- `final_cleaned_preprocessed_dataset.csv` - CSV format (169.7 MB)
- `final_cleaned_preprocessed_dataset.pkl` - Pickle format (170.8 MB)
- `final_cleaned_preprocessed_dataset_20250907_181315.csv` - Timestamped CSV
- `final_cleaned_preprocessed_dataset_20250907_181315.pkl` - Timestamped pickle

### Core Research Dataset (23 Essential Columns)
- `core_research_dataset_corrected_20250907_182917.csv` - CSV format (smaller, research-focused)
- `core_research_dataset_corrected_20250907_182917.pkl` - Pickle format (smaller, research-focused)

### False Duplicates Dataset (Similar Titles by Different Authors)
- `false_duplicates_similar_titles_20250907_182555.csv` - CSV format (21,105 records)
- `false_duplicates_similar_titles_20250907_182555.pkl` - Pickle format (21,105 records)

## Core Research Dataset Columns (23 Essential Features)

### Core Book Information
- `work_id` - Unique work identifier
- `book_id_list_en` - List of different editions of work_id
- `title` - Book title
- `publication_year` - Year of publication (2000-2017)
- `num_pages_median` - Median page count across editions
- `description` - Book description (HTML cleaned, normalized)
- `language_codes_en` - English language codes

### Author Information
- `author_id` - Unique author identifier
- `author_name` - Author name
- `author_average_rating` - Author's average rating
- `author_ratings_count` - Total ratings for author

### Series Information
- `series_id` - Series identifier (if part of series)
- `series_title` - Series name
- `series_works_count` - Number of books in series

### Ratings and Reviews
- `ratings_count_sum` - Total number of ratings
- `text_reviews_count_sum` - Total number of text reviews
- `average_rating_weighted_mean` - Weighted average rating

### Categorical Features
- `popular_shelves` - Popular shelf tags (standardized, lowercase)
- `genres` - Book genres (normalized and categorized)
- `decade` - Publication decade (2000s, 2010s)
- `book_length_category` - Page count categories
- `rating_category` - Rating quality categories
- `popularity_category` - Popularity level categories

## Data Reduction Summary
- **Starting Records**: 119,678 (raw integrated data)
- **Final Records**: 80,755 (cleaned and preprocessed)
- **Total Reduction**: 38,923 records (32.52%)

### Reduction Breakdown
1. **Missing Values Treatment**: 38,874 records removed (32.5%)
   - Missing descriptions: 2,924 records
   - Missing page counts: 35,908 records
   - Missing ratings: 42 records
2. **Outlier Treatment**: 49 records removed (0.06%)
   - Books published after 2017: 49 records

## Dataset Features

### Core Book Information
- `work_id` - Unique work identifier
- `title` - Book title
- `publication_year` - Year of publication (2000-2017)
- `num_pages_median` - Median page count across editions
- `description` - Book description (HTML cleaned, normalized)
- `language_codes_en` - English language codes

### Author Information
- `author_id` - Unique author identifier
- `author_name` - Author name
- `author_average_rating` - Author's average rating
- `author_ratings_count` - Total ratings for author

### Series Information
- `series_id` - Series identifier (if part of series)
- `series_title` - Series name
- `series_works_count` - Number of books in series
- `series_id_missing_flag` - Flag for missing series data
- `series_title_missing_flag` - Flag for missing series title
- `series_works_count_missing_flag` - Flag for missing series count

### Ratings and Reviews
- `ratings_count_sum` - Total number of ratings
- `text_reviews_count_sum` - Total number of text reviews
- `average_rating_weighted_mean` - Weighted average rating

### Categorical Features
- `popular_shelves` - Popular shelf tags (standardized, lowercase)
- `genres` - Book genres (normalized and categorized)
- `decade` - Publication decade (2000s, 2010s)
- `book_length_category` - Page count categories
- `rating_category` - Rating quality categories
- `popularity_category` - Popularity level categories

### Data Quality Flags
- `title_duplicate_flag` - Flag for potential title duplicates
- `author_id_duplicate_flag` - Flag for author ID duplicates
- `text_preprocessing_applied` - Confirmation of text preprocessing
- `text_preprocessing_timestamp` - When text preprocessing was applied

## Data Quality Assurance
- ✅ **Missing Values**: All critical missing values addressed
- ✅ **Duplicates**: Duplicate detection and flagging completed
- ✅ **Data Types**: Optimized data types for memory efficiency
- ✅ **Outliers**: Statistical outliers identified and treated
- ✅ **Text Cleaning**: HTML removed, text normalized
- ✅ **Validation**: Comprehensive quality validation passed

## Usage Recommendations
- **For NLP Research**: Use the cleaned `description` field for text analysis
- **For Statistical Analysis**: All numerical fields are validated and optimized
- **For Categorical Analysis**: Use the derived categorical features
- **For Series Analysis**: Use series-related fields with missing flags
- **For Author Analysis**: Use author fields with duplicate flags

## Pipeline History
This dataset was created through the following pipeline:
1. **Raw JSON Integration** → 119,678 records
2. **Missing Values Treatment** → 80,804 records
3. **Duplicate Detection** → 80,804 records (no duplicates found)
4. **Data Type Validation** → 80,804 records
5. **Outlier Treatment** → 80,755 records
6. **NLP Text Preprocessing** → 80,755 records (final)

## File Formats
- **CSV**: Human-readable, compatible with most tools
- **Pickle**: Preserves data types, faster loading, smaller size

## Next Steps for Research
This dataset is ready for:
- Natural Language Processing analysis
- Statistical modeling
- Machine learning applications
- Academic research and publication
- Comparative analysis with other datasets

---
*Generated on: 2025-09-07*  
*Pipeline Version: Complete 6-Step Data Quality Pipeline*  
*Total Processing Time: ~8 minutes*
