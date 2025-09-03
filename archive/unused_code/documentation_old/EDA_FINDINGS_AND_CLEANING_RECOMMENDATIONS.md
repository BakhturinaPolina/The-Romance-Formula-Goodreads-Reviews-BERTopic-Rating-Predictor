# EDA Findings and Cleaning Recommendations

## Executive Summary

This document summarizes the comprehensive Exploratory Data Analysis (EDA) of the final processed romance novel dataset and provides specific recommendations for data cleaning before NLP analysis.

**Dataset**: `final_books_2000_2020_en_20250901_024106.csv`  
**Size**: 119,678 romance novels  
**Coverage**: 2000-2019 publication years  
**Quality**: Good overall with specific areas needing attention

## Key Findings

### 1. Dataset Overview
- **Total Records**: 119,678 romance novels
- **Columns**: 20 fields including metadata, popularity metrics, and text content
- **Publication Range**: 2000-2019 (most common year: 2014 with 19,967 books)
- **Memory Usage**: 278 MB

### 2. Data Quality Assessment

#### Missing Values Analysis
- **median_publication_year**: 93.0% missing (111,301 records)
- **series_title**: 33.1% missing (39,572 records)
- **series_works_count**: 33.1% missing (39,570 records)
- **series_id**: 33.1% missing (39,570 records)
- **num_pages_median**: 30.0% missing (35,908 records)
- **description**: 5.2% missing (6,172 records)
- **average_rating_weighted_mean**: 0.1% missing (96 records)

#### Data Completeness
- **Core Fields**: 100% complete (work_id, title, author_id, author_name, publication_year)
- **Popularity Metrics**: 99.9% complete (ratings, reviews, average ratings)
- **Series Information**: 66.9% complete
- **Descriptions**: 94.8% complete

### 3. Title Analysis

#### Title Characteristics
- **Length**: Mean 27.7 characters, median 25.0 characters
- **Word Count**: Mean 4.7 words, median 4.0 words
- **Range**: 1-229 characters, 1-39 words

#### Series Patterns in Titles
- **Number:Colon**: 1,858 titles (1.6%) - e.g., "Irish trilogy collection (Gallaghers of Ardmore / Irish Trilogy #1-3)"
- **Book/Volume/Part**: 2,570 titles (2.1%) - e.g., "Seducing the Bride (Brides of Mayfair Series Book 1)"
- **Ordinal**: 100 titles (0.1%) - e.g., "Deadly Fall (The 19th Precinct, #1)"
- **End Number**: 2,047 titles (1.7%) - e.g., "kimihapetsuto 1"
- **Number(Parenthesis**: 1,059 titles (0.9%) - e.g., "Take 5 (Volume 2)"

**Total titles with series patterns**: ~8% of dataset

### 4. Author Name Analysis

#### Author Statistics
- **Unique Authors**: 26,799
- **Books per Author**: Mean 4.5, median 2.0
- **Name Length**: Mean 12.8 characters, median 12.0 characters
- **Word Count**: Mean 2.1 words, median 2.0 words

#### Author Name Issues
- **Multiple Name Variations**: 0 authors with multiple names per ID
- **Potential Duplicates**: 14,634 books with same author name but different IDs
- **Examples**: 'Sophie Kinsella', 'Nicholas Sparks', 'Julia Quinn', 'Nora Roberts'

### 5. Description Text Analysis

#### Description Characteristics
- **Length**: Mean 961.2 characters, median 893.0 characters
- **Word Count**: Mean 168.8 words, median 157.0 words
- **Range**: 3-22,742 characters
- **Missing**: 6,172 descriptions (5.2%)
- **Very Short**: 186 descriptions <50 characters (0.2%)

#### Text Quality Issues
- **HTML Tags**: 13 descriptions (0.0%)
- **HTML Entities**: 55 descriptions (0.0%)
- **Multiple Whitespace**: 113,499 descriptions (94.8%)
- **Line Breaks/Tabs**: 100,116 descriptions (83.7%)
- **Non-ASCII Characters**: 0 descriptions

### 6. Series Pattern Analysis

#### Series Coverage
- **Books in Series**: 80,108 (66.9%)
- **Books Not in Series**: 39,570 (33.1%)

#### Series Size Distribution
- **2 books**: 9,524 series
- **3 books**: 14,239 series
- **4 books**: 12,143 series
- **5 books**: 8,248 series
- **6 books**: 6,531 series

#### Series Title Embedding
- **Books with embedded series titles**: 54,053 (67.5% of series books)
- **Examples**:
  - Series: 'Shopaholic' | Book: 'Confessions of a Shopaholic'
  - Series: 'Highlander' | Book: 'The Highlander's Touch'
  - Series: 'In Death' | Book: 'Judgment in Death'

### 7. Publication & Popularity Analysis

#### Publication Trends
- **Peak Year**: 2014 (19,967 books)
- **Recent Years**: 2019 only has 3 books (data collection cutoff)
- **Distribution**: Relatively even across 2000-2018

#### Popularity Metrics
- **Average Rating**: Mean 3.88, median 3.90
- **Ratings Count**: Mean 939, median 113
- **Reviews Count**: Mean 81, median 20

### 8. Subgenre Signal Analysis

#### Popular Shelves Format
- **JSON Format**: 0% (all are comma-separated strings)
- **Structure**: Comma-separated shelf names with various formats

#### Subgenre Classification
- **Target Subgenres**: contemporary romance, historical romance, paranormal romance, romantic suspense, romantic fantasy, science fiction romance
- **Classification Method**: Parse popular shelves for subgenre keywords
- **Coverage**: Varies by subgenre (analysis needed for specific counts)

## Cleaning Recommendations

### 1. Title Normalization (Priority: High)

#### Actions Required
- Extract series numbers from titles using regex patterns
- Remove embedded series titles from book titles
- Standardize numbering formats (Book 1, Volume 2, etc.)
- Create separate columns for extracted series information

#### Implementation
```python
def clean_title(title, series_title=None):
    # Remove series title if embedded
    # Extract series numbers
    # Standardize formatting
    pass
```

### 2. Author Name Normalization (Priority: High)

#### Actions Required
- Resolve 14,634 potential duplicate author names
- Standardize name formats (Title case)
- Create normalized author name column
- Flag potential duplicates for manual review

#### Implementation
```python
def normalize_author_names(df):
    # Create normalized names
    # Identify duplicates
    # Flag for review
    pass
```

### 3. Description Text Cleaning (Priority: High)

#### Actions Required
- Remove HTML tags (13 descriptions)
- Clean HTML entities (55 descriptions)
- Normalize whitespace (94.8% affected)
- Handle line breaks and tabs (83.7% affected)
- Preserve original descriptions for comparison

#### Implementation
```python
def clean_description(description):
    # Remove HTML
    # Normalize whitespace
    # Handle special characters
    pass
```

### 4. Series Information Standardization (Priority: Medium)

#### Actions Required
- Clean series titles (remove extra whitespace, standardize case)
- Extract series positions from titles where missing
- Create consistent series numbering
- Handle missing series information (33% missing)

#### Implementation
```python
def standardize_series_info(df):
    # Clean series titles
    # Extract positions
    # Fill missing information
    pass
```

### 5. Subgenre Classification (Priority: Medium)

#### Actions Required
- Parse comma-separated popular shelves
- Apply keyword-based classification for target subgenres
- Create binary flags for each subgenre
- Assign primary subgenre based on strongest signal

#### Implementation
```python
def classify_subgenres(df):
    # Parse shelves
    # Apply classification rules
    # Create subgenre columns
    pass
```

### 6. Data Quality Improvements (Priority: Low)

#### Actions Required
- Fill missing median_publication_year using publication_year
- Handle missing num_pages_median (30% missing)
- Create data quality score for each record
- Document data quality metrics

## Expected Outcomes

### Data Quality Improvements
- **Title Consistency**: 100% of titles cleaned and normalized
- **Author Deduplication**: Resolve 14,634 potential duplicates
- **Text Cleanliness**: Remove HTML artifacts and normalize formatting
- **Series Information**: 100% of series books have clean titles and positions
- **Subgenre Coverage**: Classify all books into target subgenres

### New Columns Added
- `title_original`, `title_cleaned`
- `series_number_extracted`, `series_position`
- `author_name_normalized`, `author_potential_duplicate`
- `description_original`, `description_cleaned`
- `subgenre_*` (binary flags for each subgenre)
- `subgenre_primary`
- `data_quality_score`

### File Size Impact
- **Original**: 233 MB
- **Cleaned**: Estimated 280-300 MB (additional columns)
- **Storage**: ~30% increase due to new columns and original data preservation

## Risk Assessment

### Low Risk
- Title cleaning (regex-based, reversible)
- Description cleaning (HTML removal, whitespace normalization)
- Series standardization (formatting only)

### Medium Risk
- Author name normalization (potential for over-aggressive deduplication)
- Subgenre classification (keyword-based approach may miss nuances)

### High Risk
- Series title extraction (may remove meaningful information)
- Data quality scoring (subjective criteria)

## Success Metrics

### Quantitative
- 100% of titles cleaned and normalized
- 95%+ of descriptions cleaned
- 90%+ of series information standardized
- 80%+ of books classified into subgenres
- Data quality score >0.8 on average

### Qualitative
- Improved readability of titles and descriptions
- Consistent series information
- Clear subgenre classifications
- Comprehensive documentation

## Next Steps

1. **Review Recommendations**: Stakeholder approval of cleaning approach
2. **Implement Pipeline**: Use provided `data_cleaning_pipeline.py` script
3. **Test on Sample**: Validate cleaning logic on subset before full run
4. **Monitor Progress**: Track cleaning statistics and quality metrics
5. **Generate Reports**: Create detailed cleaning reports for documentation
6. **Prepare for NLP**: Ensure cleaned dataset is ready for topic modeling
