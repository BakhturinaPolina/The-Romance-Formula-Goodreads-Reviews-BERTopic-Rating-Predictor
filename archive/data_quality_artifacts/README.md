# Data Quality Module

## Overview

The Data Quality Module implements **Step 1** of the comprehensive EDA plan for the Romance Novel NLP Research project. This module provides comprehensive data quality assessment focusing on missing values analysis and data type validation.

## Purpose

This module addresses the critical first step in data preparation:
- **Missing Values Analysis**: Comprehensive assessment of data completeness
- **Data Type Validation**: Validation of critical variables for analysis
- **NLP Readiness Assessment**: Evaluation of dataset suitability for text analysis
- **Quality Reporting**: Detailed reports for research decision-making

## Implementation

### Coding Agent Pattern

The module follows the **Coding Agent Pattern** as defined in the project rules:

- **Code Analyzer**: Analyzes dataset structure and quality issues
- **Change Planner**: Plans data quality improvements and validations
- **Code Modifier**: Implements data validation and quality checks
- **Test Runner**: Validates data quality through systematic checks

### Key Components

#### 1. DataQualityAssessment Class

**Main Class**: `DataQualityAssessment`

**Core Methods**:
- `analyze_missing_values()`: Missing values analysis for key fields
- `validate_data_types()`: Data type validation for critical variables
- `assess_nlp_readiness()`: NLP analysis readiness assessment
- `generate_quality_summary()`: Comprehensive quality reporting

#### 2. Missing Values Analysis

**Fields Analyzed**:
- `num_pages_median`: Page count completeness
- `description`: Text content availability (critical for NLP)
- `series_id`: Series information coverage
- `genres`: Genre classification completeness

**Expected Coverage**:
- Series data: 67% coverage expected
- Descriptions: Critical for NLP analysis
- Genres: Essential for subgenre analysis

#### 3. Data Type Validation

**Critical Variables**:
- `publication_year`: Integer (2000-2020 range)
- `ratings_count_sum`: Non-negative integer
- `text_reviews_count_sum`: Non-negative integer
- `average_rating_weighted_mean`: Float (0-5 range)

**Validation Rules**:
- Type checking against expected data types
- Range validation for bounded variables
- Minimum value validation for count fields

#### 4. NLP Readiness Assessment

**Assessment Criteria**:
- Description availability and quality
- Text length analysis (minimum 50 characters)
- Genre coverage for subgenre analysis
- Exclusion criteria identification

## Usage

### Command Line Interface

```bash
# Navigate to module directory
cd src/data_quality

# Run assessment on latest dataset
python run_quality_assessment.py

# Run assessment on specific dataset
python run_quality_assessment.py final_books_2000_2020_en_enhanced_titles_20250902_001152.csv
```

### Programmatic Usage

```python
from src.data_quality import DataQualityAssessment

# Initialize assessor
assessor = DataQualityAssessment()

# Run full assessment
results = assessor.run_full_assessment()

# Access specific results
missing_analysis = results['missing_values']
type_validation = results['data_type_validation']
nlp_readiness = results['nlp_readiness']
```

### Output

The module generates:
1. **Console Output**: Real-time progress and key findings
2. **Quality Report**: Detailed text report saved to `data/processed/`
3. **Structured Results**: Python dictionary with all assessment data

## Quality Metrics

### Missing Values Tracking
- Count and percentage of missing values per field
- Completeness metrics for research planning
- NLP analysis exclusion flags

### Data Type Validation
- Type compliance checking
- Range validation results
- Error categorization and reporting

### NLP Readiness
- Books ready for text analysis
- Exclusion reasons and counts
- Description quality statistics

## Research Alignment

### Step 1 of EDA Plan
This module directly implements the first step of the comprehensive EDA plan:
- **Missing Values Handling**: Identifies data gaps for research planning
- **Data Type Validation**: Ensures analysis-ready data structure
- **NLP Preparation**: Prepares dataset for Phase 3 topic modeling

### Research Questions Support
- **RQ1 (Topic Modeling)**: Assesses text content availability
- **RQ2 (Review Analysis)**: Validates rating and review data quality
- **RQ3 (Correlation Analysis)**: Ensures numerical data integrity
- **RQ4 (Author vs. Reader Themes)**: Validates author and genre data

## Next Steps

After completing Step 1:
1. **Step 2**: Duplicate and Inconsistency Detection
2. **Step 3**: Outlier Detection and Treatment
3. **Step 4**: Variable Transformations and Feature Engineering

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **pathlib**: File path handling
- **logging**: Comprehensive logging system

## Error Handling

The module implements robust error handling:
- Graceful failure with detailed error messages
- Validation error categorization
- Rollback support for failed operations
- Comprehensive logging for debugging

## Performance

- **Efficient Loading**: Automatic dataset discovery and loading
- **Memory Management**: Optimized for large datasets
- **Progress Tracking**: Real-time progress updates
- **Batch Processing**: Handles large datasets efficiently

---

**Module Version**: 1.0.0  
**Last Updated**: September 2025  
**Status**: Production Ready  
**Next Phase**: Step 2 - Duplicate and Inconsistency Detection
