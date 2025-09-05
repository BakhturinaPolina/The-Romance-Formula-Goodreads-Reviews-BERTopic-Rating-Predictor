# CSV Building Module

## Overview

The CSV Building Module creates clean, analysis-ready datasets from raw Goodreads JSON data with enhanced null value handling and comprehensive data quality validation.

## Key Features

- **Enhanced Data Processing**: Proper handling of empty strings and null values
- **Data Quality Validation**: Comprehensive quality checks and reporting
- **Performance Optimized**: Efficient processing of large datasets
- **Comprehensive Logging**: Detailed progress tracking and error reporting

## Usage

### Quick Start

```python
from src.csv_building import OptimizedFinalCSVBuilder

# Initialize builder
builder = OptimizedFinalCSVBuilder()

# Process sample (recommended for testing)
output_path = builder.build_final_csv_optimized(sample_size=100)

# Process full dataset
output_path = builder.build_final_csv_optimized()
```

### Command Line Usage

```python
from src.csv_building import OptimizedFinalCSVBuilder

builder = OptimizedFinalCSVBuilder()
output_path = builder.build_final_csv_optimized()
print(f"CSV generated: {output_path}")
```

## Output Structure

The final CSV contains 19 essential columns:

- `work_id`: Unique work identifier
- `title`: Cleaned book title
- `publication_year`: Publication year
- `author_name`: Author name
- `series_title`: Series name (if applicable)
- `num_pages_median`: Median page count
- `description`: Book description
- `genres`: Genre classifications
- `ratings_count_sum`: Total ratings
- `average_rating_weighted_mean`: Weighted average rating
- And 9 additional metadata fields

## Data Quality Features

- **Null Value Handling**: Proper processing of empty strings and missing data
- **Data Validation**: Multiple quality checks before output
- **Error Logging**: Comprehensive error tracking and reporting
- **Quality Metrics**: Detailed statistics on data processing

## Performance

- **Full Dataset**: ~15 minutes for complete dataset
- **Memory Efficient**: Optimized for large file processing
- **Progress Tracking**: Real-time progress updates

## Archive

Legacy files are archived in the `archive/` directory:
- `final_csv_builder_working.py`: Original builder
- `test_null_fix.py`: Test comparison script
- `run_working_builder.py`: Legacy runner
- `README_old.md`: Previous documentation
