# Data Exploration Module

## Overview

The Data Exploration module contains tools and scripts for initial data inspection, schema analysis, and exploratory data analysis of the Goodreads romance dataset. These tools help understand the structure, quality, and characteristics of the raw data before processing.

## Components

### 📋 JSON Structure Inspector (`inspect_json_structure.py`)

**Purpose**: Comprehensive analysis of JSON file structures with detailed reporting.

**Features**:
- Analyzes all JSON files in the dataset
- Extracts field information, data types, and missing value patterns
- Generates HTML reports with detailed statistics
- Provides sample records for each file
- Identifies data quality issues and anomalies

**Usage**:
```python
from src.data_exploration.inspect_json_structure import JSONStructureInspector
from config.settings import DATA_FILES

# Initialize inspector
inspector = JSONStructureInspector(DATA_FILES)

# Analyze all files
inspector.inspect_all_files()

# Generate HTML report
inspector.generate_html_report()
```

**Output**:
- Console output with summary statistics
- HTML report with detailed analysis
- JSON files with structure information
- Data quality assessment

### 📅 Publication Year Explorer (`explore_publication_years.py`)

**Purpose**: Analysis of publication year distributions and data quality issues.

**Features**:
- Examines publication year field in books data
- Identifies missing or invalid publication years
- Analyzes year distribution patterns
- Provides data quality insights

**Usage**:
```bash
# Run the script directly
python src/data_exploration/explore_publication_years.py
```

**Output**:
- Console output showing publication year patterns
- Data type analysis for publication_year field
- Sample records with publication year information

### 📚 Edition Aggregation Explorer (`explore_edition_aggregation.py`)

**Purpose**: Investigation of edition aggregation patterns and work-level grouping.

**Features**:
- Analyzes works with multiple editions
- Compares individual edition vs. aggregated ratings
- Identifies patterns in edition aggregation
- Provides insights into work-level data quality

**Usage**:
```bash
# Run the script directly
python src/data_exploration/explore_edition_aggregation.py
```

**Output**:
- Console output showing edition aggregation statistics
- Sample records of works with multiple editions
- Comparison of individual vs. aggregated metrics

## Dependencies

### Internal Dependencies
- `config/settings.py`: Data file paths and configuration
- `config/logging_config.py`: Logging setup
- `src/utils/file_handlers.py`: File reading utilities

### External Dependencies
- `pandas`: Data manipulation and analysis
- `json`: JSON processing
- `gzip`: Compressed file handling
- `pathlib`: File path operations
- `collections`: Data structures for analysis
- `tqdm`: Progress bars for long operations

## Data Files Analyzed

The exploration tools work with the following Goodreads dataset files:

- `goodreads_books_romance.json.gz`: Romance books metadata
- `goodreads_reviews_romance.json.gz`: Romance book reviews
- `goodreads_book_authors.json.gz`: Author information
- `goodreads_book_genres_initial.json.gz`: Genre classifications
- `goodreads_book_series.json.gz`: Series information
- `goodreads_book_works.json.gz`: Work-level aggregations
- `goodreads_interactions_romance.json.gz`: User interactions
- `goodreads_reviews_dedup.json.gz`: Deduplicated reviews
- `goodreads_reviews_spoiler.json.gz`: Spoiler-tagged reviews

## Output Locations

### Reports and Analysis
- **HTML Reports**: Generated in `outputs/reports/` directory
- **JSON Structure Files**: Saved in `data/intermediate/schemas/`
- **Console Output**: Real-time analysis results

### Logs
- **Exploration Logs**: Written to `logs/exploration/` directory
- **Error Logs**: Detailed error information in `logs/error_logs/`

## Usage Patterns

### Initial Data Assessment
1. Run `inspect_json_structure.py` to understand all data files
2. Review HTML report for comprehensive overview
3. Identify potential data quality issues

### Publication Year Analysis
1. Run `explore_publication_years.py` to check year data quality
2. Review console output for patterns
3. Note any data quality issues for processing pipeline

### Edition Aggregation Analysis
1. Run `explore_edition_aggregation.py` after processing books data
2. Review aggregation patterns
3. Validate work-level grouping logic

## Best Practices

### Running Exploration Tools
1. **Start with Structure Inspection**: Always run the JSON structure inspector first
2. **Review Reports**: Carefully examine HTML reports for insights
3. **Document Findings**: Note any data quality issues or patterns
4. **Iterate**: Run exploration tools multiple times as data changes

### Interpreting Results
1. **Focus on Data Quality**: Pay attention to missing values and data types
2. **Look for Patterns**: Identify trends in publication years, ratings, etc.
3. **Validate Assumptions**: Use exploration results to validate processing decisions
4. **Document Insights**: Record findings for future reference

### Performance Considerations
1. **Use Sampling**: For large files, use sampling to speed up analysis
2. **Monitor Memory**: Large JSON files can consume significant memory
3. **Progress Tracking**: Use progress bars for long-running operations
4. **Error Handling**: Implement robust error handling for file reading

## Troubleshooting

### Common Issues
- **File Not Found**: Verify data file paths in `config/settings.py`
- **Memory Errors**: Use smaller sample sizes for large files
- **JSON Parse Errors**: Check for malformed JSON in source files
- **Permission Errors**: Ensure read access to data files

### Debugging Tips
- Check logs in `logs/exploration/` for detailed error information
- Use smaller sample sizes for testing
- Verify file paths and permissions
- Review console output for specific error messages

## Future Enhancements

### Planned Features
1. **Interactive Visualizations**: Add plotting capabilities for data distributions
2. **Statistical Analysis**: Include statistical tests for data quality
3. **Automated Reporting**: Generate automated data quality reports
4. **Integration with Pipeline**: Connect exploration results to processing decisions

### Potential Improvements
1. **Parallel Processing**: Implement parallel analysis for large datasets
2. **Caching**: Add caching for repeated analysis operations
3. **Configuration**: Make exploration parameters configurable
4. **API Integration**: Add programmatic access to exploration results
