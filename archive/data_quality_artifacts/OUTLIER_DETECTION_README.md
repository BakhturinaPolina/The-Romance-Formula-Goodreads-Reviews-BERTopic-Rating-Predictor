# Step 3: Outlier Detection and Treatment

## Overview
This module implements comprehensive outlier detection and treatment analysis for the romance novel dataset, addressing data quality issues identified in Step 2.

## Research Impact
Step 3 directly supports your research objectives:
- **RQ1 (Topic Modeling)**: Title duplicates may affect theme extraction accuracy
- **RQ2 (Review Analysis)**: Series inconsistencies may impact series-level analysis  
- **RQ3 (Correlation Analysis)**: Data quality issues identified for correction
- **RQ4 (Author vs. Reader Themes)**: Author consistency validated

## Analysis Components

### 1. Title Duplication Analysis
- **Purpose**: Investigate legitimate duplicate titles vs. data errors
- **Method**: Compare author, publication year, and series information
- **Output**: Classification of duplicates and disambiguation strategies

### 2. Series Data Cleaning
- **Purpose**: Validate series_works_count accuracy and identify missing books
- **Method**: Compare expected vs. actual book counts per series
- **Output**: Discrepancy report and correction recommendations

### 3. Statistical Outlier Detection
- **Purpose**: Identify anomalies in numerical fields
- **Method**: IQR-based outlier detection (1.5 * IQR rule)
- **Fields**: Publication year, rating, review count, page count

## Files

### Main Scripts
- `outlier_detection_analysis.py` - Core analysis implementation
- `run_outlier_detection.py` - Execution runner with logging

### Outputs
- `outputs/outlier_detection/` - Analysis results and visualizations
- `logs/` - Execution logs with timestamps

## Usage

### Quick Execution
```bash
cd src/data_quality
python run_outlier_detection.py
```

### Programmatic Usage
```python
from outlier_detection_analysis import OutlierDetectionAnalyzer

# Initialize analyzer
analyzer = OutlierDetectionAnalyzer()

# Run complete analysis
results = analyzer.run_complete_analysis()

# Access specific results
title_dups = results['title_duplications']
series_data = results['series_data']
outliers = results['statistical_outliers']
recommendations = results['treatment_recommendations']
```

## Expected Outputs

### 1. Analysis Report
- **File**: `step3_outlier_detection_report_YYYYMMDD_HHMMSS.txt`
- **Content**: Comprehensive analysis results with treatment recommendations

### 2. Results JSON
- **File**: `step3_outlier_detection_results_YYYYMMDD_HHMMSS.json`
- **Content**: Structured data for programmatic access

### 3. Visualizations
- **File**: `outlier_detection_visualizations_YYYYMMDD_HHMMSS.png`
- **Content**: Distribution plots for all analyzed fields

### 4. Execution Logs
- **File**: `logs/outlier_detection_YYYYMMDD_HHMMSS.log`
- **Content**: Detailed execution trace and error information

## Analysis Methods

### Title Duplication Logic
```python
# Legitimate duplicate criteria:
if unique_authors > 1:
    is_legitimate = True  # Different authors
elif year_range > 5:
    is_legitimate = True  # Different publication years
elif unique_series > 1:
    is_legitimate = True  # Different series
else:
    is_legitimate = False  # Potential error
```

### Outlier Detection
```python
# IQR method (1.5 * IQR rule)
Q1 = series.quantile(0.25)
Q3 = series.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = series[(series < lower_bound) | (series > upper_bound)]
```

## Treatment Recommendations

### Priority Levels
- **High**: Data errors requiring immediate attention
- **Medium**: Quality issues affecting analysis accuracy
- **Low**: Minor inconsistencies for future improvement

### Common Actions
1. **Manual Review**: Investigate suspicious duplicates
2. **Data Correction**: Fix series_works_count discrepancies
3. **Outlier Investigation**: Validate extreme values
4. **Series Completion**: Add missing series books

## Data Requirements

### Required Columns
- `title` - Book title for duplication analysis
- `author` - Author name for duplicate validation
- `publication_year` - Publication year for outlier detection
- `series` - Series information for completeness analysis
- `book_id` - Unique identifier for book tracking

### Optional Columns
- `series_works_count` - Expected series length
- `rating` - Book rating for outlier detection
- `review_count` - Review count for outlier detection
- `page_count` - Page count for outlier detection

## Error Handling

### Common Issues
1. **Missing Data**: Graceful handling of NaN values
2. **Column Mismatches**: Validation of expected columns
3. **Memory Issues**: Efficient processing of large datasets
4. **File I/O**: Robust error handling for file operations

### Recovery Strategies
- Log detailed error information
- Continue analysis with available data
- Provide partial results when possible
- Suggest corrective actions

## Performance Considerations

### Optimization Features
- Efficient pandas operations
- Minimal memory footprint
- Streaming-friendly data processing
- Configurable analysis depth

### Scalability
- Handles datasets up to 100K+ records
- Memory-efficient outlier detection
- Parallel processing ready
- Incremental analysis support

## Future Enhancements

### Planned Features
1. **Machine Learning Outliers**: Advanced anomaly detection
2. **Interactive Visualizations**: Plotly-based dashboards
3. **Automated Corrections**: AI-powered data fixing
4. **Real-time Monitoring**: Continuous quality assessment

### Integration Points
- **Data Pipeline**: Seamless integration with existing workflow
- **Quality Metrics**: Automated quality scoring
- **Alert System**: Notifications for critical issues
- **Version Control**: Track data quality improvements

## Troubleshooting

### Common Problems
1. **Import Errors**: Check Python path and dependencies
2. **Memory Issues**: Reduce dataset size or use sampling
3. **Visualization Errors**: Verify matplotlib/seaborn installation
4. **File Permission Errors**: Check output directory permissions

### Debug Mode
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Dependencies

### Required Packages
- `pandas` >= 1.3.0
- `numpy` >= 1.20.0
- `matplotlib` >= 3.3.0
- `seaborn` >= 0.11.0

### Installation
```bash
pip install pandas numpy matplotlib seaborn
```

## Contact
For questions or issues with the outlier detection analysis, refer to the project documentation or create an issue in the project repository.
