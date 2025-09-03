# Data Validation Module

## Overview

The Data Validation module provides comprehensive data quality assurance and validation tools for the Romance Novel NLP Research project. It implements a multi-layer validation framework that checks processed CSV datasets for quality issues, anomalies, and potential problems.

## Validation Framework

### 🔍 Data Sanity Checker (`data_sanity_checker.py`)

**Purpose**: Comprehensive validation for processed CSV datasets with detailed reporting.

**Key Features**:
- Multi-dataset validation (books, reviews, subgenre)
- Data type and format checking
- Missing value analysis
- Statistical anomaly detection
- Cross-dataset consistency validation
- Automated issue reporting

**Validation Checks**:
- **Data Completeness**: Required field presence and missing value patterns
- **Data Types**: Field type validation and conversion verification
- **Value Ranges**: Numeric field range checking and outlier detection
- **Consistency**: Cross-field and cross-dataset consistency validation
- **Relationships**: Foreign key relationships and referential integrity
- **Statistical Anomalies**: Unusual patterns and potential data quality issues

**Usage**:
```python
from src.data_validation.data_sanity_checker import DataSanityChecker

# Initialize checker
checker = DataSanityChecker("data/processed")

# Run comprehensive validation
results = checker.run_comprehensive_validation()

# Access validation results
summary = results['summary']
issues = results['issues']
detailed_results = results['detailed_results']
```

**Output**:
- Comprehensive validation report
- Issue categorization and severity levels
- Statistical summaries for each dataset
- Recommendations for data quality improvement

### 🔬 Deep Data Inspector (`deep_data_inspector.py`)

**Purpose**: Advanced data inspection for invisible issues and subtle quality problems.

**Key Features**:
- Invisible issue detection
- Pattern analysis and anomaly identification
- Data quality scoring
- Detailed inspection reports
- Machine learning-based anomaly detection

**Inspection Areas**:
- **Text Quality**: Review text analysis and quality assessment
- **Rating Patterns**: Rating distribution and anomaly detection
- **Temporal Patterns**: Publication year and review date analysis
- **Author Patterns**: Author consistency and quality assessment
- **Genre Patterns**: Genre classification quality and consistency
- **Relationship Analysis**: Book-review and author-book relationship quality

**Usage**:
```python
from src.data_validation.deep_data_inspector import DeepDataInspector

# Initialize inspector
inspector = DeepDataInspector("data/processed")

# Run deep inspection
inspection_results = inspector.run_deep_inspection()

# Generate detailed report
inspector.generate_inspection_report()
```

**Output**:
- Deep inspection report with quality scores
- Pattern analysis and anomaly identification
- Data quality recommendations
- Statistical analysis of data characteristics

### 🚀 Validation Runner (`run_data_validation.py`)

**Purpose**: Execution script for automated data validation with reporting.

**Key Features**:
- Automated validation execution
- Comprehensive reporting
- Error handling and recovery
- Integration with processing pipeline
- Email notifications for critical issues

**Usage**:
```bash
# Run validation directly
python src/data_validation/run_data_validation.py

# Run with custom output directory
python src/data_validation/run_data_validation.py --output-dir data/processed
```

**Output**:
- Validation execution logs
- Summary reports
- Detailed issue reports
- Quality metrics and statistics

### 🔬 Deep Inspection Runner (`run_deep_inspection.py`)

**Purpose**: Execution script for deep data inspection and quality analysis.

**Key Features**:
- Automated deep inspection execution
- Quality scoring and assessment
- Pattern analysis and reporting
- Integration with validation framework

**Usage**:
```bash
# Run deep inspection directly
python src/data_validation/run_deep_inspection.py

# Run with custom parameters
python src/data_validation/run_deep_inspection.py --output-dir data/processed
```

**Output**:
- Deep inspection execution logs
- Quality assessment reports
- Pattern analysis results
- Recommendations for data improvement

## Validation Categories

### Data Completeness Validation
- **Required Fields**: Check presence of essential fields
- **Missing Values**: Analyze missing value patterns and percentages
- **Field Coverage**: Verify field population rates
- **Data Completeness Score**: Overall completeness assessment

### Data Type Validation
- **Type Consistency**: Verify data types match expected schemas
- **Type Conversion**: Validate successful type conversions
- **Format Compliance**: Check data format adherence
- **Type Anomalies**: Identify unexpected data types

### Value Range Validation
- **Numeric Ranges**: Check numeric field value ranges
- **Date Ranges**: Validate date field ranges and consistency
- **Text Lengths**: Analyze text field length distributions
- **Outlier Detection**: Identify statistical outliers

### Consistency Validation
- **Cross-Field Consistency**: Check relationships between fields
- **Cross-Dataset Consistency**: Validate relationships across datasets
- **Referential Integrity**: Verify foreign key relationships
- **Business Logic**: Validate business rule compliance

### Quality Assessment
- **Data Quality Score**: Overall quality assessment
- **Issue Severity**: Categorize issues by severity level
- **Quality Trends**: Track quality over time
- **Improvement Recommendations**: Suggest quality improvements

## Validation Reports

### Summary Reports
- **Executive Summary**: High-level quality assessment
- **Issue Summary**: Categorized issue counts and severity
- **Quality Metrics**: Key quality indicators and scores
- **Recommendations**: Actionable improvement suggestions

### Detailed Reports
- **Field-Level Analysis**: Detailed analysis of each field
- **Issue Details**: Specific issue descriptions and examples
- **Statistical Analysis**: Statistical summaries and distributions
- **Pattern Analysis**: Data pattern identification and analysis

### Technical Reports
- **Validation Logs**: Detailed validation execution logs
- **Error Reports**: Error details and stack traces
- **Performance Metrics**: Validation performance statistics
- **Configuration Details**: Validation configuration and settings

## Integration with Pipeline

### Validation Points
- **Post-Processing**: Validate datasets after processing pipeline
- **Quality Gates**: Implement quality gates in processing pipeline
- **Continuous Validation**: Run validation during processing
- **Final Validation**: Comprehensive validation before dataset release

### Quality Thresholds
- **Completeness Thresholds**: Minimum required field completion rates
- **Consistency Thresholds**: Maximum allowed inconsistency rates
- **Quality Score Thresholds**: Minimum acceptable quality scores
- **Issue Severity Thresholds**: Maximum allowed critical issues

## Configuration

### Validation Rules
```yaml
validation_rules:
  completeness:
    min_required_fields: 0.95
    max_missing_rate: 0.05
    
  consistency:
    max_inconsistency_rate: 0.02
    min_referential_integrity: 0.98
    
  quality:
    min_quality_score: 0.85
    max_critical_issues: 0
```

### Issue Severity Levels
- **Critical**: Issues that prevent data use
- **High**: Issues that significantly impact analysis
- **Medium**: Issues that may affect some analyses
- **Low**: Minor issues with minimal impact
- **Info**: Informational issues for awareness

## Performance Characteristics

### Validation Performance
- **Processing Time**: ~1-3 minutes for comprehensive validation
- **Memory Usage**: Efficient memory usage for large datasets
- **Scalability**: Handles datasets up to 100,000+ records
- **Parallel Processing**: Parallel validation for independent checks

### Optimization Features
- **Incremental Validation**: Validate only changed data
- **Caching**: Cache validation results for repeated checks
- **Sampling**: Use sampling for large dataset validation
- **Progress Tracking**: Real-time progress monitoring

## Error Handling

### Error Types
- **File Access Errors**: Missing or inaccessible data files
- **Data Format Errors**: Malformed or invalid data formats
- **Configuration Errors**: Invalid validation configuration
- **Memory Errors**: Insufficient memory for large datasets
- **Processing Errors**: Errors during validation processing

### Recovery Strategies
- **Graceful Degradation**: Continue validation with available data
- **Error Reporting**: Comprehensive error reporting and logging
- **Retry Mechanisms**: Automatic retry for transient errors
- **Fallback Validation**: Simplified validation for problematic data

## Monitoring and Alerting

### Quality Metrics
- **Data Quality Score**: Overall quality assessment
- **Issue Counts**: Number of issues by severity level
- **Validation Performance**: Processing time and resource usage
- **Quality Trends**: Quality changes over time

### Alerting
- **Critical Issues**: Immediate alerts for critical quality issues
- **Quality Thresholds**: Alerts when quality falls below thresholds
- **Performance Issues**: Alerts for validation performance problems
- **Trend Analysis**: Alerts for quality degradation trends

## Testing

### Test Coverage
- **Unit Tests**: Individual validation function tests
- **Integration Tests**: End-to-end validation workflow tests
- **Performance Tests**: Validation performance benchmarks
- **Error Handling Tests**: Error scenario testing

### Test Files
- `tests/data_validation/test_data_sanity_checker.py`
- `tests/data_validation/test_deep_data_inspector.py`
- `tests/data_validation/test_run_data_validation.py`
- `tests/data_validation/test_run_deep_inspection.py`

## Future Enhancements

### Planned Features
1. **Machine Learning Validation**: ML-based anomaly detection
2. **Real-time Validation**: Continuous validation during processing
3. **Interactive Validation**: Web-based validation interface
4. **Advanced Reporting**: Interactive dashboards and visualizations
5. **Automated Fixes**: Automatic data quality improvements

### Potential Improvements
1. **Distributed Validation**: Support for distributed validation
2. **Custom Validation Rules**: User-defined validation rules
3. **Validation APIs**: REST API for validation services
4. **Quality Prediction**: Predictive quality assessment
5. **Automated Remediation**: Automatic data quality fixes
