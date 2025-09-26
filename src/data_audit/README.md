# Data Audit Module

**Statistical Analysis & Data Exploration for Romance Novel Dataset**

This module provides comprehensive data auditing and statistical analysis capabilities for the romance novel dataset. It implements rigorous statistical methodologies for schema validation, heavy-tail analysis, overdispersion testing, and data parsing.

## Overview

The Data Audit module is designed to provide deep insights into data quality and statistical properties of the romance novel dataset. It follows established statistical methodologies and provides both automated analysis and interactive exploration capabilities.

### Key Features

1. **Schema Validation**: Comprehensive column presence and data type validation
2. **Heavy-Tail Analysis**: Clauset-Shalizi-Newman (2009) power-law fitting methodology
3. **Overdispersion Testing**: Dean-Lawless and Cameron-Trivedi formal statistical tests
4. **List Parsing**: Robust parsing of list-like fields with fallback strategies
5. **Interactive Analysis**: Jupyter notebooks for exploratory data analysis
6. **Statistical Reporting**: HTML reports with comprehensive analysis results

## Module Structure

```
data_audit/
├── __init__.py                    # Module initialization
├── README.md                      # This documentation
├── requirements.txt               # Python dependencies
├── Makefile                       # Pipeline automation
├── core/                          # Core audit functionality
│   ├── __init__.py
│   └── data_auditor.py           # Main audit script (00_load_audit.py)
├── parsing/                       # Data parsing utilities
│   ├── __init__.py
│   └── list_parser.py            # List parsing script (01_parse_lists.py)
├── notebooks/                     # Interactive analysis
│   ├── 01_data_audit_and_heavytails.ipynb
│   ├── eda_unusual_page_counts_notebook.ipynb
│   ├── goodreads_shelf_normalization.ipynb
│   └── metadata_cleaning_books_duplicates.ipynb
└── utils/                         # Utility scripts
    ├── __init__.py
    └── diff_bridge_runs.py       # Bridge run comparison (02c_diff_bridge_runs.py)
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use Makefile
make install
```

### 2. Basic Usage

```bash
# Run complete audit pipeline
make audit

# Or run individual steps
make audit-only
make parse-only
```

### 3. Interactive Analysis

```bash
# Start Jupyter notebook server
jupyter notebook notebooks/

# Open 01_data_audit_and_heavytails.ipynb for comprehensive analysis
```

## Core Components

### Data Auditor (`core/data_auditor.py`)

**Purpose**: Comprehensive statistical analysis and data quality assessment

**Key Capabilities**:
- Schema validation with expected column checking
- Missing data pattern analysis
- Heavy-tail distribution analysis using CSN methodology
- Formal overdispersion tests (Dean-Lawless, Cameron-Trivedi)
- Edge case detection and zero-inflation analysis
- HTML report generation with visualizations

**Usage**:
```bash
python core/data_auditor.py \
    --data-path ../../data/processed/romance_books_main_final.csv \
    --output-dir ./audit_outputs
```

**Outputs**:
- `audit_report.html` - Comprehensive HTML report
- `audit_results.json` - Machine-readable results
- `overdispersion_tests.json` - Statistical test results
- `*.png` - Visualization plots

### List Parser (`parsing/list_parser.py`)

**Purpose**: Robust parsing of list-like fields in the dataset

**Key Capabilities**:
- Multi-strategy parsing (ast.literal_eval, regex, fallback)
- RFC-4180 compliant CSV parsing
- Data validation and quality checks
- Parquet output for efficient storage
- Comprehensive error handling

**Usage**:
```bash
python parsing/list_parser.py \
    --data-path ../../data/processed/romance_books_main_final.csv \
    --output-dir ./parse_outputs
```

**Outputs**:
- `parsed_books_YYYYmmdd_HHMMSS.parquet` - Parsed data
- `parsing_summary.json` - Processing summary
- `parsing_report.txt` - Human-readable report

### Utility Scripts (`utils/`)

**diff_bridge_runs.py**: Compare diagnostics between different bridge runs
```bash
python utils/diff_bridge_runs.py \
    --old bridge_qa_run1/diagnostics_summary.json \
    --new bridge_qa_run2/diagnostics_summary.json
```

## Statistical Methodologies

### Heavy-Tail Analysis (Clauset-Shalizi-Newman, 2009)

**Purpose**: Detect and characterize power-law distributions in count variables

**Implementation**:
- Maximum likelihood estimation for power-law parameters
- Kolmogorov-Smirnov goodness-of-fit testing
- Model comparison with alternative distributions
- Tail fraction analysis

**Variables Analyzed**:
- `ratings_count_sum`
- `text_reviews_count_sum`
- `author_ratings_count`

### Overdispersion Testing

**Purpose**: Detect violations of Poisson assumptions in count data

**Tests Implemented**:

1. **Dean-Lawless (1989)**: Pearson chi-square z-test
   - Tests: H₀: Var(Y) = E(Y) vs H₁: Var(Y) > E(Y)
   - Statistic: z = (χ² - df) / √(2df)

2. **Cameron-Trivedi (1990)**: Auxiliary OLS test
   - Tests: Var(Y|X) = μ + αμ²
   - Regresses: Y* = ((y-μ)²-y)/μ on μ

### Schema Validation

**Expected Schema** (19 columns):
```python
expected_cols = [
    'work_id', 'book_id_list_en', 'title', 'publication_year', 'num_pages_median',
    'description', 'language_codes_en', 'author_id', 'author_name', 
    'author_average_rating', 'author_ratings_count', 'series_id', 'series_title',
    'ratings_count_sum', 'text_reviews_count_sum', 'average_rating_weighted_mean',
    'genres_str', 'shelves_str', 'series_works_count_numeric'
]
```

## Dependencies

### Core Dependencies
- `pandas>=2.0` - Data manipulation
- `numpy>=1.23` - Numerical operations
- `matplotlib>=3.7` - Plotting
- `seaborn>=0.12` - Statistical visualizations
- `statsmodels>=0.14` - Statistical modeling
- `jinja2>=3.1` - HTML template rendering

### Optional Dependencies
- `powerlaw>=1.4` - Heavy-tail analysis (CSN methodology)
- `scipy>=1.10` - Advanced statistical functions

## Pipeline Integration

### With Data Quality Module
- **Input**: Raw CSV data
- **Output**: Validated and parsed data for quality assessment
- **Integration**: Provides statistical foundation for quality decisions

### With Shelf Normalization Module
- **Input**: Parsed books with list columns
- **Output**: Audit results for normalization validation
- **Integration**: Ensures data integrity before normalization

### With NLP Preprocessing Module
- **Input**: Clean, validated data
- **Output**: Statistical insights for preprocessing decisions
- **Integration**: Informs text processing strategies

## Advanced Usage

### Custom Analysis

```python
from data_audit.core.data_auditor import DataAuditor

# Initialize auditor
auditor = DataAuditor("path/to/data.csv", "output/dir")

# Run specific analyses
auditor.schema_audit()
auditor.heavy_tail_analysis()
auditor.formal_overdispersion_analysis()

# Generate custom reports
auditor.generate_html_report()
```

### Batch Processing

```bash
# Process multiple datasets
for dataset in datasets/*.csv; do
    python core/data_auditor.py \
        --data-path "$dataset" \
        --output-dir "audit_outputs/$(basename "$dataset" .csv)"
done
```

### Integration with CI/CD

```yaml
# Example GitHub Actions workflow
- name: Run Data Audit
  run: |
    cd src/data_audit
    make audit
    python utils/validate_audit_results.py
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**:
   - Use smaller datasets for testing
   - Increase system memory
   - Process data in chunks

3. **Power-law Package Issues**:
   ```bash
   pip install powerlaw
   # Or skip heavy-tail analysis if not needed
   ```

4. **Parquet Engine Issues**:
   - Scripts automatically fall back between pyarrow/fastparquet
   - Install both engines for best compatibility

### Debug Mode

```bash
# Enable verbose logging
python core/data_auditor.py --verbose [other options]
python parsing/list_parser.py --verbose [other options]
```

### Log Analysis

```bash
# Check processing stages
grep "INFO" audit_outputs/audit_results.json

# Analyze error patterns
grep "ERROR" audit_outputs/*.log
```

## Performance

### Scalability
- **Memory**: ~1-2GB for 50K books
- **Time**: ~3-5 minutes for full audit
- **Bottlenecks**: Heavy-tail analysis, HTML generation

### Optimization
- Use `--n-top-shelves` for large datasets
- Process data in chunks for memory efficiency
- Cache intermediate results for repeated analysis

## Quality Assurance

### Testing
```bash
# Run test suite
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_data_auditor.py -v
python -m pytest tests/test_list_parser.py -v
```

### Validation
- Schema consistency checks
- Statistical test validation
- Output format verification
- Cross-validation with known datasets

## References

### Statistical Methods
- Clauset, A., Shalizi, C. R., & Newman, M. E. (2009). Power-law distributions in empirical data. SIAM review, 51(4), 661-703.
- Dean, C., & Lawless, J. F. (1989). Tests for detecting overdispersion in Poisson regression models. Journal of the American Statistical Association, 84(406), 467-472.
- Cameron, A. C., & Trivedi, P. K. (1990). Regression-based tests for overdispersion in the Poisson model. Journal of econometrics, 46(3), 347-364.

### Software Packages
- Alstott, J., Bullmore, E., & Plenz, D. (2014). powerlaw: a Python package for analysis of heavy-tailed distributions. PloS one, 9(1), e85777.

## Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd romance-novel-nlp-research/src/data_audit

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
make test
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function signatures
- Document all public functions and classes
- Include comprehensive docstrings

### Testing
- Write unit tests for all new functionality
- Include integration tests for pipeline components
- Validate statistical methods against known datasets

## License

This module is part of the Romance Novel NLP Research project. See the main project repository for licensing information.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the Jupyter notebooks for examples
3. Open an issue in the project repository
4. Contact the development team

---

**Last Updated**: 2025-01-09  
**Version**: 1.0.0  
**Maintainer**: Romance Novel NLP Research Team
