# Stage 02: Data Quality

## Purpose

This stage implements a comprehensive 6-step data quality assurance pipeline and statistical audit of the dataset. It ensures data integrity, consistency, and quality before downstream analysis.

## Input Files

- `data/processed/*.csv` - Datasets from Stage 01 (e.g., `romance_books_main_final.csv`)

## Output Files

- `data/processed/*_step[1-6].*` - Datasets after each quality step
- `outputs/reports/*_quality_report_*.json` - Quality reports (JSON format)
- `outputs/reports/*_quality_report_*.md` - Quality reports (Markdown format)
- `outputs/logs/*_quality_*.log` - Execution logs

## File Descriptions

### Data Quality Pipeline (`data_quality/`)

#### `data_quality/pipeline_runner.py`
**Purpose**: Main orchestrator for the complete 6-step data quality pipeline.

**Key Features**:
- Executes all 6 quality steps in sequence
- Manages data flow between steps
- Comprehensive logging and error handling
- Quality gates and validation checks
- Generates summary reports

**Main Function**: `main()` - Runs the complete pipeline

#### `data_quality/step1_missing_values_cleaning.py`
**Purpose**: Identifies and treats missing values in the dataset.

**Key Features**:
- Detects missing values across all columns
- Applies appropriate treatment strategies (removal, imputation, flagging)
- Generates missing value reports
- Preserves data integrity

**Main Class**: `MissingValuesCleaner`

#### `data_quality/step2_duplicate_detection.py`
**Purpose**: Detects and resolves duplicate records in the dataset.

**Key Features**:
- Identifies exact and fuzzy duplicates
- Resolves duplicate conflicts
- Preserves most complete records
- Generates duplicate resolution reports

**Main Class**: `DuplicateDetector`

#### `data_quality/step3_data_type_validation.py`
**Purpose**: Validates and converts data types to ensure consistency.

**Key Features**:
- Validates data types for all columns
- Converts incompatible types
- Handles type conversion errors gracefully
- Generates validation reports

**Main Class**: `DataTypeValidator`

#### `data_quality/step4_outlier_detection.py`
**Purpose**: Detects outliers using statistical methods.

**Key Features**:
- Multiple outlier detection methods (IQR, Z-score, etc.)
- Identifies outliers across numeric columns
- Generates detailed outlier reports
- Flags potential data quality issues

**Main Class**: `OutlierDetectionReporter`

#### `data_quality/step4_outlier_treatment.py`
**Purpose**: Applies treatment strategies for detected outliers.

**Key Features**:
- Multiple treatment options (capping, removal, transformation)
- Preserves data distribution characteristics
- Generates treatment reports
- Configurable treatment strategies

**Main Class**: `OutlierTreatmentApplier`

#### `data_quality/step5_data_type_optimization.py`
**Purpose**: Optimizes data types for memory efficiency and performance.

**Key Features**:
- Converts to optimal numeric types (int8, int16, float32, etc.)
- Optimizes string columns (category types)
- Reduces memory footprint
- Maintains data accuracy

**Main Class**: `DataTypeOptimizer`

#### `data_quality/step6_final_quality_validation.py`
**Purpose**: Final quality validation and certification of the cleaned dataset.

**Key Features**:
- Comprehensive quality checks
- Validates data integrity
- Generates quality certification reports
- Ensures pipeline success criteria

**Main Class**: `FinalQualityValidator`

#### `data_quality/comprehensive_data_cleaner.py`
**Purpose**: Comprehensive data cleaning tool that implements multiple quality fixes.

**Key Features**:
- Removes books outside publication year range
- Removes books with missing descriptions
- Removes books with missing page counts
- Fixes negative work count errors
- Applies ratings/reviews cuts
- Handles low-rating authors

**Main Class**: `ComprehensiveDataCleaner`

### Data Audit (`data_audit/`)

#### `data_audit/core/data_auditor.py`
**Purpose**: Statistical analysis and audit functionality for data quality assessment.

**Key Features**:
- Schema validation
- Heavy-tail distribution analysis
- Overdispersion testing
- Statistical summary generation
- Power-law analysis

**Main Class**: `DataAuditor`

#### `data_audit/parsing/list_parser.py`
**Purpose**: Parses and processes list-type data fields (e.g., genres, shelves).

**Key Features**:
- Extracts structured data from list strings
- Normalizes list formats
- Validates list contents
- Generates parsing reports

#### `data_audit/comprehensive_data_analysis.py`
**Purpose**: Comprehensive data analysis tool that identifies data quality issues.

**Key Features**:
- Analyzes publication year ranges
- Identifies missing data patterns
- Analyzes author statistics
- Suggests data cuts and filters
- Generates detailed analysis reports

**Main Class**: `ComprehensiveDataAnalyzer`


### Utilities (`utils/`)

#### `utils/diff_bridge_runs.py`
**Purpose**: Utility for comparing different pipeline runs and identifying differences.

## How to Run

### Complete Pipeline

**Step 1**: Run the complete 6-step quality pipeline
```bash
cd src/02_data_quality
python data_quality/pipeline_runner.py
```

This will:
1. Execute all 6 quality steps in sequence
2. Generate intermediate datasets after each step
3. Create quality reports in `outputs/reports/`
4. Generate execution logs in `outputs/logs/`

### Individual Steps

Run individual quality steps if needed:

```bash
cd src/02_data_quality

# Step 1: Missing values cleaning
python data_quality/step1_missing_values_cleaning.py

# Step 2: Duplicate detection
python data_quality/step2_duplicate_detection.py

# Step 3: Data type validation
python data_quality/step3_data_type_validation.py

# Step 4: Outlier detection
python data_quality/step4_outlier_detection.py

# Step 4: Outlier treatment
python data_quality/step4_outlier_treatment.py

# Step 5: Data type optimization
python data_quality/step5_data_type_optimization.py

# Step 6: Final quality validation
python data_quality/step6_final_quality_validation.py
```

### Comprehensive Cleaning

Run the comprehensive data cleaner:

```bash
cd src/02_data_quality
python data_quality/comprehensive_data_cleaner.py
```

### Data Audit

**Option 1**: Using Makefile (recommended)
```bash
cd src/02_data_quality
make audit
```

**Option 2**: Run directly
```bash
cd src/02_data_quality
python data_audit/core/data_auditor.py --data-path ../../data/processed/main_dataset.csv
```

**Option 3**: Run list parsing
```bash
make parse
# Or directly:
python data_audit/parsing/list_parser.py --data-path ../../data/processed/main_dataset.csv
```

### Comprehensive Analysis

Run comprehensive data analysis:

```bash
cd src/02_data_quality
python data_audit/comprehensive_data_analysis.py
```

## Dependencies

### Core Dependencies
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical operations
- `scipy` - Statistical tests and distributions

### Optional Dependencies
- `Make` - For running audit commands via Makefile

## Pipeline Workflow

### Typical Workflow

1. **Step 1: Missing Values** → Identifies and treats missing data
2. **Step 2: Duplicate Detection** → Removes duplicate records
3. **Step 3: Data Type Validation** → Ensures correct data types
4. **Step 4: Outlier Detection** → Identifies statistical outliers
5. **Step 4: Outlier Treatment** → Applies outlier treatment strategies
6. **Step 5: Data Type Optimization** → Optimizes memory usage
7. **Step 6: Final Validation** → Certifies data quality

### Data Flow

```
data/processed/input.csv
  → Step 1 → *_step1.csv
  → Step 2 → *_step2.csv
  → Step 3 → *_step3.csv
  → Step 4 (detection) → *_step4_detection.csv
  → Step 4 (treatment) → *_step4_treatment.csv
  → Step 5 → *_step5.csv
  → Step 6 → *_step6_final.csv
```

## Example Usage

### Complete Pipeline Run

```bash
# 1. Navigate to stage directory
cd src/02_data_quality

# 2. Run complete pipeline
python data_quality/pipeline_runner.py

# 3. Check outputs
ls -lh data/processed/*_step*.csv
ls -lh outputs/reports/*_quality*.json
```

### Individual Step Execution

```bash
# Run specific step
python data_quality/step1_missing_values_cleaning.py \
    --input data/processed/input.csv \
    --output data/processed/output_step1.csv
```

### Data Audit

```bash
# Run complete audit
make audit

# Run with custom data path
python data_audit/core/data_auditor.py \
    --data-path ../../data/processed/romance_books_main_final.csv \
    --output-dir ./audit_outputs
```


## Key Features

### Quality Assurance Pipeline
- **6-step automated pipeline** with quality gates
- **Comprehensive logging** of all transformations
- **Detailed reporting** at each step
- **Error handling** and rollback capabilities

### Statistical Audit
- **Heavy-tail analysis** for power-law distributions
- **Overdispersion testing** for count data
- **Schema validation** for data integrity

### Data Cleaning
- **Missing value treatment** with multiple strategies
- **Duplicate resolution** with conflict handling
- **Outlier detection and treatment** with statistical methods
- **Data type optimization** for performance

### Reporting
- **JSON reports** for programmatic access
- **Markdown reports** for human readability
- **Execution logs** for debugging and reproducibility
- **Quality metrics** and statistics
