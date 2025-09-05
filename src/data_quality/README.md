# Data Quality Module - Romance Novel NLP Research

## Overview

The Data Quality Module implements a **unified 6-step data cleaning and validation pipeline** for romance novel datasets. This module provides a comprehensive, coherent approach to data quality assurance, transforming raw datasets into analysis-ready, quality-certified data.

## 🎯 **Unified Pipeline Architecture**

The module implements a complete **6-step data quality pipeline**:

```
Step 1: Missing Values Treatment (Critical Priority) ✓
Step 2: Duplicate Detection & Resolution (High Priority) ✓
Step 3: Data Type Validation & Conversion (High Priority) ✓
Step 4: Outlier Detection & Treatment (Medium Priority) ✓
Step 5: Data Type Optimization & Persistence (High Priority) ✓
Step 6: Final Quality Validation & Certification (Medium Priority) ✓
```

## 🏗️ **Complete Pipeline Components**

### Step 1: Missing Values Treatment

#### **MissingValuesCleaner** (`step1_missing_values_cleaning.py`)
- **Purpose**: Comprehensive missing values analysis and treatment
- **Strategies**: Strategic flagging and exclusion (no imputation)
- **Key Features**: Rating exclusion strategy, missing value flags, completeness validation
- **Outputs**: Treated datasets with comprehensive missing value reports

### Step 2: Duplicate Detection & Resolution

#### **DuplicateDetector** (`step2_duplicate_detection.py`)
- **Purpose**: Detect and resolve various types of duplicates
- **Detection Types**: Exact duplicates, title duplicates, author duplicates
- **Resolution**: Removal, flagging, and disambiguation strategies
- **Outputs**: Cleaned datasets with duplicate resolution reports

### Step 3: Data Type Validation & Conversion

#### **DataTypeValidator** (`step3_data_type_validation.py`)
- **Purpose**: Validate and optimize data types, create derived variables
- **Features**: Publication year validation, categorical creation, type optimization
- **Derived Variables**: Decade, book length, rating, and popularity categories
- **Outputs**: Validated datasets with optimized data types

### Step 4: Outlier Detection & Treatment

#### **OutlierDetectionReporter** (`step4_outlier_detection.py`)
- **Purpose**: Comprehensive outlier detection and reporting
- **Methods**: Z-score, IQR, percentile-based outlier detection
- **Focus**: Detection and documentation only - no data modification
- **Outputs**: Statistical analysis reports with actionable recommendations

#### **OutlierTreatmentApplier** (`step4_outlier_treatment.py`)
- **Purpose**: Apply outlier treatment strategies
- **Approach**: Conservative treatment with publication year filtering
- **Strategy**: Remove publication year anomalies (2000-2017 range)
- **Outputs**: Treated datasets with treatment reports

### Step 5: Data Type Optimization & Persistence

#### **DataTypeOptimizer** (`step5_data_type_optimization.py`)
- **Purpose**: Optimize data types for memory efficiency
- **Optimizations**: int16/32, float32, category conversions
- **Persistence**: Parquet and pickle formats for type preservation
- **Outputs**: Optimized datasets with memory usage reports

### Step 6: Final Quality Validation & Certification

#### **FinalQualityValidator** (`step6_final_quality_validation.py`)
- **Purpose**: Cross-validate all pipeline steps
- **Quality Gates**: Completeness, consistency, integrity, optimization
- **Certification**: Final data quality certification
- **Outputs**: Validation reports and quality scores

## 🚀 **Usage**

### Unified Pipeline Execution

#### **Complete Pipeline Runner** (`pipeline_runner.py`)
Execute the entire 6-step pipeline in sequence:

```python
from src.data_quality import DataQualityPipelineRunner

# Initialize pipeline runner
runner = DataQualityPipelineRunner("data/processed/romance_novels_integrated.csv")

# Run complete pipeline
results = runner.run_complete_pipeline()

# Print summary
runner.print_pipeline_summary()
```

#### **Command Line Execution**
```bash
# Navigate to data quality module
cd src/data_quality

# Run complete pipeline
python pipeline_runner.py
```

### Individual Step Execution

#### **Step-by-Step Execution**
```python
from src.data_quality import (
    MissingValuesCleaner,
    DuplicateDetector,
    DataTypeValidator,
    OutlierDetectionReporter,
    OutlierTreatmentApplier,
    DataTypeOptimizer,
    FinalQualityValidator
)

# Step 1: Missing Values Treatment
cleaner = MissingValuesCleaner("data/processed/romance_novels_integrated.csv")
step1_results = cleaner.run_complete_treatment()

# Step 2: Duplicate Detection & Resolution
detector = DuplicateDetector(cleaner.output_path)
step2_results = detector.run_complete_resolution()

# Step 3: Data Type Validation & Conversion
validator = DataTypeValidator(detector.output_path)
step3_results = validator.run_complete_validation()

# Step 4: Outlier Detection
reporter = OutlierDetectionReporter(validator.output_path)
step4a_results = reporter.run_comprehensive_analysis()

# Step 4: Outlier Treatment
applier = OutlierTreatmentApplier(validator.output_path)
step4b_results = applier.run_complete_treatment()

# Step 5: Data Type Optimization
optimizer = DataTypeOptimizer(applier.output_path)
step5_results = optimizer.run_complete_optimization()

# Step 6: Final Validation
final_validator = FinalQualityValidator(optimizer.output_path)
step6_results = final_validator.run_complete_validation()
```

### Individual Step Scripts
```bash
# Run individual steps
python step1_missing_values_cleaning.py
python step2_duplicate_detection.py
python step3_data_type_validation.py
python step4_outlier_detection.py
python step4_outlier_treatment.py
python step5_data_type_optimization.py
python step6_final_quality_validation.py
```

## 📊 **Pipeline Output Structure**

### Step-by-Step Outputs

#### **Step 1 Outputs**
- **Treated Dataset**: `romance_novels_step1_missing_values_treated_YYYYMMDD_HHMMSS.pkl`
- **Treatment Report**: `missing_values_treatment_report_step1_YYYYMMDD_HHMMSS.json`
- **Location**: `outputs/missing_values_cleaning/`

#### **Step 2 Outputs**
- **Resolved Dataset**: `romance_novels_step2_duplicates_resolved_YYYYMMDD_HHMMSS.pkl`
- **Resolution Report**: `duplicate_resolution_report_step2_YYYYMMDD_HHMMSS.json`
- **Location**: `outputs/duplicate_detection/`

#### **Step 3 Outputs**
- **Validated Dataset**: `romance_novels_step3_data_types_validated_YYYYMMDD_HHMMSS.pkl`
- **Validation Report**: `data_type_validation_report_step3_YYYYMMDD_HHMMSS.json`
- **Location**: `outputs/data_type_validation/`

#### **Step 4 Outputs**
- **Detection Report**: `outlier_detection_report_step4_YYYYMMDD_HHMMSS.json`
- **Treated Dataset**: `cleaned_romance_novels_step4_treated_YYYYMMDD_HHMMSS.pkl`
- **Treatment Report**: `outlier_treatment_report_step4_YYYYMMDD_HHMMSS.json`
- **Location**: `outputs/outlier_detection/`

#### **Step 5 Outputs**
- **Optimized Dataset**: `cleaned_romance_novels_step5_optimized_YYYYMMDD_HHMMSS.pkl`
- **Optimization Report**: `data_type_optimization_report_step5_YYYYMMDD_HHMMSS.json`
- **Location**: `outputs/data_type_optimization/`

#### **Step 6 Outputs**
- **Validation Report**: `final_quality_validation_report_step6_YYYYMMDD_HHMMSS.json`
- **Location**: `outputs/final_quality_validation/`

### **Final Pipeline Outputs**
- **Complete Pipeline Report**: `data_quality_pipeline_report_YYYYMMDD_HHMMSS.json`
- **Location**: `outputs/pipeline_execution/`

## 🔧 **Configuration**

### **Pipeline Configuration**
```python
# Customize pipeline execution
runner = DataQualityPipelineRunner()

# Enable/disable specific steps
runner.pipeline_steps['step1']['enabled'] = True
runner.pipeline_steps['step2']['enabled'] = True
# ... etc

# Custom input/output paths
runner.input_data_path = "custom/input/path.csv"
```

### **Step-Specific Configuration**
```python
# Step 1: Missing Values Treatment
cleaner = MissingValuesCleaner()
cleaner.treatment_strategies['custom_field'] = 'exclude_from_analysis'

# Step 2: Duplicate Detection
detector = DuplicateDetector()
detector.duplicate_criteria['custom_duplicates'] = ['field1', 'field2']

# Step 3: Data Type Validation
validator = DataTypeValidator()
validator.optimization_mappings['custom_field'] = 'int16'

# Step 4: Outlier Detection
reporter = OutlierDetectionReporter()
reporter.numerical_fields.append('custom_field')

# Step 5: Data Type Optimization
optimizer = DataTypeOptimizer()
optimizer.numerical_optimizations['custom_field'] = 'int32'

# Step 6: Final Validation
validator = FinalQualityValidator()
validator.quality_gates['completeness'] = 0.98  # 98% threshold
```

## 📈 **Performance Characteristics**

### **Execution Time**
- **Step 1 (Missing Values)**: 10-30 seconds for 80K records
- **Step 2 (Duplicates)**: 15-45 seconds for 80K records
- **Step 3 (Data Types)**: 20-60 seconds for 80K records
- **Step 4a (Outlier Detection)**: 15-60 seconds for 80K records
- **Step 4b (Outlier Treatment)**: 5-15 seconds for 80K records
- **Step 5 (Optimization)**: 10-30 seconds for 80K records
- **Step 6 (Validation)**: 5-20 seconds for 80K records
- **Complete Pipeline**: 2-5 minutes for 80K records

### **Memory Usage**
- **Efficient Processing**: Minimal memory overhead
- **Type Preservation**: Maintains optimized data types throughout
- **Scalable**: Linear performance scaling with dataset size
- **Memory Optimization**: 20-40% memory reduction through type optimization

## 🧪 **Testing & Validation**

### **Built-in Validation**
- **Data Integrity Checks**: Ensures no data loss during processing
- **Type Preservation**: Validates data type optimizations
- **Quality Gates**: Automated quality threshold validation
- **Cross-Validation**: Pipeline step consistency verification

### **Quality Metrics**
- **Completeness**: Missing value rates and data coverage
- **Consistency**: Data format and value consistency
- **Integrity**: Data type and range validation
- **Optimization**: Memory usage and type efficiency

## 📋 **Requirements**

### **Dependencies**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **pyarrow**: Parquet file handling (optional)
- **pathlib**: File path operations

### **Python Version**
- **Python 3.8+**: Required for type hints and modern features
- **Virtual Environment**: Recommended for dependency isolation

## 📚 **Documentation**

### **Additional Resources**
- **Step-by-Step Guide**: `README_STEP_BY_STEP.md`
- **Pipeline Overview**: Project root documentation
- **Code Examples**: Inline documentation and docstrings
- **Configuration**: YAML configuration files

### **Support**
- **Code Comments**: Comprehensive inline documentation
- **Error Messages**: Clear error descriptions and solutions
- **Logging**: Detailed execution logging for debugging
- **Examples**: Working code examples for common use cases

## 🎯 **Success Metrics**

### **Quality Targets**
- **Overall Quality Score**: ≥90% for production use
- **Data Completeness**: ≥95% non-null values
- **Type Optimization**: ≥80% columns optimized
- **Pipeline Consistency**: 100% step validation success

### **Business Impact**
- **Data Reliability**: Improved analysis confidence
- **Processing Efficiency**: Reduced memory and time requirements
- **Quality Assurance**: Systematic data validation
- **Risk Mitigation**: Early anomaly detection

## 🔄 **Pipeline Flow**

```
Raw Dataset
    ↓
Step 1: Missing Values Treatment
    ↓ (80,705 records)
Step 2: Duplicate Detection & Resolution
    ↓ (80,705 records)
Step 3: Data Type Validation & Conversion
    ↓ (80,705 records)
Step 4a: Outlier Detection
    ↓ (Analysis only)
Step 4b: Outlier Treatment
    ↓ (80,657 records)
Step 5: Data Type Optimization & Persistence
    ↓ (80,657 records, optimized)
Step 6: Final Quality Validation & Certification
    ↓
Quality-Certified Dataset
```

## 🚀 **Quick Start**

### **1. Run Complete Pipeline**
```bash
cd src/data_quality
python pipeline_runner.py
```

### **2. Run Individual Steps**
```bash
python step1_missing_values_cleaning.py
python step2_duplicate_detection.py
python step3_data_type_validation.py
python step4_outlier_detection.py
python step4_outlier_treatment.py
python step5_data_type_optimization.py
python step6_final_quality_validation.py
```

### **3. Programmatic Usage**
```python
from src.data_quality import DataQualityPipelineRunner

# Run complete pipeline
runner = DataQualityPipelineRunner("your_dataset.csv")
results = runner.run_complete_pipeline()
```

---

*The Data Quality Module provides a robust, scalable foundation for ensuring data quality in romance novel NLP research. The unified 6-step pipeline transforms raw data into analysis-ready, quality-certified datasets through systematic validation, optimization, and certification processes.*