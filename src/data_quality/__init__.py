"""
Data Quality Module for Romance Novel NLP Research
Handles comprehensive data quality assessment and validation through a unified 6-step pipeline.

Complete Pipeline Steps:
- Step 1: Missing Values Treatment
- Step 2: Duplicate Detection & Resolution
- Step 3: Data Type Validation & Conversion
- Step 4: Outlier Detection & Treatment
- Step 5: Data Type Optimization & Persistence  
- Step 6: Final Quality Validation & Certification

Author: Research Assistant
Date: 2025-09-02
"""

# Step 1: Missing Values Treatment
from .step1_missing_values_cleaning import MissingValuesCleaner

# Step 2: Duplicate Detection & Resolution
from .step2_duplicate_detection import DuplicateDetector

# Step 3: Data Type Validation & Conversion
from .step3_data_type_validation import DataTypeValidator

# Step 4: Outlier Detection & Treatment
from .step4_outlier_detection import OutlierDetectionReporter
from .step4_outlier_treatment import OutlierTreatmentApplier

# Step 5: Data Type Optimization & Persistence
from .step5_data_type_optimization import DataTypeOptimizer

# Step 6: Final Quality Validation & Certification
from .step6_final_quality_validation import FinalQualityValidator

__all__ = [
    # Step 1: Missing Values Treatment
    'MissingValuesCleaner',          # Step 1: Missing Values Treatment
    
    # Step 2: Duplicate Detection & Resolution
    'DuplicateDetector',             # Step 2: Duplicate Detection & Resolution
    
    # Step 3: Data Type Validation & Conversion
    'DataTypeValidator',             # Step 3: Data Type Validation & Conversion
    
    # Step 4: Outlier Detection & Treatment
    'OutlierDetectionReporter',      # Step 4: Outlier Detection
    'OutlierTreatmentApplier',       # Step 4: Outlier Treatment
    
    # Step 5: Data Type Optimization & Persistence
    'DataTypeOptimizer',             # Step 5: Data Type Optimization
    
    # Step 6: Final Quality Validation & Certification
    'FinalQualityValidator'          # Step 6: Final Quality Validation
]
