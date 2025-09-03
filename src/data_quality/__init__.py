"""
Data Quality Module for Romance Novel NLP Research
Handles comprehensive data quality assessment and validation through a 6-step pipeline.

Pipeline Steps:
- Step 4: Outlier Detection & Treatment
- Step 5: Data Type Optimization & Persistence  
- Step 6: Final Quality Validation & Certification

Author: Research Assistant
Date: 2025-09-02
"""

from .outlier_detection_step4 import OutlierDetectionReporter
from .apply_outlier_treatment_step4 import OutlierTreatmentApplier
from .data_type_optimization_step5 import DataTypeOptimizer
from .final_quality_validation_step6 import FinalQualityValidator

__all__ = [
    'OutlierDetectionReporter',      # Step 4: Outlier Detection
    'OutlierTreatmentApplier',       # Step 4: Outlier Treatment
    'DataTypeOptimizer',             # Step 5: Data Type Optimization
    'FinalQualityValidator'          # Step 6: Final Quality Validation
]
