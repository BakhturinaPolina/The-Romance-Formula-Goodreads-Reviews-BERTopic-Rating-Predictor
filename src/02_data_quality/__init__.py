"""
Data Quality Module

This module provides comprehensive data quality assurance pipeline and statistical audit
capabilities for the Goodreads dataset.

Main Components:
- data_quality: 6-step quality assurance pipeline
- data_audit: Statistical analysis and data exploration
- utils: Utility scripts and helper functions
"""

from .data_quality import (
    run_pipeline,
    MissingValuesCleaner,
    DuplicateDetector,
    DataTypeValidator,
    OutlierDetectionReporter,
    OutlierTreatmentApplier,
    DataTypeOptimizer,
    FinalQualityValidator,
    ComprehensiveDataCleaner
)

from .data_audit import (
    ComprehensiveDataAnalyzer,
    DataAuditor
)

__all__ = [
    'run_pipeline',
    'MissingValuesCleaner',
    'DuplicateDetector',
    'DataTypeValidator',
    'OutlierDetectionReporter',
    'OutlierTreatmentApplier',
    'DataTypeOptimizer',
    'FinalQualityValidator',
    'ComprehensiveDataCleaner',
    'ComprehensiveDataAnalyzer',
    'DataAuditor'
]
