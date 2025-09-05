# Archive Summary

## Overview
This archive contains Python files that were part of earlier development phases but are no longer needed for the current Romance Novel NLP Research project.

## What Was Archived

### Data Quality Module Artifacts (data_quality_artifacts/)
**Reason**: Development artifacts, planning scripts, and duplicate functionality from the data quality module development phases.

**Archived Files**:
- `quick_step2_planning.py` - Quick analysis for Step 2 planning
- `quick_exclusion_analysis.py` - Quick exclusion analysis for dataset planning
- `dataset_structure_analysis.py` - Dataset structure analysis for Step 2 planning
- `create_cleaned_dataset.py` - Script to create cleaned dataset with exclusions
- `duplicate_inconsistency_detection.py` - Step 2 duplicate detection (replaced by outlier detection)
- `title_duplication_cleaning.py` - Title duplication cleaning (completed functionality)
- `run_title_cleaning.py` - Runner for title cleaning (completed functionality)
- `run_outlier_detection.py` - Basic runner (replaced by enhanced version)
- `dataset_validation_script.py` - Dataset validation analysis
- `data_integration_script.py` - Data integration analysis
- `eda_title_cleaning_analysis.py` - Title cleaning EDA analysis
- Various documentation files (README.md, EXCLUSION_RATIONALE.md, etc.)

**Current Active Files**: 
- `src/data_quality/data_quality_assessment.py` - Step 1: Data Quality Assessment
- `src/data_quality/outlier_detection_analysis.py` - Step 3: Outlier Detection
- `src/data_quality/run_quality_assessment.py` - Quality assessment runner
- `src/data_quality/run_outlier_detection_enhanced.py` - Enhanced outlier detection runner

### CSV Building Files (csv_building_old/)
**Reason**: Multiple versions of CSV builders were created during development. Only `final_csv_builder_working.py` is currently needed.

**Archived Files**:
- `final_csv_builder_cleaned_titles.py` - Early version with title cleaning
- `final_csv_builder_cleaned_titles_fast.py` - Fast version of title cleaning
- `final_csv_builder_improved.py` - Improved version
- `final_csv_builder_optimized.py` - Optimized version
- `run_cleaned_titles_builder.py` - Runner for cleaned titles
- `run_fast_cleaned_titles_builder.py` - Runner for fast version
- `run_improved_builder.py` - Runner for improved version
- `run_optimized_builder.py` - Runner for optimized version
- `simple_title_cleaner.py` - Simple title cleaning utility
- `title_cleaner.py` - Title cleaning utility
- `deep_title_analysis.py` - Title analysis utility
- `analyze_title_columns.py` - Title column analysis

**Current Active File**: `src/csv_building/final_csv_builder_working.py`

### Data Cleaning Files (data_cleaning_old/)
**Reason**: Old version of the data cleaner that was replaced by the current implementation.

**Archived Files**:
- `romance_novel_cleaner_old.py` - Old version of data cleaner

**Current Active File**: `src/data_cleaning/romance_novel_cleaner.py`

### Test Files (test_files/)
**Reason**: Various test files that are not part of the main pipeline and were used during development.

**Archived Files**:
- `test_improved_cleaner.py` - Test for improved cleaner
- `test_improved_structure.py` - Test for improved structure
- `test_improved_pipeline.py` - Test for improved pipeline
- `test_cleaning_pipeline.py` - Test for cleaning pipeline

### Previously Archived Files
The following files were already in the archive from previous cleanup:
- `final_csv_builder.py` - Early CSV builder
- `pipeline_runner.py` - Old pipeline runner
- `data_integrator.py` - Data integration utilities
- `quality_filters.py` - Quality filtering utilities
- `pipeline_validator.py` - Pipeline validation
- `config_loader.py` - Configuration loading
- `data_loader.py` - Data loading utilities
- Various other utilities and exploration tools

## Current Project Structure

### Active Components
- `src/csv_building/` - Core CSV generation functionality
- `src/data_quality/` - Essential data quality assessment and outlier detection
- `notebooks/02_final_dataset_eda_and_cleaning.ipynb` - Main analysis notebook

### Project Status
- **Phase 1**: Data Exploration and Understanding ✅ COMPLETE
- **Phase 2**: Data Processing and Preparation ✅ COMPLETE  
- **Phase 3**: NLP Analysis and Topic Modeling 🔄 READY

## Archive Rationale

1. **Development History**: These files represent the evolution of the project through multiple iterations
2. **Code Quality**: The current active files are the refined, working versions
3. **Maintenance**: Removing unused files makes the codebase easier to navigate and maintain
4. **Documentation**: The archive preserves the development history for reference
5. **Focus**: Cleaned modules focus on essential functionality needed for current research phase

## Recovery Instructions

If any of these files are needed in the future:
1. Navigate to `archive/unused_code/` or `archive/data_quality_artifacts/`
2. Find the appropriate subdirectory
3. Copy the needed file back to the appropriate `src/` directory
4. Update imports and dependencies as needed

## Notes

- All archived files are preserved and can be recovered if needed
- The archive maintains the original directory structure for easy navigation
- `__pycache__` directories were cleaned up to remove compiled Python files
- The current project structure is clean and focused on the active components needed for Phase 3
- Data quality module is now streamlined with only essential functionality

### Repository Cleanup 2025-01-05 (cleanup_20250105/)
**Reason**: Remove outdated and duplicate files to maintain clean repository structure.

**Archived Files**:
- **Old Logs**: 20+ log files from September 2, 2025 (outdated)
- **Duplicate Outputs**: 8 duplicate output files with older timestamps
- **CSV Building Archive**: 4 files from src/csv_building/archive/
- **Old Reports**: 7 superseded analysis reports and summaries

**Current Active Files**: 
- Recent logs from September 4, 2025
- Latest output files with timestamp 20250904_223745
- Current analysis reports and summaries

### Repository Cleanup 2025-01-06 (cleanup_20250106/)
**Reason**: Remove Python cache files, empty log files, and duplicate outputs to maintain clean repository structure.

**Archived Files**:
- **Python Cache Files**: `__pycache__` directories from src/data_quality/ and src/csv_building/
- **Empty Log Files**: 5 empty log files (0 bytes) that were never used
- **Duplicate Outputs**: Earlier version of memory efficient quality report and unprocessed batch files
- **Development Tools**: 3 diagnostic scripts used during development (analyze_variable_quality.py, diagnose_null_sources.py, inspect_json_files.py)
- **Alternative Runners**: 4 redundant runner scripts (run_batch_processing.py, run_memory_efficient.py, run_on_csv_output.py, run_outlier_detection_step4.py)

**Current Active Files**: 
- Clean source code without cache files
- Streamlined src/ directory with only core functionality
- Latest output files with timestamp 20250904_232345
- Processed batch files (kept processed versions)

---

**Last Updated**: January 2025  
**Archive Status**: Complete  
**Next Review**: After NLP analysis phase completion
