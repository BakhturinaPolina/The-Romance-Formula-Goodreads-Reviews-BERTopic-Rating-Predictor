# Pipeline Outputs Archive Summary
**Archive Date:** 2025-09-02  
**Archive Location:** `archive/pipeline_outputs_20250902/`

## Overview
This archive contains outdated and intermediate pipeline outputs from the romance novel NLP research project. These files have been superseded by more recent pipeline steps and are no longer needed for ongoing analysis.

## Archived Contents

### 1. Step 1-3 Processed Data (`step1_3_processed/`)
**Rationale:** Superseded by step 5-6 optimized datasets
- `cleaned_romance_novels_step1_3_20250902_223102.csv` (175MB)
- `cleaned_romance_novels_step1_3_20250902_223102.pkl` (165MB)
- `cleaning_report_step1_3_20250902_223112.json` (51KB)
- `memory_optimization_report_20250902_223102.json` (341B)

### 2. Step 1-3 Outputs (`step1_3_outputs/`)
**Rationale:** Superseded by final pipeline summaries
- `cleaning_pipeline_step1_3_updated_summary.md` (8.9KB)

### 3. Step 4 Outputs (`step4_outputs/`)
**Rationale:** Outlier detection outputs superseded by step 5-6
- `STEP4_TREATMENT_APPLICATION_SUMMARY.md`
- `STEP4_OUTLIER_DETECTION_SUMMARY.md`
- `outlier_treatment_report_step4_20250902_231021.json`
- `outlier_detection_report_step4_20250902_225307.json`
- `cleaned_romance_novels_step4_treated_20250902_231021.pkl` (165MB)
- `execution_summary_step4_20250902_225308.txt`

### 4. Intermediate Data (`intermediate_data/`)
**Rationale:** Intermediate processing outputs no longer needed
- `title_cleaning_report_20250902_174006.json`
- `cleaned_titles_20250902_174006.csv` (172MB)
- `dataset_validation_report_20250902_213340.json`

## Current Active Files (Not Archived)

### Processed Data (`data/processed/`)
- `README.md` - Documentation of current pipeline state

### Outputs (`outputs/`)
- `final_quality_validation/` - Final step 6 outputs (current)
- `data_type_optimization/` - Step 5 outputs (current)
- `eda/` - Exploratory data analysis outputs

## Archive Rationale

1. **Pipeline Progression:** Steps 1-4 are intermediate stages, steps 5-6 contain the final, optimized datasets
2. **Data Supersession:** Later steps incorporate and improve upon earlier cleaning results
3. **Storage Optimization:** Removed ~500MB of intermediate data while preserving final outputs
4. **Maintenance:** Keeps only current, relevant files for ongoing analysis

## Recovery Instructions

If any archived files are needed:
1. Navigate to `archive/pipeline_outputs_20250902/`
2. Locate the specific step or data type subdirectory
3. Copy or move files back to their original locations
4. Update any references in notebooks or scripts

## Total Space Saved
- **CSV Files:** ~347MB
- **Pickle Files:** ~330MB
- **JSON Reports:** ~53KB
- **Total:** ~677MB

## Next Steps
1. Verify that current analysis notebooks reference only active files
2. Update any hardcoded file paths in scripts
3. Consider archiving older log files in `logs/` directory
4. Regular cleanup of intermediate outputs after pipeline completion
