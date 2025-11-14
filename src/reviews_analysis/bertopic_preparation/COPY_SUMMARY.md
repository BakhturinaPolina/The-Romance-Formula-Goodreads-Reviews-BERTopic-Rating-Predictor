# File Copying Summary

## ✅ Files Successfully Copied

### Common Utilities (`src/reviews_analysis/common/`)
- ✅ `config.py` - Configuration loading utilities
- ✅ `logging.py` - Logging setup utilities  
- ✅ `metrics.py` - Metrics computation utilities
- ✅ `training_utils.py` - GPU checking, output directory setup, logging
- ✅ `__init__.py` - Package initialization

### BERTopic Modeling (`src/reviews_analysis/bertopic_modeling/`)
- ✅ `bertopic_runner.py` - Main BERTopic+OCTIS optimization script
- ✅ `retrain_from_tables.py` - Retrain models with optimized hyperparameters
- ✅ `hparam_search.py` - Optimizer class for Bayesian Optimization
- ✅ `__init__.py` - Package initialization

## 📋 Next Steps Required

### 1. Update Import Paths
All copied files currently have import paths referencing:
- `from src.common.` → Should be `from src.reviews_analysis.common.`
- `from src.stage03_modeling.` → Should be `from src.reviews_analysis.bertopic_modeling.`
- `from src.stage04_experiments.` → Should be `from src.reviews_analysis.bertopic_modeling.`

**Files needing import updates:**
- `bertopic_runner.py` - Multiple imports
- `retrain_from_tables.py` - Imports from common
- `hparam_search.py` - Import from common.config (line 140)

### 2. Adapt Data Loading
**In `bertopic_runner.py`:**
- Currently loads from CSV with columns: `Author`, `Book Title`, `Chapter`, `Sentence`
- Needs to load from Parquet with columns: `sentence_text`, `review_id`, `work_id`, `pop_tier`, etc.
- Update lines ~178-204 (CSV reading) to use pandas parquet reading
- Change column reference from `'Sentence'` to `'sentence_text'`

### 3. Update OCTIS Format Conversion
**In `bertopic_runner.py`:**
- Currently creates labels as: `author + "," + book_title` (line 237)
- Should create labels using review metadata:
  - Option 1: Use `work_id` as label
  - Option 2: Use `review_id` as label  
  - Option 3: Use `pop_tier` as label
  - Option 4: Combine `work_id` and `pop_tier`

### 4. Create Test Dataset Sampling Script
Create a new script to sample small datasets from the main parquet file for quick testing:
- Sample N sentences (e.g., 10,000, 50,000, 100,000)
- Maintain distribution across pop_tiers if desired
- Save as parquet for quick iteration

### 5. Create Configuration Files
Create YAML config files similar to `romantic_novels_project_code/configs/`:
- `configs/paths.yaml` - Input/output paths
- `configs/octis.yaml` - OCTIS optimization settings
- `configs/bertopic.yaml` - BERTopic model settings

### 6. Update Path Resolution
**In `config.py`:**
- Currently assumes project root is parent of `src/` (line 63)
- May need adjustment for reviews project structure

## 📝 Notes

- All files copied as-is from `romantic_novels_project_code`
- No modifications made yet - files are in their original state
- Import paths will need updating before use
- Data loading logic needs adaptation for parquet format
- Test with small sample dataset before running on full 8.6M sentences

## 🔍 Files to Review

1. **`bertopic_runner.py`** (678 lines)
   - Main optimization script
   - Contains `BERTopicOctisModelWithEmbeddings` class
   - Handles embedding pre-calculation
   - Runs hyperparameter optimization

2. **`retrain_from_tables.py`** (1032 lines)
   - Retrains models with specific hyperparameters
   - Contains `param_rows_from_tables()` with hardcoded best parameters
   - May need to update parameter values for reviews dataset

3. **`hparam_search.py`** (588 lines)
   - Optimizer class for Bayesian Optimization
   - Mostly self-contained, just needs import path fix

## 📂 Directory Structure Created

```
src/reviews_analysis/
├── bertopic_modeling/          # NEW
│   ├── __init__.py
│   ├── bertopic_runner.py
│   ├── retrain_from_tables.py
│   └── hparam_search.py
├── common/                     # NEW
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py
│   ├── metrics.py
│   └── training_utils.py
└── bertopic_preparation/       # EXISTING
    ├── BERTOPIC_OCTIS_PIPELINE_DESCRIPTION.md
    ├── COPY_PLAN.md
    ├── COPY_SUMMARY.md
    └── ...
```

