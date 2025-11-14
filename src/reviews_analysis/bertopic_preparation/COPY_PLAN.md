# File Copying Plan for BERTopic+OCTIS Pipeline

## Files to Copy from `romantic_novels_project_code`

### Core Modeling Files
1. **`src/stage03_modeling/bertopic_runner.py`**
   - Main script for BERTopic+OCTIS optimization
   - Contains BERTopicOctisModelWithEmbeddings class
   - Handles embedding pre-calculation
   - Runs hyperparameter optimization
   - **Destination**: `src/reviews_analysis/bertopic_modeling/bertopic_runner.py`

2. **`src/stage03_modeling/retrain_from_tables.py`**
   - Retrains BERTopic models with optimized hyperparameters
   - **Destination**: `src/reviews_analysis/bertopic_modeling/retrain_from_tables.py`

### Optimizer Class
3. **`src/stage04_experiments/hparam_search.py`**
   - Contains Optimizer class for Bayesian Optimization
   - **Destination**: `src/reviews_analysis/bertopic_modeling/hparam_search.py`

### Common Utilities
4. **`src/common/config.py`**
   - Configuration loading utilities
   - **Destination**: `src/reviews_analysis/common/config.py` (or adapt existing)

5. **`src/common/training_utils.py`**
   - GPU checking, output directory setup, logging setup
   - **Destination**: `src/reviews_analysis/common/training_utils.py` (or adapt existing)

6. **`src/common/metrics.py`**
   - Metrics computation (may need topic-specific metrics)
   - **Destination**: `src/reviews_analysis/common/metrics.py` (or adapt existing)

7. **`src/common/logging.py`**
   - Logging setup utilities
   - **Destination**: `src/reviews_analysis/common/logging.py` (or adapt existing)

## Directory Structure to Create

```
src/reviews_analysis/
├── bertopic_modeling/          # NEW - Main modeling directory
│   ├── __init__.py
│   ├── bertopic_runner.py      # Copied from stage03_modeling
│   ├── retrain_from_tables.py  # Copied from stage03_modeling
│   └── hparam_search.py        # Copied from stage04_experiments
├── common/                     # NEW - Common utilities
│   ├── __init__.py
│   ├── config.py
│   ├── training_utils.py
│   ├── metrics.py
│   └── logging.py
└── bertopic_preparation/       # EXISTING
    └── ...
```

## Adaptations Needed After Copying

1. **Data Loading**
   - Change from CSV to Parquet
   - Update column names: `Sentence` → `sentence_text`
   - Handle review metadata: `review_id`, `work_id`, `pop_tier`

2. **OCTIS Format Conversion**
   - Adapt label creation to use review metadata
   - Current: `author + "," + book_title`
   - New: `work_id` or `review_id` or `pop_tier`

3. **Config Paths**
   - Update config loading to use reviews project structure
   - May need to create `configs/` directory with YAML files

4. **Import Paths**
   - Update all `from src.common.` imports
   - Update all `from src.stage03_modeling.` imports

5. **Test Dataset Sampling**
   - Create script to sample small test datasets from main parquet
   - For quick testing before running on full 8.6M sentences

## Files NOT to Copy (Project-Specific)

- `src/stage03_modeling/main.py` - CLI entry point, project-specific
- `src/common/io.py` - May have project-specific data loading
- `src/common/seed.py` - Simple, can recreate if needed
- `src/common/hw_utils.py` - Hardware utilities, may not be needed
- `src/common/check_gpu_setup.py` - Standalone script, can copy if needed

## Next Steps After Copying

1. Create test dataset sampling script
2. Adapt data loading in bertopic_runner.py
3. Update OCTIS format conversion
4. Test with small sample dataset
5. Create config files for reviews project
6. Update import paths throughout

