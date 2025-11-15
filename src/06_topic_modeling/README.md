# Stage 06: Topic Modeling

## Purpose

This stage performs topic modeling using BERTopic with OCTIS integration, including hyperparameter optimization.

**Note**: Data preparation (sentence splitting, corpus creation) has been moved to `05_prepare_reviews_corpus_for_BERTopic/`. This module focuses solely on BERTopic+OCTIS modeling.

## ⚠️ Virtual Environment Requirement

**ALL scripts MUST be run using the virtual environment at `romance-novel-nlp-research/.venv`.**

If running Python scripts directly, always use:

```bash
romance-novel-nlp-research/.venv/bin/python3 script.py
```

**Never use system Python or other virtual environments - always use `romance-novel-nlp-research/.venv` first.**

For full project-wide rules, see `.cursor/rules/venv-requirement.mdc`.

## Structure

```
06_topic_modeling/
├── core/                          # Main modeling logic
│   ├── __init__.py
│   ├── bertopic_plus_octis.py    # Main BERTopic+OCTIS script
│   ├── optimizer.py              # OCTIS optimizer configuration
│   └── load_raw_sentences.py     # Load sentences for BERTopic
├── scripts/                       # Execution and utility scripts
│   ├── __init__.py
│   ├── restart_script.py         # Resume interrupted runs
│   ├── sample_test_dataset.py    # Create test datasets
│   └── topic_npy_to_json.py      # Convert topic outputs to JSON
├── config/                        # Configuration files
│   └── config_bertopic_reviews.yaml  # BERTopic configuration
├── docs/                          # Documentation
│   ├── README.md                  # BERTopic_OCTIS documentation
│   └── CONFIG_USAGE.md            # Configuration usage guide
├── __init__.py                    # Module exports
├── README.md                      # This file
└── README_SCIENTIFIC.md           # Scientific documentation
```

## Input Files

- `data/processed/review_sentences_for_bertopic.parquet` - Sentence-level review data (prepared by `05_prepare_reviews_corpus_for_BERTopic/`)
- Configuration files in `config/`

## Output Files

- `outputs/topic_models/*.json` - Topic model results
- `outputs/reports/*_topic_modeling_*.json` - Analysis reports
- `data/intermediate/octis_reviews/` - OCTIS-formatted data

## How to Run

### Topic Modeling with OCTIS

```bash
cd src/06_topic_modeling
romance-novel-nlp-research/.venv/bin/python3 core/bertopic_plus_octis.py
```

### Hyperparameter Optimization

```bash
cd src/06_topic_modeling
romance-novel-nlp-research/.venv/bin/python3 core/optimizer.py
```

### Create Test Dataset

```bash
cd src/06_topic_modeling
romance-novel-nlp-research/.venv/bin/python3 scripts/sample_test_dataset.py
```

### Resume Interrupted Run

```bash
cd src/06_topic_modeling
romance-novel-nlp-research/.venv/bin/python3 scripts/restart_script.py
```

## Dependencies

- BERTopic
- OCTIS
- sentence-transformers
- pandas
- numpy
- spacy

## Example Usage

```bash
# Run topic modeling (with venv)
romance-novel-nlp-research/.venv/bin/python3 core/bertopic_plus_octis.py

# Optimize hyperparameters (with venv)
romance-novel-nlp-research/.venv/bin/python3 core/optimizer.py

# Create test dataset
romance-novel-nlp-research/.venv/bin/python3 scripts/sample_test_dataset.py --n_samples 10000
```

## Key Features

- BERTopic topic modeling
- OCTIS framework integration
- Hyperparameter optimization
- Multiple embedding models
- Topic coherence evaluation

## Related Modules

- **`05_prepare_reviews_corpus_for_BERTopic/`**: Prepares sentence-level corpus from reviews
  - Run this first to create `review_sentences_for_bertopic.parquet`
  - Contains data loading utilities and coverage analysis
