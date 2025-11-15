# Stage 08: Corpus Analysis

## Purpose

This stage generates corpus statistics and performs statistical analysis of the dataset.

## Input Files

- `data/processed/*.csv` - Final processed datasets
- Review datasets

## Output Files

- `outputs/reports/*_corpus_statistics_*.csv` - Corpus statistics
- `outputs/reports/*_corpus_analysis_*.md` - Analysis reports
- `outputs/visualizations/*_corpus_*.png` - Visualization plots

## How to Run

```bash
cd src/08_corpus_analysis
python generate_corpus_statistics.py
```

## Dependencies

- pandas
- numpy
- matplotlib (for visualizations)
- seaborn (for visualizations)

## Example Usage

```bash
# Generate corpus statistics
python generate_corpus_statistics.py \
    --input data/processed/main_dataset.csv \
    --output outputs/reports/corpus_statistics.csv
```

## Key Features

- Corpus size and composition statistics
- Distribution analysis
- Cross-corpus comparisons
- Statistical test results

