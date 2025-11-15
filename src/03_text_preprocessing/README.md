# Stage 03: Text Preprocessing

## Purpose

This stage performs NLP text preprocessing including HTML cleaning, text normalization, and genre categorization.

## Input Files

- `data/processed/*.csv` - Datasets from Stage 02

## Output Files

- `data/processed/*_preprocessed.csv` - Preprocessed datasets
- `outputs/reports/*_preprocessing_report_*.json` - Preprocessing reports

## How to Run

### Main Preprocessing

```bash
cd src/03_text_preprocessing
python run_preprocessor.py
```

### Testing

```bash
cd src/03_text_preprocessing
python test_preprocessor.py
```

## Dependencies

- pandas
- BeautifulSoup4 (for HTML cleaning)
- re (standard library)

## Example Usage

```bash
# Run preprocessing on main dataset
python run_preprocessor.py

# Test preprocessing on sample
python test_preprocessor.py
```

## Key Features

- HTML tag and entity removal
- Text normalization and standardization
- Genre categorization
- Shelf tag normalization

