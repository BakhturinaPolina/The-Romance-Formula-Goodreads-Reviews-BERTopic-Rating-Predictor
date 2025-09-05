# Null Value Analysis Report

**Date**: 2025-09-04  
**Issue**: Null values appearing in CSV output despite 0% nulls in raw JSON.gz files  
**Root Cause**: Data type conversion with `pd.to_numeric(..., errors='coerce')`

## Problem Summary

The JSON.gz files show 0% null values when inspected directly, but the CSV building process introduces nulls during data processing. This creates a discrepancy between the raw data quality and the processed output.

## Root Cause Analysis

### 1. **Data Type Conversion Issues**

The primary source of null values is the `pd.to_numeric(..., errors='coerce')` conversion in the CSV builder:

```python
# Lines 171-183 in final_csv_builder_working.py
df['work_id'] = pd.to_numeric(df['work_id'], errors='coerce')
df['book_id'] = pd.to_numeric(df['book_id'], errors='coerce')
df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')  # ← 326 nulls
df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')                # ← 415 nulls
df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce')
df['text_reviews_count'] = pd.to_numeric(df['text_reviews_count'], errors='coerce')
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
```

### 2. **Specific Issues Identified**

From the diagnostic analysis of 1,000 sample records:

- **`publication_year`**: 326 nulls introduced (32.6% of records)
  - Problematic values: Empty strings `''`
  - Original data: Contains empty strings that can't be converted to numeric

- **`num_pages`**: 415 nulls introduced (41.5% of records)
  - Problematic values: Empty strings `''`
  - Original data: Contains empty strings that can't be converted to numeric

- **Other fields**: No nulls introduced (work_id, book_id, ratings_count, etc.)

### 3. **Why This Happens**

1. **Raw Data Structure**: The JSON.gz files contain empty strings `""` for missing values
2. **Pandas Behavior**: `pd.to_numeric(..., errors='coerce')` converts non-numeric strings to `NaN`
3. **Empty String Handling**: Empty strings `""` are not considered null in JSON, but become null when converted to numeric

## Impact Analysis

### Data Loss
- **Publication Year**: 32.6% of records lose publication year information
- **Page Count**: 41.5% of records lose page count information
- **Total Impact**: Significant data loss in key fields

### Processing Implications
- Works without publication years are filtered out (2000-2020 range filter)
- Works without page counts have incomplete metadata
- Aggregation calculations may be affected

## Solutions

### 1. **Immediate Fix: Handle Empty Strings**

Modify the data type conversion to handle empty strings before numeric conversion:

```python
def safe_numeric_conversion(series, default_value=None):
    """Convert series to numeric, handling empty strings."""
    # Replace empty strings with None first
    series_cleaned = series.replace('', None)
    # Convert to numeric
    return pd.to_numeric(series_cleaned, errors='coerce').fillna(default_value)

# Usage:
df['publication_year'] = safe_numeric_conversion(df['publication_year'], default_value=0)
df['num_pages'] = safe_numeric_conversion(df['num_pages'], default_value=0)
```

### 2. **Enhanced Data Cleaning**

Add explicit data cleaning before type conversion:

```python
def clean_numeric_field(series, field_name):
    """Clean numeric fields before conversion."""
    # Handle empty strings
    series = series.replace('', None)
    
    # Handle other common non-numeric values
    if field_name == 'publication_year':
        series = series.replace(['Unknown', 'N/A', 'TBD'], None)
    elif field_name == 'num_pages':
        series = series.replace(['Unknown', 'N/A', 'TBD'], None)
    
    return pd.to_numeric(series, errors='coerce')
```

### 3. **Validation and Logging**

Add validation to track data quality issues:

```python
def validate_numeric_conversion(original_series, converted_series, field_name):
    """Validate numeric conversion and log issues."""
    nulls_introduced = converted_series.isnull().sum() - original_series.isnull().sum()
    
    if nulls_introduced > 0:
        logger.warning(f"{field_name}: {nulls_introduced} nulls introduced during conversion")
        
        # Log examples of problematic values
        problematic = original_series[converted_series.isnull() & original_series.notnull()]
        logger.debug(f"Problematic values in {field_name}: {problematic.unique()[:10]}")
    
    return converted_series
```

## Recommended Implementation

### 1. **Update CSV Builder**

Modify the `load_books_dataframe` method in `final_csv_builder_working.py`:

```python
def load_books_dataframe(self, file_path: Path) -> pd.DataFrame:
    """Load books data into a pandas DataFrame for efficient processing."""
    # ... existing loading code ...
    
    # Enhanced data type conversion with proper handling
    if 'work_id' in df.columns:
        df['work_id'] = pd.to_numeric(df['work_id'], errors='coerce')
    if 'book_id' in df.columns:
        df['book_id'] = pd.to_numeric(df['book_id'], errors='coerce')
    if 'publication_year' in df.columns:
        # Handle empty strings before conversion
        df['publication_year'] = df['publication_year'].replace('', None)
        df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
    if 'num_pages' in df.columns:
        # Handle empty strings before conversion
        df['num_pages'] = df['num_pages'].replace('', None)
        df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')
    # ... rest of conversions ...
```

### 2. **Add Data Quality Metrics**

Track and report data quality issues:

```python
def _log_data_quality_metrics(self, df: pd.DataFrame):
    """Log data quality metrics after conversion."""
    print(f"📊 Data Quality Metrics:")
    print(f"  - Publication year nulls: {df['publication_year'].isnull().sum()}")
    print(f"  - Page count nulls: {df['num_pages'].isnull().sum()}")
    print(f"  - Work ID nulls: {df['work_id'].isnull().sum()}")
    print(f"  - Book ID nulls: {df['book_id'].isnull().sum()}")
```

## Testing Recommendations

### 1. **Before/After Comparison**

Run the diagnostic script before and after implementing fixes:

```bash
python3 src/data_quality/diagnose_null_sources.py
```

### 2. **Sample Processing**

Test with a small sample to verify the fix:

```python
builder = OptimizedFinalCSVBuilder()
output_path = builder.build_final_csv_optimized(sample_size=100)
```

### 3. **Data Quality Validation**

Add validation to ensure no unexpected nulls are introduced:

```python
def validate_output_quality(df):
    """Validate output data quality."""
    issues = []
    
    if df['work_id'].isnull().any():
        issues.append("Work ID contains nulls")
    
    if df['publication_year'].isnull().sum() > len(df) * 0.5:
        issues.append("Too many null publication years")
    
    return issues
```

## Conclusion

The null values in the CSV output are **not** present in the raw JSON.gz files but are introduced during the data type conversion process. The primary culprits are:

1. **Empty strings** in `publication_year` and `num_pages` fields
2. **`pd.to_numeric(..., errors='coerce')`** converting empty strings to `NaN`

**Solution**: Handle empty strings before numeric conversion to preserve data integrity and reduce null values in the final output.

This explains the discrepancy between the raw data inspection (0% nulls) and the CSV building process (significant nulls in key fields).
