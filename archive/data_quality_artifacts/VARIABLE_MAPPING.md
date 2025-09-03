# Variable Mapping Guide - Outlier Detection Analysis

## Overview
This document provides the exact mapping between expected variable names and actual results structure to prevent KeyError issues.

## Results Structure Mapping

### 1. Title Duplications
**Expected Key**: `results['title_duplications']`
**Actual Structure**:
```json
{
  "title_duplications": {
    "total_duplicate_titles": 5800,
    "total_duplicate_books": "17922",
    "duplicate_titles_list": {...},
    "author_name_analysis": {
      "Title Name": {
        "count": 42,
        "unique_author_names": 42,
        "author_names": ["Author1", "Author2", ...],
        "year_range": 5,
        "is_legitimate": true,
        "strategy": "Different authors - legitimate duplicate",
        "books_data": [...]
      }
    },
    "disambiguation_strategies": [...]
  }
}
```

**Key Fields**:
- ✅ `total_duplicate_titles` (int)
- ✅ `total_duplicate_books` (str/int)
- ✅ `author_name_analysis` (dict) - NOT `author_analysis`
- ✅ `disambiguation_strategies` (list)

### 2. Series Data
**Expected Key**: `results['series_title_data']` - NOT `series_data`
**Actual Structure**:
```json
{
  "series_title_data": {
    "total_series_title_books": 53971,
    "unique_series_title": 21996,
    "series_title_works_count_analysis": {
      "total_series": 21996,
      "series_title_with_discrepancies": 0,
      "accuracy_rate": 100.0
    },
    "missing_books_analysis": {
      "incomplete_series_title": 72,
      "incomplete_series_title_details": {...}
    },
    "correction_recommendations": [...]
  }
}
```

**Key Fields**:
- ✅ `total_series_title_books` (int) - NOT `total_series_books`
- ✅ `unique_series_title` (int) - NOT `unique_series`
- ✅ `series_title_works_count_analysis` (dict)
- ✅ `missing_books_analysis` (dict)

### 3. Statistical Outliers
**Expected Key**: `results['statistical_outliers']`
**Actual Structure**:
```json
{
  "statistical_outliers": {
    "publication_year_outliers": {...},
    "rating_outliers": {...},
    "review_count_outliers": {...},
    "page_count_outliers": {...},
    "summary": {
      "total_outliers_detected": 3013,
      "fields_analyzed": ["publication_year", "rating", "review_count", "page_count"]
    }
  }
}
```

**Key Fields**:
- ✅ `summary.total_outliers_detected` (int)
- ✅ `summary.fields_analyzed` (list)

### 4. Treatment Recommendations
**Expected Key**: `results['treatment_recommendations']`
**Actual Structure**:
```json
{
  "treatment_recommendations": {
    "title_duplications": [...],
    "series_data": [...],
    "statistical_outliers": [...],
    "priority_actions": [
      {
        "issue": "Found 5800 duplicate titles",
        "action": "Review each duplicate for legitimacy",
        "priority": "High"
      }
    ],
    "research_impact": {...}
  }
}
```

**Key Fields**:
- ✅ `priority_actions` (list)
- ✅ `research_impact` (dict)

## Common Mistakes to Avoid

### ❌ WRONG - Will Cause KeyError:
```python
# Wrong key names
results['series_data']  # Should be 'series_title_data'
results['author_analysis']  # Should be 'author_name_analysis'
results['total_series_books']  # Should be 'total_series_title_books'
results['unique_series']  # Should be 'unique_series_title'
```

### ✅ CORRECT - Will Work:
```python
# Correct key names
results['series_title_data']
results['author_name_analysis']
results['series_title_data']['total_series_title_books']
results['series_title_data']['unique_series_title']
```

## Validation Function

The enhanced runner includes a validation function that checks all these mappings:

```python
def validate_results_structure(results: Dict[str, Any], logger: logging.Logger) -> bool:
    """Validate the complete results structure to prevent KeyError issues."""
    expected_structure = {
        'title_duplications': {
            'total_duplicate_titles': int,
            'total_duplicate_books': (int, str),
            'author_name_analysis': dict
        },
        'series_title_data': {
            'total_series_title_books': int,
            'unique_series_title': int,
            'series_title_works_count_analysis': dict,
            'missing_books_analysis': dict
        },
        'statistical_outliers': {
            'summary': {
                'total_outliers_detected': int,
                'fields_analyzed': list
            }
        },
        'treatment_recommendations': {
            'priority_actions': list,
            'research_impact': dict
        }
    }
    # ... validation logic
```

## Usage Examples

### Safe Access Pattern:
```python
# Always use .get() with defaults
title_count = results.get('title_duplications', {}).get('total_duplicate_titles', 0)
series_count = results.get('series_title_data', {}).get('total_series_title_books', 0)

# Or validate first
if 'series_title_data' in results and 'total_series_title_books' in results['series_title_data']:
    series_count = results['series_title_data']['total_series_title_books']
else:
    series_count = 0
```

### Enhanced Runner Usage:
```bash
# Use the enhanced runner for better error prevention
python src/data_quality/run_outlier_detection_enhanced.py
```

## Maintenance

When updating the analysis script:
1. Update this mapping document
2. Update the validation function
3. Test with the enhanced runner
4. Verify all field names match exactly

This prevents future KeyError issues and ensures consistent data access patterns.
