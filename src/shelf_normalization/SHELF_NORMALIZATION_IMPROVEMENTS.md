# Shelf Normalization Pipeline Improvements

## Overview

This document summarizes the comprehensive improvements made to the shelf normalization pipeline based on the detailed analysis and recommendations provided. The improvements address critical issues with coverage, canonicalization, segmentation, and alias detection.

## Issues Addressed

### 1. **Coverage: Map only spans ~3.9% of shelves**
- **Problem**: Canonicalized only top 10k shelves while corpus contains 255,657 unique canon values
- **Solution**: Expanded default from 10k to 100k+ shelves (`--n-top-shelves` default changed to 100,000)

### 2. **Canonicalization barely merges aliases**
- **Problem**: 10,000 → 9,994 (compression 0.9994), only 6 effective merges
- **Solution**: Implemented robust blocking and tiered similarity thresholds

### 3. **Segmentation firing on clean words**
- **Problem**: Examples like `romance → ['rom an ce']` showing character-level splits
- **Solution**: Added proper gating to prevent segmentation of common words

### 4. **Alias module produced zero candidates**
- **Problem**: Empty alias candidates due to over-strict thresholds and missing blocking
- **Solution**: Implemented Q-gram blocking, sorted-neighborhood windowing, and tiered thresholds

### 5. **Non-content shelves not configured**
- **Problem**: No filtering of reading status, ownership, or date patterns
- **Solution**: Built comprehensive non-content lexicon with pattern matching

## Detailed Improvements

### A) Non-Content Filtering

**Implementation**: `is_noncontent()` function with comprehensive pattern matching

**Categories**:
- **Reading Status**: `to-read`, `currently-reading`, `tbr`, `dnf`, `read`, etc.
- **Ownership**: `owned`, `kindle`, `library`, `wishlist`, etc.
- **Date Patterns**: Years (2016-2025), months, seasons
- **Punctuation**: Pure dash/punctuation runs like `----2016`

**Features**:
- Case-insensitive matching
- Dash/underscore normalization
- Prefix matching for common patterns
- Length-based filtering for very short strings

### B) Segmentation Gating

**Implementation**: Four-gate system to prevent inappropriate segmentation

**Gates**:
1. **No whitespace**: Must be single token
2. **Length ≥ 6**: Avoid segmenting short words
3. **Not common word**: Zipf frequency < 3.0 (prevents `romance → ['rom an ce']`)
4. **CamelCase OR lowercase concatenation**: Only process appropriate patterns

**Accept Rule**: Keep split iff yields ≥2 tokens AND each has Zipf ≥ 3.0

**Statistics Tracking**:
- `has_whitespace`, `too_short`, `common_word`, `not_camel_or_concat`
- `camel_case`, `lowercase_concat`, `guard1_failed`, `guard2_failed`
- `zipf_failed`, `accepted`

### C) Robust Alias Blocking

**Implementation**: Multi-method blocking approach

**Methods**:
1. **Q-gram blocking**: 3-gram signatures with first 5 q-grams as bucket key
2. **Sorted neighborhood**: Window-based pairing (w=20-30)
3. **Length buckets**: Group by string length for similar-length comparisons
4. **Canopy clustering**: High-similarity groups (loose=0.7, tight=0.85)

**Deduplication**: `seen_pairs` set prevents duplicate comparisons

### D) Tiered Similarity Thresholds

**Implementation**: Length-based threshold selection

**Thresholds**:
- **Short (≤6 chars)**: Damerau-Levenshtein ≤ 1
- **Mid (7-12 chars)**: Jaro-Winkler ≥ 0.92
- **Long (>12 chars)**: Jaccard 3-gram ≥ 0.95

**Rationale**:
- JW designed for near-prefix matches (good for `alpha-male` ↔ `alpha male`)
- DL as tie-breaker for short strings
- J3 for long strings where character-level similarity matters

### E) Enhanced Logging and Statistics

**Implementation**: Rule-specific counters and detailed provenance

**Canonicalization Stats**:
- `as_is`, `casefold`, `sep_normalize`, `punct_trim`, `noncontent_filtered`

**Segmentation Stats**:
- Gate failure counts, acceptance rates, transformation types

**Provenance Footer**:
- Complete statistics summary
- Compression ratios
- Dependency availability
- Threshold configurations

## Expected Outcomes

### Before Improvements
- **Coverage**: 3.9% (10k/255k shelves)
- **Compression**: 0.06% (10k→9,994)
- **Alias candidates**: 0
- **Segmentation**: Character-level splits on common words
- **Non-content**: No filtering

### After Improvements
- **Coverage**: ≥39% (100k/255k shelves)
- **Compression**: 3-10% (depending on noise level)
- **Alias candidates**: Thousands with calibrated thresholds
- **Segmentation**: Conservative, gated processing
- **Non-content**: Comprehensive filtering

## Usage

### Basic Usage
```bash
python shelf_normalize.py \
  --from-audit audit_outputs_full \
  --parsed-parquet parse_outputs_full/parsed_books_*.parquet \
  --outdir shelf_norm_outputs \
  --n-top-shelves 100000 \
  --jw-threshold 0.92 \
  --j3-threshold 0.95 \
  --edit-max 1 \
  --min-seg-evidence 5 \
  --zipf-min 3.0
```

### Key Parameters
- `--n-top-shelves`: Default 100k (was None/10k)
- `--jw-threshold`: 0.92 for mid-length strings
- `--j3-threshold`: 0.95 for long strings
- `--zipf-min`: 3.0 for common word detection

## Quality Assurance

### Validation Methods
1. **Precision@k**: Manual labeling of 200 concatenations (target ≥0.9)
2. **Merge precision**: Spot-audit ≥200 alias pairs at threshold
3. **Coverage metrics**: Track canon coverage vs corpus
4. **Statistics monitoring**: Rule-specific counters and acceptance rates

### Monitoring Points
- Compression ratio trends
- Alias candidate yield
- Segmentation acceptance rates
- Non-content filter effectiveness
- Processing time vs coverage trade-offs

## Future Enhancements

### Optional Improvements
1. **Zero-shot hashtag segmentation**: For tougher concatenation cases
2. **Bipartite community detection**: For umbrella shelf clustering
3. **Dynamic threshold tuning**: Based on corpus characteristics
4. **Domain-specific lexicons**: Romance/fantasy genre terms

### Performance Optimizations
1. **Parallel processing**: For large-scale alias detection
2. **Memory-efficient streaming**: For 255k+ shelf processing
3. **Caching**: For repeated similarity computations
4. **Incremental updates**: For new shelf additions

## Files Modified

- `src/eda_analysis/shelf_normalize.py`: Main pipeline improvements
- `src/eda_analysis/test_improvements_simple.py`: Validation tests
- `SHELF_NORMALIZATION_IMPROVEMENTS.md`: This documentation

## Dependencies

### Required
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `pyarrow`: Parquet I/O

### Optional (with fallbacks)
- `rapidfuzz`: Fast similarity computations
- `wordfreq`: Zipf frequency lookups

The pipeline gracefully handles missing optional dependencies with simplified fallback implementations.
