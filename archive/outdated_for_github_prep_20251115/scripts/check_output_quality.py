#!/usr/bin/env python3
"""
Quality and completeness check for review_sentences_for_bertopic.parquet output.

This script validates:
1. File existence and basic structure
2. Required columns presence
3. Data completeness (no unexpected missing values)
4. Data integrity (sentence_id continuity, relationships)
5. Expected statistics and distributions
"""

import pandas as pd
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = DATA_PROCESSED / "review_sentences_for_bertopic.parquet"

# Expected columns from the script
EXPECTED_COLUMNS = [
    'sentence_id',
    'sentence_text',
    'review_id',
    'work_id',
    'pop_tier',
    'rating',
    'sentence_index',
    'n_sentences_in_review'
]

def check_file_exists():
    """Check if output file exists."""
    if not OUTPUT_FILE.exists():
        print(f"❌ ERROR: Output file not found: {OUTPUT_FILE}")
        return False
    print(f"✅ File exists: {OUTPUT_FILE}")
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size_mb:.2f} MB")
    return True

def check_basic_structure(df):
    """Check basic DataFrame structure."""
    print("\n=== BASIC STRUCTURE ===")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    
    # Check for expected columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    
    if missing_cols:
        print(f"❌ Missing expected columns: {missing_cols}")
        return False
    if extra_cols:
        print(f"⚠️  Extra columns (not expected): {extra_cols}")
    else:
        print("✅ All expected columns present")
    
    return True

def check_data_types(df):
    """Check data types are appropriate."""
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    
    issues = []
    if df['sentence_id'].dtype not in ['int64', 'int32', 'int']:
        issues.append(f"sentence_id should be integer, got {df['sentence_id'].dtype}")
    if df['sentence_text'].dtype != 'object':
        issues.append(f"sentence_text should be string/object, got {df['sentence_text'].dtype}")
    if df['review_id'].dtype not in ['int64', 'int32', 'int', 'object']:
        issues.append(f"review_id unexpected type: {df['review_id'].dtype}")
    if df['work_id'].dtype not in ['int64', 'int32', 'int', 'object']:
        issues.append(f"work_id unexpected type: {df['work_id'].dtype}")
    
    if issues:
        print("⚠️  Data type issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Data types look appropriate")
        return True

def check_missing_values(df):
    """Check for missing values in critical columns."""
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_nonzero = missing[missing > 0]
    
    if len(missing_nonzero) == 0:
        print("✅ No missing values")
        return True
    
    print("Missing values by column:")
    for col, count in missing_nonzero.items():
        pct = count / len(df) * 100
        print(f"   {col}: {count:,} ({pct:.2f}%)")
    
    # Check critical columns
    critical_cols = ['sentence_id', 'sentence_text', 'review_id', 'work_id', 'pop_tier']
    critical_missing = [col for col in critical_cols if missing[col] > 0]
    
    if critical_missing:
        print(f"❌ ERROR: Missing values in critical columns: {critical_missing}")
        return False
    
    # rating and sentence_index can have missing values
    print("✅ Critical columns have no missing values")
    return True

def check_sentence_id_integrity(df):
    """Check sentence_id is unique and continuous."""
    print("\n=== SENTENCE ID INTEGRITY ===")
    
    # Check uniqueness
    n_unique = df['sentence_id'].nunique()
    n_total = len(df)
    if n_unique != n_total:
        print(f"❌ ERROR: sentence_id not unique! {n_unique:,} unique values for {n_total:,} rows")
        duplicates = df[df['sentence_id'].duplicated(keep=False)]
        print(f"   Found {len(duplicates):,} duplicate sentence_ids")
        return False
    print(f"✅ sentence_id is unique ({n_unique:,} unique values)")
    
    # Check continuity (should be 0 to n-1)
    min_id = df['sentence_id'].min()
    max_id = df['sentence_id'].max()
    expected_max = n_total - 1
    
    if min_id != 0:
        print(f"⚠️  WARNING: sentence_id starts at {min_id}, expected 0")
    
    if max_id != expected_max:
        print(f"⚠️  WARNING: sentence_id max is {max_id}, expected {expected_max}")
    
    # Check for gaps
    all_ids = set(range(min_id, max_id + 1))
    actual_ids = set(df['sentence_id'])
    gaps = all_ids - actual_ids
    
    if gaps:
        print(f"⚠️  WARNING: Found {len(gaps)} gaps in sentence_id sequence")
        if len(gaps) <= 10:
            print(f"   Gaps: {sorted(gaps)[:10]}")
    else:
        print(f"✅ sentence_id is continuous (range: {min_id} to {max_id})")
    
    return True

def check_relationships(df):
    """Check relationships between columns."""
    print("\n=== RELATIONSHIP INTEGRITY ===")
    
    # Check sentence_index is within bounds
    invalid_indices = df[df['sentence_index'] >= df['n_sentences_in_review']]
    if len(invalid_indices) > 0:
        print(f"❌ ERROR: Found {len(invalid_indices)} sentences with sentence_index >= n_sentences_in_review")
        return False
    print("✅ sentence_index is within bounds")
    
    # Check that n_sentences_in_review matches actual counts per review
    actual_counts = df.groupby('review_id').size()
    expected_counts = df.groupby('review_id')['n_sentences_in_review'].first()
    
    mismatches = actual_counts != expected_counts
    if mismatches.any():
        n_mismatches = mismatches.sum()
        print(f"❌ ERROR: Found {n_mismatches} reviews where n_sentences_in_review doesn't match actual count")
        return False
    print("✅ n_sentences_in_review matches actual sentence counts per review")
    
    return True

def check_data_completeness(df):
    """Check data completeness statistics."""
    print("\n=== DATA COMPLETENESS ===")
    
    print(f"Total sentences: {len(df):,}")
    print(f"Unique reviews: {df['review_id'].nunique():,}")
    print(f"Unique works: {df['work_id'].nunique():,}")
    print(f"Unique sentence texts: {df['sentence_text'].nunique():,}")
    
    # Check pop_tier distribution
    if 'pop_tier' in df.columns:
        print(f"\nPop tier distribution:")
        tier_counts = df['pop_tier'].value_counts()
        for tier, count in tier_counts.items():
            pct = count / len(df) * 100
            print(f"   {tier}: {count:,} ({pct:.1f}%)")
    
    # Check sentence length
    sentence_lengths = df['sentence_text'].str.len()
    print(f"\nSentence length statistics:")
    print(f"   Min: {sentence_lengths.min()} chars")
    print(f"   Max: {sentence_lengths.max()} chars")
    print(f"   Mean: {sentence_lengths.mean():.1f} chars")
    print(f"   Median: {sentence_lengths.median():.1f} chars")
    
    # Check for empty sentences
    empty_sentences = df[df['sentence_text'].str.len() == 0]
    if len(empty_sentences) > 0:
        print(f"❌ ERROR: Found {len(empty_sentences)} empty sentences")
        return False
    print("✅ No empty sentences")
    
    # Check minimum length (should be >= 10 after cleaning)
    too_short = df[df['sentence_text'].str.len() < 10]
    if len(too_short) > 0:
        print(f"⚠️  WARNING: Found {len(too_short)} sentences shorter than 10 characters")
        print(f"   This may indicate cleaning issues")
    else:
        print("✅ All sentences meet minimum length requirement (>= 10 chars)")
    
    return True

def check_sample_data(df):
    """Display sample data for manual inspection."""
    print("\n=== SAMPLE DATA (first 5 rows) ===")
    sample_cols = ['sentence_id', 'sentence_text', 'review_id', 'work_id', 'pop_tier']
    sample = df[sample_cols].head(5)
    for idx, row in sample.iterrows():
        text_preview = row['sentence_text'][:80] + "..." if len(row['sentence_text']) > 80 else row['sentence_text']
        print(f"\n[{row['sentence_id']}] {text_preview}")
        print(f"    review_id={row['review_id']}, work_id={row['work_id']}, tier={row['pop_tier']}")

def main():
    """Run all quality checks."""
    print("=" * 80)
    print("OUTPUT QUALITY AND COMPLETENESS CHECK")
    print("=" * 80)
    print(f"File: {OUTPUT_FILE}")
    print()
    
    # Check file exists
    if not check_file_exists():
        sys.exit(1)
    
    # Load data
    print("\nLoading data...")
    try:
        df = pd.read_parquet(OUTPUT_FILE)
        print(f"✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to load parquet file: {e}")
        sys.exit(1)
    
    # Run checks
    checks_passed = []
    
    checks_passed.append(("Basic structure", check_basic_structure(df)))
    checks_passed.append(("Data types", check_data_types(df)))
    checks_passed.append(("Missing values", check_missing_values(df)))
    checks_passed.append(("Sentence ID integrity", check_sentence_id_integrity(df)))
    checks_passed.append(("Relationships", check_relationships(df)))
    checks_passed.append(("Data completeness", check_data_completeness(df)))
    
    # Show sample
    check_sample_data(df)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in checks_passed if result)
    total = len(checks_passed)
    
    for check_name, result in checks_passed:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All quality checks passed!")
        return 0
    else:
        print("❌ Some quality checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

