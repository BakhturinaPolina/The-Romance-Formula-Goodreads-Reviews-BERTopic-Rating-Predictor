#!/usr/bin/env python3
"""
Short script to review Goodreads romance books dataset.
Shows variable names, sample records, and basic statistics.
"""

import gzip
import json
from collections import Counter
from pathlib import Path

# Dataset path (go up to project root)
dataset_path = Path(__file__).parent.parent.parent / "data/raw/goodreads_books_romance.json.gz"

print(f"Reviewing dataset: {dataset_path}")
print("=" * 80)

# Track all keys seen across records
all_keys = Counter()
sample_records = []
total_records = 0
max_samples = 5

# Read and analyze
with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if line.strip():
            try:
                record = json.loads(line)
                all_keys.update(record.keys())
                
                # Save first few records as samples
                if len(sample_records) < max_samples:
                    sample_records.append(record)
                
                total_records += 1
                
                # Limit reading for quick review (remove this to process full dataset)
                if total_records >= 1000:
                    break
                    
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error at line {i+1}: {e}")
                continue

print(f"\nTotal records reviewed: {total_records}")
print(f"\nVariable names (keys) found ({len(all_keys)} unique):")
print("-" * 80)
for key, count in sorted(all_keys.items()):
    print(f"  {key:40s} (appears in {count:4d} records)")

print(f"\n\nSample records (first {len(sample_records)}):")
print("=" * 80)
for i, record in enumerate(sample_records, 1):
    print(f"\n--- Record {i} ---")
    for key, value in record.items():
        # Truncate long values for display
        value_str = str(value)
        if len(value_str) > 100:
            value_str = value_str[:97] + "..."
        print(f"  {key}: {value_str}")

print("\n" + "=" * 80)
print("Review complete!")

