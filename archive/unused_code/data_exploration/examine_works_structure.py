#!/usr/bin/env python3
import gzip
import json

# Examine works data structure
sample_works = []
with gzip.open('data/raw/goodreads_book_works.json.gz', 'rt') as f:
    for i, line in enumerate(f):
        if i < 5:
            sample_works.append(json.loads(line.strip()))
        else:
            break

print('Sample works data structure:')
for i, work in enumerate(sample_works):
    print(f'Work {i+1}:')
    for key, value in work.items():
        print(f'  {key}: {value}')
    print()
