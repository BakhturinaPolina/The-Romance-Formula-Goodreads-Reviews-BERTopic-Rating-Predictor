#!/usr/bin/env python3
import json
import gzip

count = 0
with gzip.open('data/raw/goodreads_books_romance.json.gz', 'rt') as f:
    for line in f:
        if count < 10:
            data = json.loads(line)
            pub_year = data.get('publication_year', 'MISSING')
            print(f'Book {count}: publication_year = {pub_year} (type: {type(pub_year)})')
            count += 1
        else:
            break
