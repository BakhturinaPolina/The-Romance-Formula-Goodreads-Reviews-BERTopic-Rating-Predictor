#!/usr/bin/env python3
"""
Load OpenLibrary data from the dump file into Elasticsearch
Focuses on editions and works for book title matching
"""

import json
import sys
import gzip
from elasticsearch import Elasticsearch
from tqdm import tqdm
import argparse

def load_openlibrary_data_to_elasticsearch(dump_file, index_name="openlibrary_books", max_records=None):
    """Load OpenLibrary data from dump file into Elasticsearch"""
    
    # Connect to Elasticsearch
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    
    # Check if index exists, create if not
    if not es.indices.exists(index=index_name):
        print(f"Creating index: {index_name}")
        es.indices.create(index=index_name)
    
    # Count total lines for progress bar (approximate)
    print("Counting lines in dump file...")
    total_lines = 0
    with gzip.open(dump_file, 'rt', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            if total_lines > 100000:  # Sample first 100k lines for estimation
                break
    
    # Estimate total based on sample
    estimated_total = total_lines * 10  # Rough estimate
    if max_records:
        estimated_total = min(estimated_total, max_records)
    
    print(f"Loading OpenLibrary data (estimated {estimated_total:,} records)...")
    
    # Load data
    records_loaded = 0
    with gzip.open(dump_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, total=estimated_total):
            if max_records and records_loaded >= max_records:
                break
                
            try:
                # Parse tab-separated line
                parts = line.strip().split('\t', 4)
                if len(parts) < 5:
                    continue
                
                record_type, ol_key, revision, last_modified, json_data = parts
                
                # Only process editions and works (books)
                if record_type not in ['/type/edition', '/type/work']:
                    continue
                
                # Parse JSON data
                data = json.loads(json_data)
                
                # Extract book information
                book_info = {
                    'ol_key': ol_key,
                    'type': record_type,
                    'revision': int(revision),
                    'last_modified': last_modified,
                    'data': data
                }
                
                # Extract title and author for easier searching
                title = None
                authors = []
                
                if record_type == '/type/edition':
                    title = data.get('title', '')
                    # Get authors from work reference
                    works = data.get('works', [])
                    if works:
                        work_key = works[0].get('key', '')
                        book_info['work_key'] = work_key
                elif record_type == '/type/work':
                    title = data.get('title', '')
                    authors_data = data.get('authors', [])
                    for author in authors_data:
                        if isinstance(author, dict) and 'author' in author:
                            author_key = author['author'].get('key', '')
                            book_info['author_keys'] = book_info.get('author_keys', []) + [author_key]
                
                # Only index if we have a title
                if title:
                    book_info['title'] = title
                    book_info['authors'] = authors
                    
                    # Use ol_key as document ID
                    es.index(index=index_name, id=ol_key, body=book_info)
                    records_loaded += 1
                
            except json.JSONDecodeError as e:
                continue
            except Exception as e:
                print(f"Error processing record: {e}")
                continue
    
    # Refresh index to make data searchable
    es.indices.refresh(index=index_name)
    
    # Get final count
    count = es.count(index=index_name)
    print(f"Successfully loaded {count['count']:,} book records into index '{index_name}'")
    
    return count['count']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load OpenLibrary data into Elasticsearch')
    parser.add_argument('--input', required=True, help='Input OpenLibrary dump file (.gz)')
    parser.add_argument('--index', default='openlibrary_books', help='Elasticsearch index name')
    parser.add_argument('--max-records', type=int, help='Maximum number of records to load (for testing)')
    
    args = parser.parse_args()
    
    load_openlibrary_data_to_elasticsearch(args.input, args.index, args.max_records)
