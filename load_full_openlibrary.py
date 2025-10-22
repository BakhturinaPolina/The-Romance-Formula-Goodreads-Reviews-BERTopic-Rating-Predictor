#!/usr/bin/env python3
"""
Load the complete OpenLibrary dataset (all records, no limit)
This will process the full 15.8GB dataset with ~11.8 million records
"""

import json
import sys
import gzip
from elasticsearch import Elasticsearch
from tqdm import tqdm
import argparse
import time

def load_full_openlibrary_data(dump_file, index_name="openlibrary_books_full"):
    """Load the complete OpenLibrary dataset into Elasticsearch"""
    
    # Connect to Elasticsearch
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    
    # Check if index exists, create if not
    if not es.indices.exists(index=index_name):
        print(f"Creating index: {index_name}")
        es.indices.create(index=index_name)
    
    print("Loading complete OpenLibrary dataset (estimated 11.8 million records)...")
    print("This may take several hours. Progress will be shown below.")
    
    # Load data
    records_loaded = 0
    start_time = time.time()
    
    with gzip.open(dump_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading OpenLibrary data"):
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
                
                # Extract title for easier searching
                title = None
                
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
                    
                    # Use ol_key as document ID
                    es.index(index=index_name, id=ol_key, body=book_info)
                    records_loaded += 1
                    
                    # Refresh index every 10,000 records for better performance
                    if records_loaded % 10000 == 0:
                        es.indices.refresh(index=index_name)
                        elapsed = time.time() - start_time
                        rate = records_loaded / elapsed
                        print(f"Loaded {records_loaded:,} records ({rate:.0f} records/sec)")
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                if records_loaded % 1000 == 0:  # Only print errors occasionally
                    print(f"Error processing record: {e}")
                continue
    
    # Final refresh
    es.indices.refresh(index=index_name)
    
    # Get final count
    count = es.count(index=index_name)
    elapsed = time.time() - start_time
    
    print(f"\n🎉 SUCCESS!")
    print(f"Loaded {count['count']:,} book records into index '{index_name}'")
    print(f"Total time: {elapsed/3600:.1f} hours")
    print(f"Average rate: {count['count']/elapsed:.0f} records/second")
    
    return count['count']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load complete OpenLibrary dataset')
    parser.add_argument('--input', required=True, help='Input OpenLibrary dump file (.gz)')
    parser.add_argument('--index', default='openlibrary_books_full', help='Elasticsearch index name')
    
    args = parser.parse_args()
    
    load_full_openlibrary_data(args.input, args.index)
