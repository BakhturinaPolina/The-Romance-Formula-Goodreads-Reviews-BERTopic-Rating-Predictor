#!/usr/bin/env python3
"""
Simple script to load Z-Library data from JSONL file into Elasticsearch
"""

import json
import sys
from elasticsearch import Elasticsearch
from tqdm import tqdm

def load_zlib_data_to_elasticsearch(jsonl_file, index_name="zlib_records"):
    """Load Z-Library data from JSONL file into Elasticsearch"""
    
    # Connect to Elasticsearch
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    
    # Check if index exists, create if not
    if not es.indices.exists(index=index_name):
        print(f"Creating index: {index_name}")
        es.indices.create(index=index_name)
    
    # Count total lines for progress bar
    print("Counting lines in file...")
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Loading {total_lines} records into Elasticsearch...")
    
    # Load data
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=total_lines)):
            try:
                data = json.loads(line.strip())
                
                # Extract the document ID from aacid
                doc_id = data.get('aacid', f'doc_{i}')
                
                # Index the document
                es.index(index=index_name, id=doc_id, body=data)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i+1}: {e}")
                continue
            except Exception as e:
                print(f"Error indexing document {i+1}: {e}")
                continue
    
    # Refresh index to make data searchable
    es.indices.refresh(index=index_name)
    
    # Get final count
    count = es.count(index=index_name)
    print(f"Successfully loaded {count['count']} documents into index '{index_name}'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 load_zlib_to_elasticsearch.py <jsonl_file>")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    load_zlib_data_to_elasticsearch(jsonl_file)
