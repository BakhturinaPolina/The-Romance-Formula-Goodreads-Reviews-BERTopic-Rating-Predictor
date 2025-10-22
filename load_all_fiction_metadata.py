#!/usr/bin/env python3
"""
Load all available fiction metadata datasets from Anna's Archive
"""

import argparse
import json
import os
from elasticsearch import Elasticsearch
from tqdm import tqdm

def load_jsonl_to_elasticsearch(jsonl_file, index_name, source_name):
    """Load JSONL file into Elasticsearch with progress tracking"""
    
    # Connect to Elasticsearch
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    
    # Create index if it doesn't exist
    if not es.indices.exists(index=index_name):
        print(f"📝 Creating index: {index_name}")
        es.indices.create(index=index_name)
    else:
        print(f"📝 Using existing index: {index_name}")
    
    # Count lines for progress tracking
    print(f"📊 Counting records in {jsonl_file}...")
    with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"📚 Loading {total_lines:,} records from {source_name}...")
    
    # Load records
    successful = 0
    failed = 0
    
    with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc=f"Loading {source_name}"), 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Add source information
                data['_source_name'] = source_name
                data['_file_source'] = os.path.basename(jsonl_file)
                
                # Index the document
                es.index(index=index_name, body=data)
                successful += 1
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error at line {line_num}: {e}")
                failed += 1
            except Exception as e:
                print(f"❌ Error indexing line {line_num}: {e}")
                failed += 1
    
    print(f"✅ Successfully loaded {successful:,} records from {source_name}")
    if failed > 0:
        print(f"❌ Failed to load {failed:,} records")
    
    # Get final count
    try:
        count_result = es.count(index=index_name)
        print(f"📊 Total records in {index_name}: {count_result['count']:,}")
    except Exception as e:
        print(f"⚠️  Could not get final count: {e}")
    
    return successful

def load_all_fiction_metadata():
    """Load all available fiction metadata datasets"""
    
    print("🎯 Loading All Fiction Metadata Datasets")
    print("=" * 60)
    
    # Define datasets to load (prioritizing fiction-relevant ones)
    datasets = [
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__goodreads_records__20240913T115838Z--20240913T115838Z.jsonl',
            'index': 'goodreads_books',
            'name': 'Goodreads Books',
            'description': 'Book metadata from Goodreads (excellent for fiction)'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__libby_records__20240911T184811Z--20240911T184811Z.jsonl',
            'index': 'libby_books',
            'name': 'Libby Library Books',
            'description': 'Library books from Libby (includes fiction)'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__upload_records__20240627T210538Z--20240627T230953Z.jsonl',
            'index': 'upload_books',
            'name': 'User Uploaded Books',
            'description': 'Books uploaded by users (includes fiction)'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__worldcat__20231001T025039Z--20231001T235839Z.jsonl',
            'index': 'worldcat_books',
            'name': 'WorldCat Library Catalog',
            'description': 'Global library catalog (comprehensive fiction coverage)'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__trantor_records__20240911T134314Z--20240911T134314Z.jsonl',
            'index': 'trantor_books',
            'name': 'Trantor Books',
            'description': 'Books from Trantor platform'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__gbooks_records__20240920T051416Z--20240920T051416Z.jsonl',
            'index': 'gbooks_books',
            'name': 'Google Books',
            'description': 'Google Books metadata'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__hathitrust_records__20060710T152310Z--20250301T052542Z.jsonl',
            'index': 'hathitrust_books',
            'name': 'HathiTrust Books',
            'description': 'HathiTrust digital library books'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__ia2_records__20240126T065114Z--20240126T070601Z.jsonl',
            'index': 'internet_archive_books',
            'name': 'Internet Archive Books',
            'description': 'Books from Internet Archive'
        }
    ]
    
    total_loaded = 0
    successful_datasets = 0
    
    for dataset in datasets:
        print(f"\n📖 Loading {dataset['name']}...")
        print(f"   Description: {dataset['description']}")
        
        if not os.path.exists(dataset['file']):
            print(f"❌ File not found: {dataset['file']}")
            continue
        
        try:
            loaded_count = load_jsonl_to_elasticsearch(
                dataset['file'], 
                dataset['index'], 
                dataset['name']
            )
            total_loaded += loaded_count
            successful_datasets += 1
            
        except Exception as e:
            print(f"❌ Error loading {dataset['name']}: {e}")
            continue
    
    print(f"\n🎉 Fiction Metadata Loading Complete!")
    print(f"   Successfully loaded {successful_datasets} datasets")
    print(f"   Total records loaded: {total_loaded:,}")
    
    return total_loaded

def update_unified_matcher():
    """Update the unified matcher to include all new fiction sources"""
    
    print(f"\n🔄 Updating Unified Matcher...")
    
    # Read current unified matcher
    with open('unified_title_matcher.py', 'r') as f:
        content = f.read()
    
    # Add new sources to the sources dictionary
    new_sources = '''
                'goodreads_books': {
                    'index': 'goodreads_books',
                    'name': 'Goodreads Books',
                    'description': 'Book metadata from Goodreads (excellent for fiction)',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.authors',
                        'year': 'metadata.publishedDate'
                    }
                },
                'libby_books': {
                    'index': 'libby_books',
                    'name': 'Libby Library Books',
                    'description': 'Library books from Libby (includes fiction)',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.authors',
                        'year': 'metadata.publishedDate'
                    }
                },
                'upload_books': {
                    'index': 'upload_books',
                    'name': 'User Uploaded Books',
                    'description': 'Books uploaded by users (includes fiction)',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.authors',
                        'year': 'metadata.publishedDate'
                    }
                },
                'worldcat_books': {
                    'index': 'worldcat_books',
                    'name': 'WorldCat Library Catalog',
                    'description': 'Global library catalog (comprehensive fiction coverage)',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.authors',
                        'year': 'metadata.publishedDate'
                    }
                },
                'trantor_books': {
                    'index': 'trantor_books',
                    'name': 'Trantor Books',
                    'description': 'Books from Trantor platform',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.authors',
                        'year': 'metadata.publishedDate'
                    }
                },
                'gbooks_books': {
                    'index': 'gbooks_books',
                    'name': 'Google Books',
                    'description': 'Google Books metadata',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.authors',
                        'year': 'metadata.publishedDate'
                    }
                },
                'hathitrust_books': {
                    'index': 'hathitrust_books',
                    'name': 'HathiTrust Books',
                    'description': 'HathiTrust digital library books',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.authors',
                        'year': 'metadata.publishedDate'
                    }
                },
                'internet_archive_books': {
                    'index': 'internet_archive_books',
                    'name': 'Internet Archive Books',
                    'description': 'Books from Internet Archive',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.authors',
                        'year': 'metadata.publishedDate'
                    }
                },'''
    
    # Insert new sources before the closing bracket
    if "'duxiu':" in content:
        content = content.replace("'duxiu':", new_sources + "\n            'duxiu':")
    else:
        # Fallback: add before the last source
        content = content.replace("'libgenli_books':", new_sources + "\n            'libgenli_books':")
    
    # Write updated content
    with open('unified_title_matcher.py', 'w') as f:
        f.write(content)
    
    print(f"✅ Updated unified matcher with {8} new fiction sources")

def main():
    parser = argparse.ArgumentParser(description='Load all fiction metadata datasets')
    parser.add_argument('--update-matcher', action='store_true', help='Update unified matcher with new sources')
    args = parser.parse_args()
    
    # Load all fiction metadata
    total_loaded = load_all_fiction_metadata()
    
    # Update unified matcher if requested
    if args.update_matcher:
        update_unified_matcher()
    
    print(f"\n🚀 Fiction Metadata Loading Complete!")
    print(f"   Total records loaded: {total_loaded:,}")
    print(f"   Ready for comprehensive fiction title matching!")

if __name__ == "__main__":
    main()
