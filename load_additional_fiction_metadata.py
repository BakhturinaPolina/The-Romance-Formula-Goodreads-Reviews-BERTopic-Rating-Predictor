#!/usr/bin/env python3
"""
Load additional fiction metadata datasets from Anna's Archive
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

def load_additional_fiction_metadata():
    """Load additional fiction metadata datasets"""
    
    print("🎯 Loading Additional Fiction Metadata Datasets")
    print("=" * 60)
    
    # Define additional datasets to load
    datasets = [
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__magzdb_records__20240906T130340Z--20240906T130340Z.jsonl',
            'index': 'magzdb_publications',
            'name': 'MagzDB Publications',
            'description': 'Russian science fiction magazines and publications (includes fiction)'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__rgb_records__20240919T161201Z--20240919T161201Z.jsonl',
            'index': 'rgb_library_books',
            'name': 'RGB Library Books',
            'description': 'Russian State Library records (academic and rare books)'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__cerlalc_records__20240918T044206Z--20240918T044206Z.jsonl',
            'index': 'cerlalc_books',
            'name': 'CERLALC Books',
            'description': 'Latin American book publishing records (Spanish books)'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__nexusstc_records__20240130T000000Z--20240305T000000Z.jsonl',
            'index': 'nexusstc_books',
            'name': 'NexusSTC Books',
            'description': 'NexusSTC book records'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__airitibooks_records__20241228T170208Z--20241228T170208Z.jsonl',
            'index': 'airitibooks_books',
            'name': 'AiritiBooks',
            'description': 'AiritiBooks digital library records'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__kulturpass_records__20241229T210957Z--20241229T210957Z.jsonl',
            'index': 'kulturpass_books',
            'name': 'Kulturpass Books',
            'description': 'Kulturpass cultural books'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__ebscohost_records__20240823T161729Z--Wk44RExtNXgJ3346eBgRk9.jsonl',
            'index': 'ebscohost_books',
            'name': 'EBSCOhost Books',
            'description': 'EBSCOhost academic and library books'
        },
        {
            'file': 'annas-archive-outer/annas-archive/aacid_small/annas_archive_meta__aacid__hentai_records__20241229T110308Z--20241229T110308Z.jsonl',
            'index': 'hentai_books',
            'name': 'Hentai Books',
            'description': 'Hentai manga and graphic novels'
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
    
    print(f"\n🎉 Additional Fiction Metadata Loading Complete!")
    print(f"   Successfully loaded {successful_datasets} datasets")
    print(f"   Total records loaded: {total_loaded:,}")
    
    return total_loaded

def update_unified_matcher_additional():
    """Update the unified matcher to include additional fiction sources"""
    
    print(f"\n🔄 Updating Unified Matcher with Additional Sources...")
    
    # Read current unified matcher
    with open('unified_title_matcher.py', 'r') as f:
        content = f.read()
    
    # Add new sources to the sources dictionary
    new_sources = '''
                'magzdb_publications': {
                    'index': 'magzdb_publications',
                    'name': 'MagzDB Publications',
                    'description': 'Russian science fiction magazines and publications',
                    'fields': {
                        'title': 'metadata.record.title',
                        'author': 'metadata.record.description',
                        'year': 'metadata.record.yearRange'
                    }
                },
                'rgb_library_books': {
                    'index': 'rgb_library_books',
                    'name': 'RGB Library Books',
                    'description': 'Russian State Library records (academic and rare books)',
                    'fields': {
                        'title': 'metadata.record.fields.245.subfields.a',
                        'author': 'metadata.record.fields.100.subfields.a',
                        'year': 'metadata.record.fields.260.subfields.c'
                    }
                },
                'cerlalc_books': {
                    'index': 'cerlalc_books',
                    'name': 'CERLALC Books',
                    'description': 'Latin American book publishing records (Spanish books)',
                    'fields': {
                        'title': 'metadata.record.titulos.titulo',
                        'author': 'metadata.record.titulos_autores_rows.colaboradores_rows.nombre',
                        'year': 'metadata.record.titulos.fecha_aparicion'
                    }
                },
                'nexusstc_books': {
                    'index': 'nexusstc_books',
                    'name': 'NexusSTC Books',
                    'description': 'NexusSTC book records',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.author',
                        'year': 'metadata.year'
                    }
                },
                'airitibooks_books': {
                    'index': 'airitibooks_books',
                    'name': 'AiritiBooks',
                    'description': 'AiritiBooks digital library records',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.author',
                        'year': 'metadata.year'
                    }
                },
                'kulturpass_books': {
                    'index': 'kulturpass_books',
                    'name': 'Kulturpass Books',
                    'description': 'Kulturpass cultural books',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.author',
                        'year': 'metadata.year'
                    }
                },
                'ebscohost_books': {
                    'index': 'ebscohost_books',
                    'name': 'EBSCOhost Books',
                    'description': 'EBSCOhost academic and library books',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.author',
                        'year': 'metadata.year'
                    }
                },
                'hentai_books': {
                    'index': 'hentai_books',
                    'name': 'Hentai Books',
                    'description': 'Hentai manga and graphic novels',
                    'fields': {
                        'title': 'metadata.title',
                        'author': 'metadata.author',
                        'year': 'metadata.year'
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
    
    print(f"✅ Updated unified matcher with {8} additional fiction sources")

def main():
    parser = argparse.ArgumentParser(description='Load additional fiction metadata datasets')
    parser.add_argument('--update-matcher', action='store_true', help='Update unified matcher with new sources')
    args = parser.parse_args()
    
    # Load additional fiction metadata
    total_loaded = load_additional_fiction_metadata()
    
    # Update unified matcher if requested
    if args.update_matcher:
        update_unified_matcher_additional()
    
    print(f"\n🚀 Additional Fiction Metadata Loading Complete!")
    print(f"   Total records loaded: {total_loaded:,}")
    print(f"   Ready for even more comprehensive fiction title matching!")

if __name__ == "__main__":
    main()
