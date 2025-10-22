#!/usr/bin/env python3
"""
Extract MD5 hashes from all data sources for direct downloading
"""

import csv
import json
from elasticsearch import Elasticsearch
from tqdm import tqdm

def extract_md5_hashes():
    """Extract MD5 hashes from all sources that have them"""
    
    # Connect to Elasticsearch
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    
    all_hashes = []
    
    print("🔍 Extracting MD5 hashes from all sources...")
    
    # 1. Z-Library (has MD5 hashes)
    print("\n📚 Extracting from Z-Library...")
    try:
        result = es.search(
            index='zlib_records',
            body={
                'query': {'match_all': {}},
                'size': 10000
            }
        )
        
        for hit in result['hits']['hits']:
            data = hit['_source']
            if 'metadata' in data and 'md5_reported' in data['metadata']:
                hash_info = {
                    'source': 'Z-Library',
                    'md5': data['metadata']['md5_reported'],
                    'title': data['metadata'].get('title', 'Unknown'),
                    'author': data['metadata'].get('author', 'Unknown'),
                    'year': data['metadata'].get('year', 'Unknown'),
                    'extension': data['metadata'].get('extension', 'Unknown'),
                    'filesize': data['metadata'].get('filesize_reported', 'Unknown'),
                    'zlibrary_id': data['metadata'].get('zlibrary_id', 'Unknown'),
                    'download_url': f"https://b-ok.cc/md5/{data['metadata']['md5_reported']}"
                }
                all_hashes.append(hash_info)
        
        print(f"  ✅ Found {len([h for h in all_hashes if h['source'] == 'Z-Library'])} Z-Library books with MD5 hashes")
        
    except Exception as e:
        print(f"  ❌ Error extracting Z-Library hashes: {e}")
    
    # 2. LibGen Fiction (has MD5 hashes)
    print("\n📚 Extracting from LibGen Fiction...")
    try:
        result = es.search(
            index='libgen_fiction',
            body={
                'query': {'match_all': {}},
                'size': 10000
            }
        )
        
        for hit in result['hits']['hits']:
            data = hit['_source']
            if 'md5' in data and data['md5']:
                hash_info = {
                    'source': 'LibGen Fiction',
                    'md5': data['md5'],
                    'title': data.get('title', 'Unknown'),
                    'author': data.get('author', 'Unknown'),
                    'year': data.get('year', 'Unknown'),
                    'extension': data.get('extension', 'Unknown'),
                    'filesize': data.get('filesize', 'Unknown'),
                    'libgen_id': data.get('id', 'Unknown'),
                    'download_url': f"https://libgen.li/ads.php?md5={data['md5']}"
                }
                all_hashes.append(hash_info)
        
        print(f"  ✅ Found {len([h for h in all_hashes if h['source'] == 'LibGen Fiction'])} LibGen Fiction books with MD5 hashes")
        
    except Exception as e:
        print(f"  ❌ Error extracting LibGen Fiction hashes: {e}")
    
    # 3. Check other sources for any MD5-like identifiers
    print("\n📚 Checking other sources for identifiers...")
    
    # OpenLibrary - check for any hash-like fields
    try:
        result = es.search(
            index='openlibrary_books',
            body={
                'query': {'match_all': {}},
                'size': 1000  # Sample
            }
        )
        
        openlibrary_count = 0
        for hit in result['hits']['hits']:
            data = hit['_source']
            # OpenLibrary doesn't typically have MD5 hashes, but has other identifiers
            if 'data' in data and 'key' in data['data']:
                openlibrary_count += 1
        
        print(f"  ℹ️  OpenLibrary has {openlibrary_count} books (no MD5 hashes, but has OpenLibrary keys)")
        
    except Exception as e:
        print(f"  ❌ Error checking OpenLibrary: {e}")
    
    return all_hashes

def save_hashes_to_csv(hashes, filename='book_md5_hashes.csv'):
    """Save MD5 hashes to CSV file"""
    
    if not hashes:
        print("❌ No MD5 hashes found to save")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['source', 'md5', 'title', 'author', 'year', 'extension', 'filesize', 'download_url', 'additional_info']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for hash_info in hashes:
            # Prepare additional info based on source
            additional_info = {}
            if hash_info['source'] == 'Z-Library' and 'zlibrary_id' in hash_info:
                additional_info['zlibrary_id'] = hash_info['zlibrary_id']
            elif hash_info['source'] == 'LibGen Fiction' and 'libgen_id' in hash_info:
                additional_info['libgen_id'] = hash_info['libgen_id']
            
            writer.writerow({
                'source': hash_info['source'],
                'md5': hash_info['md5'],
                'title': hash_info['title'],
                'author': hash_info['author'],
                'year': hash_info['year'],
                'extension': hash_info['extension'],
                'filesize': hash_info['filesize'],
                'download_url': hash_info['download_url'],
                'additional_info': json.dumps(additional_info)
            })
    
    print(f"✅ Saved {len(hashes)} MD5 hashes to {filename}")

def create_download_script(hashes, filename='download_books.sh'):
    """Create a shell script to download books using MD5 hashes"""
    
    if not hashes:
        print("❌ No MD5 hashes found to create download script")
        return
    
    with open(filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Download books using MD5 hashes\n")
        f.write("# Usage: ./download_books.sh\n\n")
        
        f.write("mkdir -p downloads\n")
        f.write("cd downloads\n\n")
        
        for hash_info in hashes:
            title_clean = "".join(c for c in hash_info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            author_clean = "".join(c for c in hash_info['author'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename_clean = f"{author_clean} - {title_clean}.{hash_info['extension']}"
            
            f.write(f"# {hash_info['title']} by {hash_info['author']} ({hash_info['year']})\n")
            f.write(f"echo \"Downloading: {hash_info['title']} by {hash_info['author']}\"\n")
            
            if hash_info['source'] == 'Z-Library':
                f.write(f"curl -L -o \"{filename_clean}\" \"{hash_info['download_url']}\"\n")
            elif hash_info['source'] == 'LibGen Fiction':
                f.write(f"curl -L -o \"{filename_clean}\" \"{hash_info['download_url']}\"\n")
            
            f.write("\n")
    
    print(f"✅ Created download script: {filename}")
    print("   To use: chmod +x download_books.sh && ./download_books.sh")

def main():
    print("🎯 MD5 Hash Extractor for Book Downloads")
    print("=" * 50)
    
    # Extract hashes
    hashes = extract_md5_hashes()
    
    if not hashes:
        print("\n❌ No MD5 hashes found in any source")
        return
    
    print(f"\n🎉 Found {len(hashes)} books with MD5 hashes!")
    
    # Show summary by source
    sources = {}
    for hash_info in hashes:
        source = hash_info['source']
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
    
    print("\n📊 Summary by source:")
    for source, count in sources.items():
        print(f"  {source}: {count} books")
    
    # Save to CSV
    save_hashes_to_csv(hashes)
    
    # Create download script
    create_download_script(hashes)
    
    print(f"\n🚀 Ready to download {len(hashes)} books!")
    print("   Files created:")
    print("   - book_md5_hashes.csv (detailed list)")
    print("   - download_books.sh (download script)")

if __name__ == "__main__":
    main()
