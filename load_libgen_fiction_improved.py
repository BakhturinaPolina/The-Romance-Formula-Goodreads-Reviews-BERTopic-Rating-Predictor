#!/usr/bin/env python3
"""
Load LibGen Fiction dataset with improved SQL parsing
"""

import argparse
import json
import re
from elasticsearch import Elasticsearch
from tqdm import tqdm

def parse_libgen_fiction_sql_improved(sql_file_path):
    """Parse LibGen Fiction SQL dump file with improved parsing."""
    books = []
    
    print(f"📖 Parsing LibGen Fiction SQL file: {sql_file_path}")
    
    with open(sql_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find the INSERT statement
    insert_match = re.search(r"INSERT INTO `libgenrs_fiction` VALUES(.*?);", content, re.DOTALL)
    if not insert_match:
        print("❌ No INSERT statement found in SQL file")
        return books
    
    values_str = insert_match.group(1)
    
    # Split by '),(' to get individual records
    # The format is: (1,"hash","title","author",...) ,(2,"hash","title","author",...)
    records = re.findall(r'\(([^)]*)\)', values_str)
    
    print(f"🔍 Found {len(records)} records to parse")
    
    for i, record_str in enumerate(records, 1):
        try:
            # Parse the record using a more robust method
            # Split by comma, but handle quoted strings properly
            parts = []
            current_part = ""
            in_quotes = False
            quote_char = None
            escape_next = False
            
            for j, char in enumerate(record_str):
                if escape_next:
                    current_part += char
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    current_part += char
                    continue
                    
                if char in ['"', "'"] and not in_quotes:
                    in_quotes = True
                    quote_char = char
                    current_part += char
                elif char == quote_char and in_quotes:
                    in_quotes = False
                    quote_char = None
                    current_part += char
                elif char == ',' and not in_quotes:
                    parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
            
            # Add the last part
            if current_part:
                parts.append(current_part.strip())
            
            if len(parts) >= 24:  # Ensure we have enough fields
                # Clean up the parts
                cleaned_parts = []
                for part in parts:
                    if part.startswith('"') and part.endswith('"'):
                        part = part[1:-1]  # Remove quotes
                    # Handle escaped characters
                    part = part.replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
                    cleaned_parts.append(part)
                
                book = {
                    "id": int(cleaned_parts[0]) if cleaned_parts[0].isdigit() else 0,
                    "md5": cleaned_parts[1],
                    "title": cleaned_parts[2],
                    "author": cleaned_parts[3],
                    "series": cleaned_parts[4],
                    "edition": cleaned_parts[5],
                    "language": cleaned_parts[6],
                    "year": cleaned_parts[7],
                    "publisher": cleaned_parts[8],
                    "pages": cleaned_parts[9],
                    "identifier": cleaned_parts[10],
                    "googlebook_id": cleaned_parts[11],
                    "asin": cleaned_parts[12],
                    "cover_url": cleaned_parts[13],
                    "extension": cleaned_parts[14],
                    "filesize": int(cleaned_parts[15]) if cleaned_parts[15].isdigit() else 0,
                    "library": cleaned_parts[16],
                    "issue": cleaned_parts[17],
                    "locator": cleaned_parts[18],
                    "commentary": cleaned_parts[19],
                    "generic": cleaned_parts[20],
                    "visible": cleaned_parts[21],
                    "time_added": cleaned_parts[22],
                    "time_last_modified": cleaned_parts[23]
                }
                books.append(book)
            else:
                print(f"⚠️  Skipping record {i}: insufficient fields ({len(parts)})")
                if i <= 5:  # Show first few problematic records
                    print(f"    Parts: {parts[:10]}...")  # Show first 10 parts
                
        except Exception as e:
            print(f"❌ Error parsing record {i}: {e}")
            if i <= 5:  # Show first few errors
                print(f"    Record: {record_str[:100]}...")
            continue
    
    print(f"✅ Successfully parsed {len(books)} LibGen Fiction books")
    return books

def load_libgen_fiction_to_elasticsearch(sql_file, index_name="libgen_fiction_full"):
    """Load LibGen Fiction data from SQL dump file into Elasticsearch"""
    
    # Connect to Elasticsearch
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    
    # Create index if it doesn't exist
    if not es.indices.exists(index=index_name):
        print(f"📝 Creating index: {index_name}")
        es.indices.create(index=index_name)
    else:
        print(f"📝 Using existing index: {index_name}")
    
    # Parse the SQL file
    books = parse_libgen_fiction_sql_improved(sql_file)
    
    if not books:
        print("❌ No books found in SQL file")
        return
    
    print(f"📚 Loading {len(books)} LibGen Fiction books into Elasticsearch...")
    
    # Index books one by one
    successful = 0
    failed = 0
    
    for book in tqdm(books, desc="Loading LibGen Fiction books"):
        try:
            es.index(index=index_name, body=book)
            successful += 1
        except Exception as e:
            print(f"❌ Error indexing book {book.get('title', 'Unknown')}: {e}")
            failed += 1
    
    print(f"✅ Successfully loaded {successful} LibGen Fiction books into {index_name}")
    if failed > 0:
        print(f"❌ Failed to load {failed} books")
    
    # Get final count
    try:
        count_result = es.count(index=index_name)
        print(f"📊 Total books in {index_name}: {count_result['count']}")
    except Exception as e:
        print(f"⚠️  Could not get final count: {e}")
    
    # Show sample of loaded books
    if successful > 0:
        print(f"\n📖 Sample of loaded books:")
        for i, book in enumerate(books[:5], 1):
            print(f"  {i}. \"{book['title']}\" by {book['author']} ({book['year']}) - {book['extension']}")

def main():
    parser = argparse.ArgumentParser(description='Load LibGen Fiction data with improved parsing')
    parser.add_argument('--input', 
                       default='annas-archive-outer/annas-archive/test/data-dumps/mariadb/allthethings.libgenrs_fiction.00000.sql',
                       help='Path to the LibGen Fiction SQL dump file')
    parser.add_argument('--index', default='libgen_fiction_full', help='Elasticsearch index name')
    args = parser.parse_args()
    
    print("🎯 LibGen Fiction Improved Loader")
    print("=" * 50)
    
    load_libgen_fiction_to_elasticsearch(args.input, args.index)

if __name__ == "__main__":
    main()
