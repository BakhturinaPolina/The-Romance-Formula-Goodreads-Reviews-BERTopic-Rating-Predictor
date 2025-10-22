#!/usr/bin/env python3
"""
Load LibGen.li data (excluding LibGen.rs fiction)
"""

import argparse
import json
import re
from elasticsearch import Elasticsearch
from tqdm import tqdm

def parse_libgenli_editions_sql(sql_file_path):
    """Parse LibGen.li editions SQL dump file and extract book records."""
    books = []
    
    print(f"📖 Parsing LibGen.li editions SQL file: {sql_file_path}")
    
    with open(sql_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find the INSERT statement
    insert_match = re.search(r"INSERT INTO `libgenli_editions` VALUES(.*?);", content, re.DOTALL)
    if not insert_match:
        print("❌ No INSERT statement found in SQL file")
        return books
    
    values_str = insert_match.group(1)
    
    # Split by '),(' to get individual records
    records = re.findall(r'\(([^)]*)\)', values_str)
    
    print(f"🔍 Found {len(records)} records to parse")
    
    for i, record_str in enumerate(records, 1):
        try:
            # Parse the record using a more robust method
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
            
            if len(parts) >= 30:  # LibGen.li editions has more fields
                # Clean up the parts
                cleaned_parts = []
                for part in parts:
                    if part.startswith('"') and part.endswith('"'):
                        part = part[1:-1]  # Remove quotes
                    # Handle escaped characters
                    part = part.replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
                    cleaned_parts.append(part)
                
                # Map LibGen.li editions fields to our book structure
                book = {
                    "id": int(cleaned_parts[0]) if cleaned_parts[0].isdigit() else 0,
                    "md5": cleaned_parts[-1] if len(cleaned_parts) > 30 else "",  # MD5 is usually last
                    "title": cleaned_parts[1] if len(cleaned_parts) > 1 else "",
                    "author": cleaned_parts[2] if len(cleaned_parts) > 2 else "",
                    "series": cleaned_parts[3] if len(cleaned_parts) > 3 else "",
                    "edition": cleaned_parts[4] if len(cleaned_parts) > 4 else "",
                    "language": cleaned_parts[5] if len(cleaned_parts) > 5 else "",
                    "year": cleaned_parts[10] if len(cleaned_parts) > 10 else "",  # Year is usually around position 10
                    "publisher": cleaned_parts[6] if len(cleaned_parts) > 6 else "",
                    "pages": cleaned_parts[7] if len(cleaned_parts) > 7 else "",
                    "identifier": cleaned_parts[8] if len(cleaned_parts) > 8 else "",
                    "googlebook_id": cleaned_parts[9] if len(cleaned_parts) > 9 else "",
                    "asin": "",
                    "cover_url": "",
                    "extension": "",
                    "filesize": 0,
                    "library": "LibGen.li",
                    "issue": "",
                    "locator": "",
                    "commentary": "",
                    "generic": "",
                    "visible": "",
                    "time_added": cleaned_parts[28] if len(cleaned_parts) > 28 else "",
                    "time_last_modified": cleaned_parts[29] if len(cleaned_parts) > 29 else "",
                    "source": "libgenli_editions"
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
    
    print(f"✅ Successfully parsed {len(books)} LibGen.li edition books")
    return books

def parse_libgenli_files_sql(sql_file_path):
    """Parse LibGen.li files SQL dump file and extract file records."""
    files = []
    
    print(f"📁 Parsing LibGen.li files SQL file: {sql_file_path}")
    
    with open(sql_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find the INSERT statement
    insert_match = re.search(r"INSERT INTO `libgenli_files` VALUES(.*?);", content, re.DOTALL)
    if not insert_match:
        print("❌ No INSERT statement found in SQL file")
        return files
    
    values_str = insert_match.group(1)
    
    # Split by '),(' to get individual records
    records = re.findall(r'\(([^)]*)\)', values_str)
    
    print(f"🔍 Found {len(records)} file records to parse")
    
    for i, record_str in enumerate(records, 1):
        try:
            # Parse the record
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
            
            if len(parts) >= 20:  # LibGen.li files has many fields
                # Clean up the parts
                cleaned_parts = []
                for part in parts:
                    if part.startswith('"') and part.endswith('"'):
                        part = part[1:-1]  # Remove quotes
                    # Handle escaped characters
                    part = part.replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
                    cleaned_parts.append(part)
                
                # Map LibGen.li files fields to our file structure
                file_record = {
                    "id": int(cleaned_parts[0]) if cleaned_parts[0].isdigit() else 0,
                    "md5": cleaned_parts[1] if len(cleaned_parts) > 1 else "",
                    "title": "",  # Files don't have titles directly
                    "author": "",
                    "series": "",
                    "edition": "",
                    "language": "",
                    "year": "",
                    "publisher": "",
                    "pages": "",
                    "identifier": "",
                    "googlebook_id": "",
                    "asin": "",
                    "cover_url": "",
                    "extension": cleaned_parts[21] if len(cleaned_parts) > 21 else "",  # Extension is usually around position 21
                    "filesize": int(cleaned_parts[20]) if len(cleaned_parts) > 20 and cleaned_parts[20].isdigit() else 0,
                    "library": "LibGen.li",
                    "issue": "",
                    "locator": cleaned_parts[22] if len(cleaned_parts) > 22 else "",
                    "commentary": "",
                    "generic": "",
                    "visible": "",
                    "time_added": cleaned_parts[4] if len(cleaned_parts) > 4 else "",
                    "time_last_modified": cleaned_parts[5] if len(cleaned_parts) > 5 else "",
                    "source": "libgenli_files"
                }
                files.append(file_record)
            else:
                print(f"⚠️  Skipping file record {i}: insufficient fields ({len(parts)})")
                
        except Exception as e:
            print(f"❌ Error parsing file record {i}: {e}")
            continue
    
    print(f"✅ Successfully parsed {len(files)} LibGen.li file records")
    return files

def load_libgenli_to_elasticsearch(editions_file, files_file, index_name="libgenli_books"):
    """Load LibGen.li data from SQL dump files into Elasticsearch"""
    
    # Connect to Elasticsearch
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    
    # Create index if it doesn't exist
    if not es.indices.exists(index=index_name):
        print(f"📝 Creating index: {index_name}")
        es.indices.create(index=index_name)
    else:
        print(f"📝 Using existing index: {index_name}")
    
    all_books = []
    
    # Parse editions file
    if editions_file:
        editions = parse_libgenli_editions_sql(editions_file)
        all_books.extend(editions)
    
    # Parse files file
    if files_file:
        files = parse_libgenli_files_sql(files_file)
        all_books.extend(files)
    
    if not all_books:
        print("❌ No books found in SQL files")
        return
    
    print(f"📚 Loading {len(all_books)} LibGen.li records into Elasticsearch...")
    
    # Index books one by one
    successful = 0
    failed = 0
    
    for book in tqdm(all_books, desc="Loading LibGen.li records"):
        try:
            es.index(index=index_name, body=book)
            successful += 1
        except Exception as e:
            print(f"❌ Error indexing record {book.get('title', 'Unknown')}: {e}")
            failed += 1
    
    print(f"✅ Successfully loaded {successful} LibGen.li records into {index_name}")
    if failed > 0:
        print(f"❌ Failed to load {failed} records")
    
    # Get final count
    try:
        count_result = es.count(index=index_name)
        print(f"📊 Total records in {index_name}: {count_result['count']}")
    except Exception as e:
        print(f"⚠️  Could not get final count: {e}")
    
    # Show sample of loaded books
    if successful > 0:
        print(f"\n📖 Sample of loaded records:")
        for i, book in enumerate(all_books[:5], 1):
            title = book.get('title', 'No title')
            author = book.get('author', 'Unknown author')
            year = book.get('year', 'Unknown year')
            extension = book.get('extension', 'Unknown format')
            source = book.get('source', 'Unknown source')
            print(f"  {i}. \"{title}\" by {author} ({year}) - {extension} [{source}]")

def main():
    parser = argparse.ArgumentParser(description='Load LibGen.li data (excluding LibGen.rs fiction)')
    parser.add_argument('--editions', 
                       default='annas-archive-outer/annas-archive/test/data-dumps/mariadb/allthethings.libgenli_editions.00000.sql',
                       help='Path to the LibGen.li editions SQL dump file')
    parser.add_argument('--files', 
                       default='annas-archive-outer/annas-archive/test/data-dumps/mariadb/allthethings.libgenli_files.00000.sql',
                       help='Path to the LibGen.li files SQL dump file')
    parser.add_argument('--index', default='libgenli_books', help='Elasticsearch index name')
    args = parser.parse_args()
    
    print("🎯 LibGen.li Data Loader (Excluding LibGen.rs Fiction)")
    print("=" * 60)
    
    load_libgenli_to_elasticsearch(args.editions, args.files, args.index)

if __name__ == "__main__":
    main()
