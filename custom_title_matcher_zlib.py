#!/usr/bin/env python3
"""
Custom title matcher for Z-Library data structure
"""

import csv
import sys
import argparse
from elasticsearch import Elasticsearch

def search_zlib_books(es, title, author=None, year=None):
    """Search for books in Z-Library data"""
    
    # Build search query
    must_clauses = []
    
    # Title search
    if title:
        must_clauses.append({
            "match": {
                "metadata.title": {
                    "query": title,
                    "fuzziness": "AUTO"
                }
            }
        })
    
    # Author search
    if author:
        must_clauses.append({
            "match": {
                "metadata.author": {
                    "query": author,
                    "fuzziness": "AUTO"
                }
            }
        })
    
    # Year search
    if year:
        must_clauses.append({
            "term": {
                "metadata.year": str(year)
            }
        })
    
    query = {
        "bool": {
            "must": must_clauses
        }
    }
    
    # Execute search
    result = es.search(
        index="zlib_records",
        body={
            "query": query,
            "size": 10
        }
    )
    
    return result['hits']['hits']

def main():
    parser = argparse.ArgumentParser(description='Match book titles against Z-Library data')
    parser.add_argument('--input', required=True, help='Input CSV file with book data')
    parser.add_argument('--output', required=True, help='Output CSV file with matches')
    
    args = parser.parse_args()
    
    # Connect to Elasticsearch
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    
    # Read input CSV
    matches = []
    
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            title = row.get('title', '').strip()
            author = row.get('author_name', '').strip()
            year = row.get('publication_year', '').strip()
            
            if not title:
                continue
            
            print(f"Searching for: {title} by {author} ({year})")
            
            # Search for matches
            hits = search_zlib_books(es, title, author, year)
            
            if hits:
                for hit in hits:
                    source = hit['_source']
                    metadata = source['metadata']
                    
                    match = {
                        'input_title': title,
                        'input_author': author,
                        'input_year': year,
                        'matched_title': metadata.get('title', ''),
                        'matched_author': metadata.get('author', ''),
                        'matched_year': metadata.get('year', ''),
                        'md5_hash': metadata.get('md5_reported', ''),
                        'zlibrary_id': metadata.get('zlibrary_id', ''),
                        'publisher': metadata.get('publisher', ''),
                        'language': metadata.get('language', ''),
                        'extension': metadata.get('extension', ''),
                        'filesize': metadata.get('filesize_reported', ''),
                        'isbns': ', '.join(metadata.get('isbns', [])),
                        'score': hit['_score'],
                        'aacid': source.get('aacid', '')
                    }
                    matches.append(match)
                    print(f"  ✓ Found: {metadata.get('title', '')} by {metadata.get('author', '')} (Score: {hit['_score']:.2f})")
            else:
                print(f"  ✗ No matches found")
    
    # Write results
    if matches:
        fieldnames = ['input_title', 'input_author', 'input_year', 'matched_title', 
                     'matched_author', 'matched_year', 'md5_hash', 'zlibrary_id', 
                     'publisher', 'language', 'extension', 'filesize', 'isbns', 
                     'score', 'aacid']
        
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matches)
        
        print(f"\nFound {len(matches)} matches. Results saved to {args.output}")
    else:
        print("\nNo matches found.")

if __name__ == "__main__":
    main()
