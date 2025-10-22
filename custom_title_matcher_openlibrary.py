#!/usr/bin/env python3
"""
Custom title matcher for OpenLibrary data structure
"""

import csv
import sys
import argparse
from elasticsearch import Elasticsearch

def search_openlibrary_books(es, title, author=None, year=None):
    """Search for books in OpenLibrary data"""
    
    # Build search query
    must_clauses = []
    
    # Title search
    if title:
        must_clauses.append({
            "match": {
                "title": {
                    "query": title,
                    "fuzziness": "AUTO"
                }
            }
        })
    
    # Year search (in publish_date field)
    if year:
        must_clauses.append({
            "match": {
                "data.publish_date": str(year)
            }
        })
    
    query = {
        "bool": {
            "must": must_clauses
        }
    }
    
    # Execute search
    result = es.search(
        index="openlibrary_books",
        body={
            "query": query,
            "size": 10
        }
    )
    
    return result['hits']['hits']

def main():
    parser = argparse.ArgumentParser(description='Match book titles against OpenLibrary data')
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
            hits = search_openlibrary_books(es, title, author, year)
            
            if hits:
                for hit in hits:
                    source = hit['_source']
                    data = source.get('data', {})
                    
                    match = {
                        'input_title': title,
                        'input_author': author,
                        'input_year': year,
                        'matched_title': source.get('title', ''),
                        'matched_type': source.get('type', ''),
                        'matched_ol_key': source.get('ol_key', ''),
                        'publish_date': data.get('publish_date', ''),
                        'publishers': ', '.join(data.get('publishers', [])),
                        'isbn_10': ', '.join(data.get('isbn_10', [])),
                        'isbn_13': ', '.join(data.get('isbn_13', [])),
                        'physical_format': data.get('physical_format', ''),
                        'number_of_pages': data.get('number_of_pages', ''),
                        'subjects': ', '.join(data.get('subjects', [])),
                        'score': hit['_score']
                    }
                    matches.append(match)
                    print(f"  ✓ Found: {source.get('title', '')} (Score: {hit['_score']:.2f})")
            else:
                print(f"  ✗ No matches found")
    
    # Write results
    if matches:
        fieldnames = ['input_title', 'input_author', 'input_year', 'matched_title', 
                     'matched_type', 'matched_ol_key', 'publish_date', 'publishers',
                     'isbn_10', 'isbn_13', 'physical_format', 'number_of_pages', 
                     'subjects', 'score']
        
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matches)
        
        print(f"\nFound {len(matches)} matches. Results saved to {args.output}")
    else:
        print("\nNo matches found.")

if __name__ == "__main__":
    main()
