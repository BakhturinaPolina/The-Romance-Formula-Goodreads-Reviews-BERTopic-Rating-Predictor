#!/usr/bin/env python3
"""
Unified title matcher that searches across all data sources:
- Z-Library (fiction books with MD5 hashes)
- OpenLibrary (comprehensive book metadata)
- Project Gutenberg (public domain books)
- Duxiu (Chinese academic books)
"""

import csv
import sys
import argparse
from elasticsearch import Elasticsearch
import json

class UnifiedTitleMatcher:
    def __init__(self):
        """Initialize the unified matcher with Elasticsearch connection"""
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
        self.sources = {
            'zlib': {
                'index': 'zlib_records',
                'name': 'Z-Library',
                'description': 'Fiction books with MD5 hashes for downloads',
                'fields': {
                    'title': 'metadata.title',
                    'author': 'metadata.author',
                    'year': 'metadata.year'
                }
            },
            'openlibrary': {
                'index': 'openlibrary_books',
                'name': 'OpenLibrary',
                'description': 'Comprehensive book metadata with ISBNs',
                'fields': {
                    'title': 'title',
                    'author': 'data.authors',
                    'year': 'data.publish_date'
                }
            },
            'gutenberg': {
                'index': 'gutenberg_books',
                'name': 'Project Gutenberg',
                'description': 'Public domain books with free downloads',
                'fields': {
                    'title': 'title',
                    'author': 'author',
                    'year': None
                }
            },
            'duxiu': {
                'index': 'aa_records',
                'name': 'Duxiu Academic',
                'description': 'Chinese academic books',
                'fields': {
                    'title': 'metadata.record.title',
                    'author': 'metadata.record.author',
                    'year': 'metadata.record.year'
                }
            }
        }
    
    def search_source(self, source_key, title, author=None, year=None):
        """Search a specific data source"""
        source = self.sources[source_key]
        
        # Build search query
        must_clauses = []
        
        # Title search
        if title:
            must_clauses.append({
                "match": {
                    source['fields']['title']: {
                        "query": title,
                        "fuzziness": "AUTO"
                    }
                }
            })
        
        # Author search
        if author and source['fields']['author']:
            must_clauses.append({
                "match": {
                    source['fields']['author']: {
                        "query": author,
                        "fuzziness": "AUTO"
                    }
                }
            })
        
        # Year search
        if year and source['fields']['year']:
            must_clauses.append({
                "match": {
                    source['fields']['year']: str(year)
                }
            })
        
        query = {
            "bool": {
                "must": must_clauses
            }
        }
        
        # Execute search
        try:
            result = self.es.search(
                index=source['index'],
                body={
                    "query": query,
                    "size": 5
                }
            )
            return result['hits']['hits']
        except Exception as e:
            print(f"Error searching {source['name']}: {e}")
            return []
    
    def search_all_sources(self, title, author=None, year=None):
        """Search across all data sources"""
        all_matches = []
        
        print(f"🔍 Searching for: '{title}' by {author} ({year})")
        print("=" * 60)
        
        for source_key, source in self.sources.items():
            print(f"\n📚 Searching {source['name']}...")
            hits = self.search_source(source_key, title, author, year)
            
            if hits:
                print(f"  ✅ Found {len(hits)} matches")
                for hit in hits:
                    match_data = self.extract_match_data(source_key, source, hit)
                    all_matches.append(match_data)
                    print(f"    - {match_data['matched_title']} (Score: {hit['_score']:.2f})")
            else:
                print(f"  ❌ No matches found")
        
        return all_matches
    
    def extract_match_data(self, source_key, source, hit):
        """Extract standardized match data from different sources"""
        source_data = hit['_source']
        
        # Base match data
        match = {
            'source': source['name'],
            'source_key': source_key,
            'score': hit['_score'],
            'matched_title': '',
            'matched_author': '',
            'matched_year': '',
            'download_info': '',
            'metadata': {}
        }
        
        # Extract data based on source
        if source_key == 'zlib':
            metadata = source_data.get('metadata', {})
            match.update({
                'matched_title': metadata.get('title', ''),
                'matched_author': metadata.get('author', ''),
                'matched_year': metadata.get('year', ''),
                'download_info': f"MD5: {metadata.get('md5_reported', 'N/A')}",
                'metadata': {
                    'publisher': metadata.get('publisher', ''),
                    'language': metadata.get('language', ''),
                    'extension': metadata.get('extension', ''),
                    'filesize': metadata.get('filesize_reported', ''),
                    'isbns': ', '.join(metadata.get('isbns', []))
                }
            })
        
        elif source_key == 'openlibrary':
            data = source_data.get('data', {})
            match.update({
                'matched_title': source_data.get('title', ''),
                'matched_author': ', '.join([a.get('author', {}).get('key', '') for a in data.get('authors', [])]),
                'matched_year': data.get('publish_date', ''),
                'download_info': f"OL Key: {source_data.get('ol_key', '')}",
                'metadata': {
                    'publishers': ', '.join(data.get('publishers', [])),
                    'isbn_10': ', '.join(data.get('isbn_10', [])),
                    'isbn_13': ', '.join(data.get('isbn_13', [])),
                    'physical_format': data.get('physical_format', ''),
                    'number_of_pages': data.get('number_of_pages', '')
                }
            })
        
        elif source_key == 'gutenberg':
            match.update({
                'matched_title': source_data.get('title', ''),
                'matched_author': source_data.get('author', ''),
                'matched_year': '',
                'download_info': f"Gutenberg ID: {source_data.get('gutenberg_id', '')}",
                'metadata': {
                    'source_url': source_data.get('source_url', ''),
                    'public_domain': source_data.get('public_domain', ''),
                    'download_available': source_data.get('download_available', ''),
                    'formats': ', '.join(source_data.get('formats', []))
                }
            })
        
        elif source_key == 'duxiu':
            record = source_data.get('metadata', {}).get('record', {})
            match.update({
                'matched_title': record.get('title', ''),
                'matched_author': record.get('author', ''),
                'matched_year': record.get('year', ''),
                'download_info': f"AACID: {source_data.get('aacid', '')}",
                'metadata': {
                    'publisher': record.get('publisher', ''),
                    'isbn': record.get('isbn', '')
                }
            })
        
        return match
    
    def get_system_stats(self):
        """Get statistics about all data sources"""
        stats = {}
        total_books = 0
        
        print("📊 SYSTEM STATISTICS")
        print("=" * 50)
        
        for source_key, source in self.sources.items():
            try:
                count = self.es.count(index=source['index'])
                stats[source_key] = {
                    'name': source['name'],
                    'count': count['count'],
                    'description': source['description']
                }
                total_books += count['count']
                print(f"{source['name']}: {count['count']:,} books")
            except Exception as e:
                print(f"{source['name']}: Error - {e}")
                stats[source_key] = {'name': source['name'], 'count': 0, 'error': str(e)}
        
        print(f"\nTOTAL BOOKS: {total_books:,}")
        return stats

def main():
    parser = argparse.ArgumentParser(description='Unified title matcher across all data sources')
    parser.add_argument('--input', help='Input CSV file with book data')
    parser.add_argument('--output', help='Output CSV file with matches')
    parser.add_argument('--stats', action='store_true', help='Show system statistics')
    
    args = parser.parse_args()
    
    # Initialize matcher
    matcher = UnifiedTitleMatcher()
    
    # Show stats if requested
    if args.stats:
        matcher.get_system_stats()
        return
    
    # Check required arguments for matching
    if not args.input or not args.output:
        print("Error: --input and --output are required for matching")
        print("Use --stats to show system statistics only")
        return
    
    # Read input CSV
    all_matches = []
    
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            title = row.get('title', '').strip()
            author = row.get('author_name', '').strip()
            year = row.get('publication_year', '').strip()
            
            if not title:
                continue
            
            # Search all sources
            matches = matcher.search_all_sources(title, author, year)
            
            # Add input data to matches
            for match in matches:
                match.update({
                    'input_title': title,
                    'input_author': author,
                    'input_year': year
                })
                all_matches.append(match)
            
            print("\n" + "=" * 60)
    
    # Write results
    if all_matches:
        fieldnames = [
            'input_title', 'input_author', 'input_year',
            'source', 'source_key', 'matched_title', 'matched_author', 
            'matched_year', 'download_info', 'score',
            'metadata'
        ]
        
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for match in all_matches:
                # Convert metadata dict to string for CSV
                match_copy = match.copy()
                match_copy['metadata'] = json.dumps(match['metadata'])
                writer.writerow(match_copy)
        
        print(f"\n🎉 SUCCESS!")
        print(f"Found {len(all_matches)} total matches across all sources")
        print(f"Results saved to {args.output}")
    else:
        print("\n❌ No matches found across any sources")

if __name__ == "__main__":
    main()
