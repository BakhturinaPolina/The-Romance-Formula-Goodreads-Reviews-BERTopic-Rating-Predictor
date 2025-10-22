#!/usr/bin/env python3
"""
Robust Project Gutenberg catalog loader
Uses Project Gutenberg's official catalog data and API
"""

import requests
import json
import time
from elasticsearch import Elasticsearch
from tqdm import tqdm
import argparse
import re

def get_gutenberg_books_from_api():
    """Get Project Gutenberg books using their search API"""
    print("📥 Fetching Project Gutenberg books via API...")
    
    books = []
    
    # Project Gutenberg search API endpoints
    search_terms = [
        "romance",
        "fiction", 
        "novel",
        "love",
        "adventure",
        "mystery",
        "fantasy"
    ]
    
    for term in search_terms:
        try:
            print(f"Searching for: {term}")
            
            # Use Project Gutenberg's search API
            url = f"https://www.gutenberg.org/ebooks/search/?query={term}&submit_search=Go%21"
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                content = response.text
                
                # Extract book information using more robust regex
                # Look for book links in the format /ebooks/12345
                book_pattern = r'/ebooks/(\d+)[^>]*>([^<]+)</a>'
                matches = re.findall(book_pattern, content)
                
                for book_id, title in matches:
                    # Clean up the title
                    title = re.sub(r'\s+', ' ', title.strip())
                    if len(title) > 3 and not title.startswith('['):
                        books.append({
                            'gutenberg_id': book_id,
                            'title': title,
                            'source_url': f"https://www.gutenberg.org/ebooks/{book_id}",
                            'source': 'Project Gutenberg',
                            'search_term': term
                        })
                
                print(f"Found {len(matches)} books for '{term}'")
                time.sleep(2)  # Be respectful to the server
                
        except Exception as e:
            print(f"Error searching for '{term}': {e}")
            continue
    
    # Remove duplicates based on Gutenberg ID
    unique_books = {}
    for book in books:
        unique_books[book['gutenberg_id']] = book
    
    books = list(unique_books.values())
    print(f"Total unique Project Gutenberg books found: {len(books)}")
    
    return books

def create_sample_gutenberg_books():
    """Create a sample of well-known Project Gutenberg books for testing"""
    print("📚 Creating sample of well-known Project Gutenberg books...")
    
    # Well-known public domain books from Project Gutenberg
    sample_books = [
        {"gutenberg_id": "1342", "title": "Pride and Prejudice", "author": "Jane Austen"},
        {"gutenberg_id": "11", "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll"},
        {"gutenberg_id": "46", "title": "A Christmas Carol", "author": "Charles Dickens"},
        {"gutenberg_id": "74", "title": "The Adventures of Tom Sawyer", "author": "Mark Twain"},
        {"gutenberg_id": "76", "title": "Adventures of Huckleberry Finn", "author": "Mark Twain"},
        {"gutenberg_id": "84", "title": "Frankenstein", "author": "Mary Wollstonecraft Shelley"},
        {"gutenberg_id": "1260", "title": "Jane Eyre", "author": "Charlotte Brontë"},
        {"gutenberg_id": "514", "title": "Little Women", "author": "Louisa May Alcott"},
        {"gutenberg_id": "174", "title": "The Picture of Dorian Gray", "author": "Oscar Wilde"},
        {"gutenberg_id": "345", "title": "Dracula", "author": "Bram Stoker"},
        {"gutenberg_id": "1661", "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle"},
        {"gutenberg_id": "768", "title": "Wuthering Heights", "author": "Emily Brontë"},
        {"gutenberg_id": "2701", "title": "Moby Dick", "author": "Herman Melville"},
        {"gutenberg_id": "1322", "title": "Leaves of Grass", "author": "Walt Whitman"},
        {"gutenberg_id": "28054", "title": "The Great Gatsby", "author": "F. Scott Fitzgerald"}
    ]
    
    books = []
    for book in sample_books:
        books.append({
            'gutenberg_id': book['gutenberg_id'],
            'title': book['title'],
            'author': book['author'],
            'source_url': f"https://www.gutenberg.org/ebooks/{book['gutenberg_id']}",
            'source': 'Project Gutenberg',
            'public_domain': True,
            'download_available': True,
            'formats': ['HTML', 'EPUB', 'Kindle', 'Plain text']
        })
    
    print(f"Created {len(books)} sample Project Gutenberg books")
    return books

def load_gutenberg_to_elasticsearch(books, index_name="gutenberg_books"):
    """Load Project Gutenberg books into Elasticsearch"""
    
    # Connect to Elasticsearch
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    
    # Check if index exists, create if not
    if not es.indices.exists(index=index_name):
        print(f"Creating index: {index_name}")
        es.indices.create(index=index_name)
    
    print(f"Loading {len(books)} Project Gutenberg books into Elasticsearch...")
    
    # Load data
    for book in tqdm(books, desc="Loading Gutenberg books"):
        try:
            # Create document
            doc = {
                'gutenberg_id': book['gutenberg_id'],
                'title': book['title'],
                'source_url': book['source_url'],
                'source': book['source'],
                'public_domain': book.get('public_domain', True),
                'download_available': book.get('download_available', True),
                'formats': book.get('formats', ['HTML', 'EPUB', 'Kindle', 'Plain text'])
            }
            
            # Add author if available
            if 'author' in book:
                doc['author'] = book['author']
            
            # Use Gutenberg ID as document ID
            es.index(index=index_name, id=book['gutenberg_id'], body=doc)
            
        except Exception as e:
            print(f"Error indexing book {book['gutenberg_id']}: {e}")
            continue
    
    # Refresh index
    es.indices.refresh(index=index_name)
    
    # Get final count
    count = es.count(index=index_name)
    print(f"Successfully loaded {count['count']} Project Gutenberg books into index '{index_name}'")
    
    return count['count']

def create_gutenberg_matcher():
    """Create a title matcher for Project Gutenberg books"""
    matcher_code = '''#!/usr/bin/env python3
"""
Custom title matcher for Project Gutenberg data
"""

import csv
import sys
import argparse
from elasticsearch import Elasticsearch

def search_gutenberg_books(es, title, author=None, year=None):
    """Search for books in Project Gutenberg data"""
    
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
    
    # Author search if available
    if author:
        must_clauses.append({
            "match": {
                "author": {
                    "query": author,
                    "fuzziness": "AUTO"
                }
            }
        })
    
    query = {
        "bool": {
            "must": must_clauses
        }
    }
    
    # Execute search
    result = es.search(
        index="gutenberg_books",
        body={
            "query": query,
            "size": 10
        }
    )
    
    return result['hits']['hits']

def main():
    parser = argparse.ArgumentParser(description='Match book titles against Project Gutenberg data')
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
            hits = search_gutenberg_books(es, title, author, year)
            
            if hits:
                for hit in hits:
                    source = hit['_source']
                    
                    match = {
                        'input_title': title,
                        'input_author': author,
                        'input_year': year,
                        'matched_title': source.get('title', ''),
                        'matched_author': source.get('author', ''),
                        'gutenberg_id': source.get('gutenberg_id', ''),
                        'source_url': source.get('source_url', ''),
                        'public_domain': source.get('public_domain', ''),
                        'download_available': source.get('download_available', ''),
                        'formats': ', '.join(source.get('formats', [])),
                        'score': hit['_score']
                    }
                    matches.append(match)
                    print(f"  ✓ Found: {source.get('title', '')} by {source.get('author', 'N/A')} (Score: {hit['_score']:.2f})")
            else:
                print(f"  ✗ No matches found")
    
    # Write results
    if matches:
        fieldnames = ['input_title', 'input_author', 'input_year', 'matched_title', 
                     'matched_author', 'gutenberg_id', 'source_url', 'public_domain', 
                     'download_available', 'formats', 'score']
        
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matches)
        
        print(f"\\nFound {len(matches)} matches. Results saved to {args.output}")
    else:
        print("\\nNo matches found.")

if __name__ == "__main__":
    main()
'''
    
    with open('custom_title_matcher_gutenberg.py', 'w') as f:
        f.write(matcher_code)
    
    print("Created custom_title_matcher_gutenberg.py")

def main():
    parser = argparse.ArgumentParser(description='Load Project Gutenberg catalog into Elasticsearch')
    parser.add_argument('--index', default='gutenberg_books', help='Elasticsearch index name')
    parser.add_argument('--sample-only', action='store_true', help='Load only sample books for testing')
    
    args = parser.parse_args()
    
    if args.sample_only:
        # Load sample books for testing
        books = create_sample_gutenberg_books()
    else:
        # Try to get books from API, fallback to sample
        books = get_gutenberg_books_from_api()
        if not books:
            print("API method failed, using sample books...")
            books = create_sample_gutenberg_books()
    
    if books:
        # Load into Elasticsearch
        count = load_gutenberg_to_elasticsearch(books, args.index)
        
        # Create matcher
        create_gutenberg_matcher()
        
        print(f"\\n🎉 SUCCESS!")
        print(f"Loaded {count} Project Gutenberg books")
        print(f"Created custom_title_matcher_gutenberg.py")
    else:
        print("No books found to load")

if __name__ == "__main__":
    main()
