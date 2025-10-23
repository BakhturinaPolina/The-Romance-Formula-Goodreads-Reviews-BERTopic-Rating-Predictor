#!/usr/bin/env python3
"""
Alternative Sources for Romance Book Datasets
Find romance/fiction books from other sources when Anna's Archive datasets are not available
"""

import requests
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class AlternativeSourceFinder:
    """
    Find romance books from alternative sources
    """
    
    def __init__(self):
        """Initialize the alternative source finder"""
        self.sources = {
            'openlibrary': self._search_openlibrary,
            'gutenberg': self._search_gutenberg,
            'internet_archive': self._search_internet_archive,
            'zlib': self._search_zlib
        }
    
    def find_romance_books(self, title: str, author: str, 
                          sources: List[str] = None) -> List[Dict]:
        """
        Search for romance books across multiple sources
        
        Args:
            title: Book title
            author: Author name
            sources: List of sources to search (default: all)
            
        Returns:
            List of found books with metadata
        """
        if sources is None:
            sources = list(self.sources.keys())
        
        results = []
        
        for source in sources:
            if source in self.sources:
                try:
                    source_results = self.sources[source](title, author)
                    for result in source_results:
                        result['source'] = source
                    results.extend(source_results)
                except Exception as e:
                    logger.error(f"Error searching {source}: {e}")
        
        return results
    
    def _search_openlibrary(self, title: str, author: str) -> List[Dict]:
        """
        Search Open Library for romance books
        
        Args:
            title: Book title
            author: Author name
            
        Returns:
            List of book metadata
        """
        results = []
        
        try:
            # Open Library search API
            search_url = "https://openlibrary.org/search.json"
            params = {
                'title': title,
                'author': author,
                'subject': 'romance',
                'language': 'eng',
                'limit': 10
            }
            
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for doc in data.get('docs', []):
                # Check if it's romance/fiction
                subjects = doc.get('subject', [])
                is_romance = any('romance' in subject.lower() for subject in subjects)
                is_fiction = any('fiction' in subject.lower() for subject in subjects)
                
                if is_romance or is_fiction:
                    book_info = {
                        'title': doc.get('title', ''),
                        'author': ', '.join(doc.get('author_name', [])),
                        'isbn': doc.get('isbn', [None])[0] if doc.get('isbn') else None,
                        'publish_year': doc.get('first_publish_year'),
                        'language': doc.get('language', [None])[0] if doc.get('language') else None,
                        'subjects': subjects,
                        'download_url': None,  # Open Library doesn't provide direct downloads
                        'md5_hash': None
                    }
                    results.append(book_info)
        
        except Exception as e:
            logger.error(f"Open Library search failed: {e}")
        
        return results
    
    def _search_gutenberg(self, title: str, author: str) -> List[Dict]:
        """
        Search Project Gutenberg for romance books
        
        Args:
            title: Book title
            author: Author name
            
        Returns:
            List of book metadata
        """
        results = []
        
        try:
            # Project Gutenberg search API
            search_url = "https://www.gutenberg.org/ebooks/search/"
            params = {
                'query': f"{title} {author}",
                'submit_search': 'Search'
            }
            
            # Note: This would need proper HTML parsing
            # For now, return empty list
            logger.info("Project Gutenberg search not implemented (requires HTML parsing)")
        
        except Exception as e:
            logger.error(f"Project Gutenberg search failed: {e}")
        
        return results
    
    def _search_internet_archive(self, title: str, author: str) -> List[Dict]:
        """
        Search Internet Archive for romance books
        
        Args:
            title: Book title
            author: Author name
            
        Returns:
            List of book metadata
        """
        results = []
        
        try:
            # Internet Archive search API
            search_url = "https://archive.org/advancedsearch.php"
            params = {
                'q': f"title:({title}) AND creator:({author}) AND mediatype:texts AND language:eng",
                'fl': 'identifier,title,creator,date,language,subject',
                'rows': 10,
                'output': 'json'
            }
            
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for doc in data.get('response', {}).get('docs', []):
                # Check if it's romance/fiction
                subjects = doc.get('subject', [])
                is_romance = any('romance' in subject.lower() for subject in subjects)
                is_fiction = any('fiction' in subject.lower() for subject in subjects)
                
                if is_romance or is_fiction:
                    book_info = {
                        'title': doc.get('title', ''),
                        'author': ', '.join(doc.get('creator', [])) if isinstance(doc.get('creator'), list) else doc.get('creator', ''),
                        'identifier': doc.get('identifier', ''),
                        'publish_year': doc.get('date'),
                        'language': doc.get('language', ''),
                        'subjects': subjects,
                        'download_url': f"https://archive.org/details/{doc.get('identifier', '')}",
                        'md5_hash': None
                    }
                    results.append(book_info)
        
        except Exception as e:
            logger.error(f"Internet Archive search failed: {e}")
        
        return results
    
    def _search_zlib(self, title: str, author: str) -> List[Dict]:
        """
        Search Z-Library for romance books (if accessible)
        
        Args:
            title: Book title
            author: Author name
            
        Returns:
            List of book metadata
        """
        results = []
        
        try:
            # Note: Z-Library access varies by region and may require special handling
            logger.info("Z-Library search not implemented (access restrictions)")
        
        except Exception as e:
            logger.error(f"Z-Library search failed: {e}")
        
        return results
    
    def create_alternative_dataset(self, romance_csv: str, output_csv: str) -> None:
        """
        Create a dataset of romance books from alternative sources
        
        Args:
            romance_csv: Path to your romance books CSV
            output_csv: Path to save alternative dataset
        """
        logger.info(f"Creating alternative dataset from {romance_csv}")
        
        # Load your romance books
        df = pd.read_csv(romance_csv)
        
        alternative_books = []
        
        for idx, book in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing book {idx}/{len(df)}")
            
            # Search for the book in alternative sources
            results = self.find_romance_books(
                book['title'], 
                book['author_name'],
                sources=['openlibrary', 'internet_archive']
            )
            
            if results:
                # Add the best match
                best_match = results[0]
                best_match['original_work_id'] = book['work_id']
                best_match['original_title'] = book['title']
                best_match['original_author'] = book['author_name']
                alternative_books.append(best_match)
        
        # Save results
        if alternative_books:
            alt_df = pd.DataFrame(alternative_books)
            alt_df.to_csv(output_csv, index=False)
            logger.info(f"Saved {len(alternative_books)} alternative books to {output_csv}")
        else:
            logger.warning("No alternative books found")


def main():
    """
    Main function for alternative source finding
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Alternative Romance Book Sources')
    parser.add_argument('--romance-csv', required=True,
                       help='Path to romance books CSV file')
    parser.add_argument('--output-csv', default='alternative_romance_books.csv',
                       help='Output CSV file for alternative books')
    parser.add_argument('--test-search', action='store_true',
                       help='Test search with a single book')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize finder
    finder = AlternativeSourceFinder()
    
    if args.test_search:
        # Test with a single book
        df = pd.read_csv(args.romance_csv)
        test_book = df.iloc[0]
        
        print(f"Testing search for: {test_book['title']} by {test_book['author_name']}")
        results = finder.find_romance_books(test_book['title'], test_book['author_name'])
        
        print(f"Found {len(results)} results:")
        for result in results:
            print(f"  - {result['title']} by {result['author']} ({result['source']})")
    else:
        # Create full alternative dataset
        finder.create_alternative_dataset(args.romance_csv, args.output_csv)


if __name__ == "__main__":
    main()
