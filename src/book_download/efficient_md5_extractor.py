#!/usr/bin/env python3
"""
Efficient MD5 Hash Extractor for Anna's Archive
Uses the working API-based search from existing codebase
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import time

# Add the book_download directory to path
sys.path.insert(0, str(Path(__file__).parent))

from anna_api_client import AnnaAPIClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EfficientMD5Extractor:
    """Efficient MD5 extractor using the working API client"""
    
    def __init__(self):
        """Initialize the extractor with the working API client"""
        self.client = AnnaAPIClient()
        
        # Statistics
        self.stats = {
            'books_searched': 0,
            'books_found': 0,
            'md5s_found': 0,
            'errors': 0
        }
        
        logger.info("Efficient MD5 Extractor initialized with working API client")
    
    def search_book_md5(self, title: str, author: str) -> Optional[Dict]:
        """
        Search for a book using the working API client
        
        Args:
            title: Book title
            author: Author name
            
        Returns:
            Dictionary with book info and MD5 hash or None if not found
        """
        self.stats['books_searched'] += 1
        
        try:
            logger.info(f"Searching: {title} by {author}")
            
            # Use the working search_md5 method from the API client
            md5_hash = self.client.search_md5(title, author, prefer_exts=("epub", "mobi", "pdf"))
            
            if md5_hash:
                self.stats['books_found'] += 1
                self.stats['md5s_found'] += 1
                logger.info(f"Found MD5: {md5_hash} for {title} by {author}")
                
                return {
                    'original_title': title,
                    'original_author': author,
                    'md5_hash': md5_hash,
                    'found_title': title,  # API doesn't return found title, use original
                    'found_author': author,  # API doesn't return found author, use original
                    'file_formats': ['epub'],  # Assume epub for now
                    'match_quality': 1.0  # API search is considered high quality
                }
            else:
                logger.info(f"No MD5 found: {title} by {author}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching for '{title}' by '{author}': {e}")
            self.stats['errors'] += 1
            return None
    
    def extract_md5s_from_csv(self, csv_path: str, output_path: str, max_books: int = None) -> Dict:
        """
        Extract MD5 hashes for books in CSV file
        
        Args:
            csv_path: Path to input CSV file
            output_path: Path to output CSV file
            max_books: Maximum number of books to process (None for all)
            
        Returns:
            Dictionary with extraction statistics
        """
        logger.info(f"Loading books from {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        if max_books:
            df = df.head(max_books)
        
        logger.info(f"Processing {len(df)} books")
        
        # Prepare results
        results = []
        
        for idx, row in df.iterrows():
            title = str(row.get('title', '')).strip()
            author = str(row.get('author_name', '')).strip()
            
            if not title or not author or title == 'nan' or author == 'nan':
                logger.warning(f"Skipping row {idx}: missing title or author")
                continue
            
            logger.info(f"Processing book {idx + 1}/{len(df)}: {title} by {author}")
            
            # Search for MD5
            result = self.search_book_md5(title, author)
            
            if result:
                results.append({
                    'work_id': row.get('work_id', ''),
                    'title': title,
                    'author_name': author,
                    'publication_year': row.get('publication_year', ''),
                    'md5_hash': result['md5_hash'],
                    'found_title': result['found_title'],
                    'found_author': result['found_author'],
                    'file_formats': ','.join(result['file_formats']),
                    'match_quality': result['match_quality'],
                    'download_links': ''  # API doesn't provide direct links
                })
            else:
                results.append({
                    'work_id': row.get('work_id', ''),
                    'title': title,
                    'author_name': author,
                    'publication_year': row.get('publication_year', ''),
                    'md5_hash': '',
                    'found_title': '',
                    'found_author': '',
                    'file_formats': '',
                    'match_quality': 0.0,
                    'download_links': ''
                })
            
            # Small delay to be respectful
            time.sleep(1)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Found MD5 hashes for {len([r for r in results if r['md5_hash']])} out of {len(results)} books")
        
        # Print statistics
        self._print_statistics()
        
        return {
            'total_books': len(df),
            'books_with_md5': len([r for r in results if r['md5_hash']]),
            'success_rate': len([r for r in results if r['md5_hash']]) / len(df) if df else 0,
            'stats': self.stats
        }
    
    def _print_statistics(self):
        """Print extraction statistics"""
        logger.info("=" * 50)
        logger.info("EXTRACTION STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Books searched: {self.stats['books_searched']}")
        logger.info(f"Books found: {self.stats['books_found']}")
        logger.info(f"MD5s found: {self.stats['md5s_found']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info("=" * 50)


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Efficiently extract MD5 hashes from Anna's Archive for books in CSV")
    parser.add_argument("--csv", required=True, help="Input CSV file with books")
    parser.add_argument("--output", required=True, help="Output CSV file for results")
    parser.add_argument("--max-books", type=int, help="Maximum number of books to process")
    
    args = parser.parse_args()
    
    # Check if API key is available
    if not os.getenv("ANNAS_SECRET_KEY"):
        logger.error("Environment variable ANNAS_SECRET_KEY not set – export it before running.")
        return 1
    
    # Initialize extractor
    extractor = EfficientMD5Extractor()
    
    # Extract MD5s
    results = extractor.extract_md5s_from_csv(
        csv_path=args.csv,
        output_path=args.output,
        max_books=args.max_books
    )
    
    print(f"\nExtraction complete!")
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Found MD5s for {results['books_with_md5']} out of {results['total_books']} books")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
