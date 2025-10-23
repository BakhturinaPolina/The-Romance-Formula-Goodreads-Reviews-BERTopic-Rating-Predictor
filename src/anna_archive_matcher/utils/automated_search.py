#!/usr/bin/env python3
"""
Fully Automated Anna's Archive Search
Programmatically searches and extracts MD5 hashes without manual intervention
"""

import requests
import json
import logging
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote, urljoin
import random

logger = logging.getLogger(__name__)


class AutomatedAnnaArchiveSearcher:
    """
    Fully automated searcher for Anna's Archive
    """
    
    def __init__(self, delay_range: Tuple[float, float] = (1.0, 3.0)):
        """
        Initialize the automated searcher
        
        Args:
            delay_range: Random delay range between requests (min, max) seconds
        """
        self.base_url = "https://annas-archive.org"
        self.search_url = f"{self.base_url}/search"
        self.delay_range = delay_range
        
        # Session for maintaining cookies and headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Statistics
        self.stats = {
            'books_searched': 0,
            'books_found': 0,
            'downloads_found': 0,
            'errors': 0
        }
        
        logger.info("Automated Anna Archive searcher initialized")
    
    def search_book(self, title: str, author: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Search for a single book on Anna's Archive
        
        Args:
            title: Book title
            author: Author name
            max_retries: Maximum number of retry attempts
            
        Returns:
            Book metadata with download info or None if not found
        """
        self.stats['books_searched'] += 1
        
        # Create search query
        query = f'"{title}" "{author}"'
        
        for attempt in range(max_retries):
            try:
                # Random delay to avoid rate limiting
                delay = random.uniform(*self.delay_range)
                time.sleep(delay)
                
                # Search parameters
                params = {
                    'q': query,
                    'content': 'book',
                    'language': 'en',
                    'extension': 'epub,pdf'
                }
                
                logger.info(f"Searching: {title} by {author} (attempt {attempt + 1})")
                
                # Make search request
                response = self.session.get(self.search_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse search results
                results = self._parse_search_results(response.text, title, author)
                
                if results:
                    self.stats['books_found'] += 1
                    logger.info(f"Found: {title} by {author}")
                    return results
                else:
                    logger.info(f"Not found: {title} by {author}")
                    return None
                    
            except Exception as e:
                logger.error(f"Search error for '{title}' by '{author}' (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.stats['errors'] += 1
        
        return None
    
    def _parse_search_results(self, html_content: str, title: str, author: str) -> Optional[Dict]:
        """
        Parse search results HTML to find book matches
        
        Args:
            html_content: HTML content from search response
            title: Original book title
            author: Original author name
            
        Returns:
            Book metadata if found, None otherwise
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for search result items
            result_items = soup.find_all('div', class_='h-[125]') or soup.find_all('div', class_='h-[125px]')
            
            if not result_items:
                # Try alternative selectors
                result_items = soup.find_all('div', class_=re.compile(r'h-\[125'))
            
            if not result_items:
                # Look for any divs that might contain book results
                result_items = soup.find_all('div', class_=re.compile(r'book|result|item'))
            
            for item in result_items:
                # Extract book information
                book_info = self._extract_book_info(item, title, author)
                if book_info:
                    return book_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return None
    
    def _extract_book_info(self, item_element, original_title: str, original_author: str) -> Optional[Dict]:
        """
        Extract book information from a search result item
        
        Args:
            item_element: BeautifulSoup element containing book info
            original_title: Original book title for matching
            original_author: Original author for matching
            
        Returns:
            Book metadata if it's a good match, None otherwise
        """
        try:
            # Extract title
            title_element = item_element.find('h3') or item_element.find('h2') or item_element.find('h1')
            if not title_element:
                return None
            
            found_title = title_element.get_text(strip=True)
            
            # Extract author
            author_element = item_element.find('p') or item_element.find('div', class_=re.compile(r'author'))
            found_author = author_element.get_text(strip=True) if author_element else ""
            
            # Check if this is a good match
            if not self._is_good_match(original_title, original_author, found_title, found_author):
                return None
            
            # Extract download links
            download_links = self._extract_download_links(item_element)
            if not download_links:
                return None
            
            # Extract MD5 hash from download URL
            md5_hash = self._extract_md5_from_links(download_links)
            
            return {
                'original_title': original_title,
                'original_author': original_author,
                'found_title': found_title,
                'found_author': found_author,
                'download_links': download_links,
                'md5_hash': md5_hash,
                'file_formats': self._extract_file_formats(download_links),
                'match_quality': self._calculate_match_quality(original_title, original_author, found_title, found_author)
            }
            
        except Exception as e:
            logger.error(f"Error extracting book info: {e}")
            return None
    
    def _is_good_match(self, orig_title: str, orig_author: str, found_title: str, found_author: str) -> bool:
        """
        Determine if the found book is a good match for the original
        
        Args:
            orig_title: Original book title
            orig_author: Original author
            found_title: Found book title
            found_author: Found author
            
        Returns:
            True if it's a good match
        """
        # Normalize strings for comparison
        orig_title_norm = re.sub(r'[^\w\s]', '', orig_title.lower())
        found_title_norm = re.sub(r'[^\w\s]', '', found_title.lower())
        orig_author_norm = re.sub(r'[^\w\s]', '', orig_author.lower())
        found_author_norm = re.sub(r'[^\w\s]', '', found_author.lower())
        
        # Check title similarity (at least 70% of words should match)
        orig_title_words = set(orig_title_norm.split())
        found_title_words = set(found_title_norm.split())
        
        if len(orig_title_words) > 0:
            title_similarity = len(orig_title_words.intersection(found_title_words)) / len(orig_title_words)
            if title_similarity < 0.7:
                return False
        
        # Check author similarity
        orig_author_words = set(orig_author_norm.split())
        found_author_words = set(found_author_norm.split())
        
        if len(orig_author_words) > 0:
            author_similarity = len(orig_author_words.intersection(found_author_words)) / len(orig_author_words)
            if author_similarity < 0.8:
                return False
        
        return True
    
    def _extract_download_links(self, item_element) -> List[str]:
        """
        Extract download links from a search result item
        
        Args:
            item_element: BeautifulSoup element
            
        Returns:
            List of download URLs
        """
        links = []
        
        # Look for download links
        for link in item_element.find_all('a', href=True):
            href = link['href']
            if '/md5/' in href or '/download/' in href:
                full_url = urljoin(self.base_url, href)
                links.append(full_url)
        
        return links
    
    def _extract_md5_from_links(self, links: List[str]) -> Optional[str]:
        """
        Extract MD5 hash from download links
        
        Args:
            links: List of download URLs
            
        Returns:
            MD5 hash if found, None otherwise
        """
        for link in links:
            # Look for MD5 in URL
            md5_match = re.search(r'/md5/([a-f0-9]{32})', link)
            if md5_match:
                return md5_match.group(1)
        
        return None
    
    def _extract_file_formats(self, links: List[str]) -> List[str]:
        """
        Extract file formats from download links
        
        Args:
            links: List of download URLs
            
        Returns:
            List of file formats
        """
        formats = []
        for link in links:
            if '.epub' in link.lower():
                formats.append('epub')
            elif '.pdf' in link.lower():
                formats.append('pdf')
            elif '.mobi' in link.lower():
                formats.append('mobi')
        
        return list(set(formats))
    
    def _calculate_match_quality(self, orig_title: str, orig_author: str, found_title: str, found_author: str) -> float:
        """
        Calculate match quality score (0.0 to 1.0)
        
        Args:
            orig_title: Original title
            orig_author: Original author
            found_title: Found title
            found_author: Found author
            
        Returns:
            Match quality score
        """
        # Simple similarity calculation
        title_sim = self._calculate_similarity(orig_title, found_title)
        author_sim = self._calculate_similarity(orig_author, found_author)
        
        return (title_sim + author_sim) / 2
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using simple word overlap
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        words1 = set(re.sub(r'[^\w\s]', '', str1.lower()).split())
        words2 = set(re.sub(r'[^\w\s]', '', str2.lower()).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def search_books_batch(self, books_df: pd.DataFrame, max_books: int = None) -> pd.DataFrame:
        """
        Search multiple books in batch
        
        Args:
            books_df: DataFrame with book information
            max_books: Maximum number of books to search
            
        Returns:
            DataFrame with search results
        """
        if max_books:
            books_df = books_df.head(max_books)
        
        results = []
        
        for idx, book in books_df.iterrows():
            logger.info(f"Processing book {idx + 1}/{len(books_df)}")
            
            result = self.search_book(book['title'], book['author_name'])
            
            if result:
                result['work_id'] = book['work_id']
                result['publication_year'] = book.get('publication_year')
                result['original_rating'] = book.get('average_rating_weighted_mean')
                result['original_reviews'] = book.get('ratings_count_sum')
                results.append(result)
            
            # Show progress every 10 books
            if (idx + 1) % 10 == 0:
                self._show_progress()
        
        return pd.DataFrame(results)
    
    def _show_progress(self):
        """Show current search progress"""
        success_rate = (self.stats['books_found'] / self.stats['books_searched'] * 100) if self.stats['books_searched'] > 0 else 0
        
        logger.info(f"Progress: {self.stats['books_searched']} searched, "
                   f"{self.stats['books_found']} found ({success_rate:.1f}%), "
                   f"{self.stats['errors']} errors")
    
    def get_statistics(self) -> Dict:
        """Get search statistics"""
        return self.stats.copy()


def main():
    """
    Main function for automated search
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Anna Archive Search')
    parser.add_argument('--romance-csv', required=True,
                       help='Path to romance books CSV file')
    parser.add_argument('--output-csv', default='automated_search_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--max-books', type=int, default=100,
                       help='Maximum number of books to search')
    parser.add_argument('--delay-min', type=float, default=1.0,
                       help='Minimum delay between requests (seconds)')
    parser.add_argument('--delay-max', type=float, default=3.0,
                       help='Maximum delay between requests (seconds)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('automated_search.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load books
    logger.info(f"Loading books from {args.romance_csv}")
    books_df = pd.read_csv(args.romance_csv)
    
    # Initialize searcher
    searcher = AutomatedAnnaArchiveSearcher(
        delay_range=(args.delay_min, args.delay_max)
    )
    
    # Search books
    logger.info(f"Starting automated search for {min(args.max_books, len(books_df))} books")
    results_df = searcher.search_books_batch(books_df, args.max_books)
    
    # Save results
    if not results_df.empty:
        results_df.to_csv(args.output_csv, index=False)
        logger.info(f"Saved {len(results_df)} results to {args.output_csv}")
        
        # Create download-ready CSV
        download_df = results_df[results_df['md5_hash'].notna()].copy()
        if not download_df.empty:
            download_csv = args.output_csv.replace('.csv', '_download_ready.csv')
            download_df[['work_id', 'original_title', 'original_author', 'md5_hash', 'file_formats']].to_csv(download_csv, index=False)
            logger.info(f"Created download-ready CSV: {download_csv}")
    else:
        logger.warning("No books found!")
    
    # Show final statistics
    stats = searcher.get_statistics()
    logger.info(f"Final statistics: {stats}")


if __name__ == "__main__":
    main()
