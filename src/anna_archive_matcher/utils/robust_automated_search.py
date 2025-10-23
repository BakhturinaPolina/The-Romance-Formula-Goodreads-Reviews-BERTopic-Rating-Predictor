#!/usr/bin/env python3
"""
Robust Automated Anna's Archive Search
Handles SSL issues and provides multiple fallback strategies
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
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class RobustAutomatedSearcher:
    """
    Robust automated searcher with multiple fallback strategies
    """
    
    def __init__(self, delay_range: Tuple[float, float] = (2.0, 5.0)):
        """
        Initialize the robust searcher
        
        Args:
            delay_range: Random delay range between requests (min, max) seconds
        """
        self.delay_range = delay_range
        
        # Multiple base URLs to try
        self.base_urls = [
            "https://annas-archive.org",
            "https://annas-archive.li",
            "https://annas-archive.net"
        ]
        
        # Session configuration
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification
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
            'errors': 0,
            'ssl_errors': 0,
            'timeout_errors': 0
        }
        
        logger.info("Robust automated searcher initialized")
    
    def search_book(self, title: str, author: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Search for a single book with multiple fallback strategies
        
        Args:
            title: Book title
            author: Author name
            max_retries: Maximum number of retry attempts
            
        Returns:
            Book metadata with download info or None if not found
        """
        self.stats['books_searched'] += 1
        
        # Try multiple search strategies
        search_strategies = [
            f'"{title}" "{author}"',
            f'"{title}" {author}',
            f'{title} {author} romance',
            f'"{title}" romance',
            f'{author} romance'
        ]
        
        for strategy_idx, query in enumerate(search_strategies):
            for attempt in range(max_retries):
                try:
                    # Random delay
                    delay = random.uniform(*self.delay_range)
                    time.sleep(delay)
                    
                    # Try different base URLs
                    for base_url in self.base_urls:
                        try:
                            search_url = f"{base_url}/search"
                            
                            params = {
                                'q': query,
                                'content': 'book',
                                'language': 'en',
                                'extension': 'epub,pdf'
                            }
                            
                            logger.info(f"Searching: {title} by {author} (strategy {strategy_idx + 1}, attempt {attempt + 1}, URL: {base_url})")
                            
                            response = self.session.get(search_url, params=params, timeout=30)
                            response.raise_for_status()
                            
                            # Parse results
                            results = self._parse_search_results(response.text, title, author)
                            
                            if results:
                                self.stats['books_found'] += 1
                                logger.info(f"Found: {title} by {author}")
                                return results
                            
                        except requests.exceptions.SSLError as e:
                            self.stats['ssl_errors'] += 1
                            logger.warning(f"SSL error with {base_url}: {e}")
                            continue
                        except requests.exceptions.Timeout as e:
                            self.stats['timeout_errors'] += 1
                            logger.warning(f"Timeout with {base_url}: {e}")
                            continue
                        except Exception as e:
                            logger.warning(f"Error with {base_url}: {e}")
                            continue
                    
                    # If all URLs failed, wait before retry
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        
                except Exception as e:
                    logger.error(f"Search error for '{title}' by '{author}' (strategy {strategy_idx + 1}, attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        self.stats['errors'] += 1
        
        logger.info(f"Not found: {title} by {author}")
        return None
    
    def _parse_search_results(self, html_content: str, title: str, author: str) -> Optional[Dict]:
        """
        Parse search results with multiple parsing strategies
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Strategy 1: Look for specific Anna's Archive result containers
            result_selectors = [
                'div[class*="h-[125"]',
                'div[class*="book"]',
                'div[class*="result"]',
                'div[class*="item"]',
                '.search-result',
                '.book-item'
            ]
            
            result_items = []
            for selector in result_selectors:
                items = soup.select(selector)
                if items:
                    result_items = items
                    break
            
            # Strategy 2: Look for links containing MD5 hashes
            if not result_items:
                md5_links = soup.find_all('a', href=re.compile(r'/md5/[a-f0-9]{32}'))
                if md5_links:
                    # Create mock result items from MD5 links
                    result_items = md5_links
            
            # Strategy 3: Look for any links that might be book downloads
            if not result_items:
                download_links = soup.find_all('a', href=re.compile(r'/(md5|download)/'))
                if download_links:
                    result_items = download_links
            
            for item in result_items:
                book_info = self._extract_book_info(item, title, author)
                if book_info:
                    return book_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return None
    
    def _extract_book_info(self, item_element, original_title: str, original_author: str) -> Optional[Dict]:
        """
        Extract book information with flexible parsing
        """
        try:
            # Extract title (try multiple methods)
            title = self._extract_text(item_element, ['h1', 'h2', 'h3', 'h4', '.title', '.book-title'])
            
            # Extract author (try multiple methods)
            author = self._extract_text(item_element, ['p', '.author', '.book-author', 'span'])
            
            # If we don't have title/author from text, try to extract from links
            if not title or not author:
                link_text = item_element.get_text(strip=True) if hasattr(item_element, 'get_text') else str(item_element)
                if not title:
                    title = self._extract_title_from_text(link_text, original_title)
                if not author:
                    author = self._extract_author_from_text(link_text, original_author)
            
            # Check if this is a good match
            if not self._is_good_match(original_title, original_author, title, author):
                return None
            
            # Extract download links
            download_links = self._extract_download_links(item_element)
            if not download_links:
                return None
            
            # Extract MD5 hash
            md5_hash = self._extract_md5_from_links(download_links)
            
            return {
                'original_title': original_title,
                'original_author': original_author,
                'found_title': title,
                'found_author': author,
                'download_links': download_links,
                'md5_hash': md5_hash,
                'file_formats': self._extract_file_formats(download_links),
                'match_quality': self._calculate_match_quality(original_title, original_author, title, author)
            }
            
        except Exception as e:
            logger.error(f"Error extracting book info: {e}")
            return None
    
    def _extract_text(self, element, selectors: List[str]) -> str:
        """Extract text using multiple selectors"""
        for selector in selectors:
            try:
                if selector.startswith('.'):
                    found = element.find(class_=selector[1:])
                else:
                    found = element.find(selector)
                if found:
                    return found.get_text(strip=True)
            except:
                continue
        return ""
    
    def _extract_title_from_text(self, text: str, original_title: str) -> str:
        """Extract title from text using heuristics"""
        # Look for patterns that might be titles
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 5 and len(line) < 100:
                # Check if it contains words from original title
                orig_words = set(original_title.lower().split())
                line_words = set(line.lower().split())
                if len(orig_words.intersection(line_words)) > 0:
                    return line
        return ""
    
    def _extract_author_from_text(self, text: str, original_author: str) -> str:
        """Extract author from text using heuristics"""
        # Look for patterns that might be authors
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 3 and len(line) < 50:
                # Check if it contains words from original author
                orig_words = set(original_author.lower().split())
                line_words = set(line.lower().split())
                if len(orig_words.intersection(line_words)) > 0:
                    return line
        return ""
    
    def _is_good_match(self, orig_title: str, orig_author: str, found_title: str, found_author: str) -> bool:
        """Flexible matching with lower thresholds"""
        if not found_title or not found_author:
            return False
        
        # Normalize strings
        orig_title_norm = re.sub(r'[^\w\s]', '', orig_title.lower())
        found_title_norm = re.sub(r'[^\w\s]', '', found_title.lower())
        orig_author_norm = re.sub(r'[^\w\s]', '', orig_author.lower())
        found_author_norm = re.sub(r'[^\w\s]', '', found_author.lower())
        
        # Check title similarity (lowered threshold)
        orig_title_words = set(orig_title_norm.split())
        found_title_words = set(found_title_norm.split())
        
        if len(orig_title_words) > 0:
            title_similarity = len(orig_title_words.intersection(found_title_words)) / len(orig_title_words)
            if title_similarity < 0.5:  # Lowered from 0.7
                return False
        
        # Check author similarity (lowered threshold)
        orig_author_words = set(orig_author_norm.split())
        found_author_words = set(found_author_norm.split())
        
        if len(orig_author_words) > 0:
            author_similarity = len(orig_author_words.intersection(found_author_words)) / len(orig_author_words)
            if author_similarity < 0.6:  # Lowered from 0.8
                return False
        
        return True
    
    def _extract_download_links(self, item_element) -> List[str]:
        """Extract download links with multiple strategies"""
        links = []
        
        # Strategy 1: Direct href extraction
        if hasattr(item_element, 'find_all'):
            for link in item_element.find_all('a', href=True):
                href = link['href']
                if '/md5/' in href or '/download/' in href:
                    full_url = urljoin(self.base_urls[0], href)
                    links.append(full_url)
        
        # Strategy 2: Extract from text if it's a link
        if not links and hasattr(item_element, 'get_text'):
            text = item_element.get_text()
            md5_matches = re.findall(r'/md5/([a-f0-9]{32})', text)
            for md5 in md5_matches:
                links.append(f"{self.base_urls[0]}/md5/{md5}")
        
        return links
    
    def _extract_md5_from_links(self, links: List[str]) -> Optional[str]:
        """Extract MD5 hash from links"""
        for link in links:
            md5_match = re.search(r'/md5/([a-f0-9]{32})', link)
            if md5_match:
                return md5_match.group(1)
        return None
    
    def _extract_file_formats(self, links: List[str]) -> List[str]:
        """Extract file formats from links"""
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
        """Calculate match quality score"""
        title_sim = self._calculate_similarity(orig_title, found_title)
        author_sim = self._calculate_similarity(orig_author, found_author)
        return (title_sim + author_sim) / 2
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity"""
        words1 = set(re.sub(r'[^\w\s]', '', str1.lower()).split())
        words2 = set(re.sub(r'[^\w\s]', '', str2.lower()).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def search_books_batch(self, books_df: pd.DataFrame, max_books: int = None) -> pd.DataFrame:
        """Search multiple books in batch"""
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
            
            # Show progress every 5 books
            if (idx + 1) % 5 == 0:
                self._show_progress()
        
        return pd.DataFrame(results)
    
    def _show_progress(self):
        """Show current search progress"""
        success_rate = (self.stats['books_found'] / self.stats['books_searched'] * 100) if self.stats['books_searched'] > 0 else 0
        
        logger.info(f"Progress: {self.stats['books_searched']} searched, "
                   f"{self.stats['books_found']} found ({success_rate:.1f}%), "
                   f"{self.stats['errors']} errors, "
                   f"{self.stats['ssl_errors']} SSL errors, "
                   f"{self.stats['timeout_errors']} timeouts")
    
    def get_statistics(self) -> Dict:
        """Get search statistics"""
        return self.stats.copy()


def main():
    """Main function for robust automated search"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust Automated Anna Archive Search')
    parser.add_argument('--romance-csv', required=True,
                       help='Path to romance books CSV file')
    parser.add_argument('--output-csv', default='robust_automated_search_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--max-books', type=int, default=50,
                       help='Maximum number of books to search')
    parser.add_argument('--delay-min', type=float, default=2.0,
                       help='Minimum delay between requests (seconds)')
    parser.add_argument('--delay-max', type=float, default=5.0,
                       help='Maximum delay between requests (seconds)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('robust_automated_search.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load books
    logger.info(f"Loading books from {args.romance_csv}")
    books_df = pd.read_csv(args.romance_csv)
    
    # Initialize searcher
    searcher = RobustAutomatedSearcher(
        delay_range=(args.delay_min, args.delay_max)
    )
    
    # Search books
    logger.info(f"Starting robust automated search for {min(args.max_books, len(books_df))} books")
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
