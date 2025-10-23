#!/usr/bin/env python3
"""
MD5 Hash Extractor for Anna's Archive
Searches Anna's Archive to find MD5 hashes for books in CSV format
"""

import os
import re
import json
import logging
import requests
import subprocess
import shutil
import time
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin
import urllib3
from bs4 import BeautifulSoup

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnnaArchiveMD5Extractor:
    """Extract MD5 hashes from Anna's Archive for books in CSV"""
    
    def __init__(self, delay_range: Tuple[float, float] = (2.0, 5.0), use_tor: bool = True):
        """
        Initialize the MD5 extractor
        
        Args:
            delay_range: Random delay range between requests (min, max) seconds
            use_tor: Whether to use Tor for requests
        """
        self.delay_range = delay_range
        self.use_tor = use_tor
        
        # Multiple base URLs to try
        self.base_urls = [
            "https://annas-archive.org",
            "https://annas-archive.li",
            "https://annas-archive.net",
            "https://annas-archive.se"
        ]
        
        # Check for torsocks availability
        self.torsocks_available = shutil.which("torsocks") is not None
        if self.use_tor and not self.torsocks_available:
            logger.warning("torsocks not found - will try without Tor")
            self.use_tor = False
        
        # Initialize session
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
            'md5s_found': 0,
            'errors': 0,
            'ssl_errors': 0,
            'timeout_errors': 0
        }
        
        logger.info(f"MD5 Extractor initialized with {len(self.base_urls)} mirrors, Tor: {self.use_tor}")
    
    def _make_request(self, url: str, params: Dict = None) -> requests.Response:
        """Make a request with Tor support if available"""
        if self.use_tor and self.torsocks_available:
            return self._make_tor_request(url, params)
        else:
            return self.session.get(url, params=params, timeout=30)
    
    def _make_tor_request(self, url: str, params: Dict = None) -> requests.Response:
        """Make a request through Tor using torsocks"""
        cmd = ["torsocks", "curl", "-s", "-L"]
        
        # Add parameters
        if params:
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url_with_params = f"{url}?{param_str}"
        else:
            url_with_params = url
        
        cmd.append(url_with_params)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Create a mock response object
            response = requests.Response()
            response.status_code = 200 if result.returncode == 0 else 500
            response._content = result.stdout.encode('utf-8')
            response.headers = {'content-type': 'text/html'}
            response.url = url_with_params
            
            return response
            
        except subprocess.TimeoutExpired:
            response = requests.Response()
            response.status_code = 408
            response._content = b''
            response.headers = {}
            response.url = url_with_params
            return response
        except Exception as e:
            response = requests.Response()
            response.status_code = 500
            response._content = str(e).encode('utf-8')
            response.headers = {}
            response.url = url_with_params
            return response
    
    def search_book_md5(self, title: str, author: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Search for a book and extract MD5 hash
        
        Args:
            title: Book title
            author: Author name
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with book info and MD5 hash or None if not found
        """
        self.stats['books_searched'] += 1
        
        # Try multiple search strategies
        search_strategies = [
            f'"{title}" "{author}"',
            f'"{title}" {author}',
            f'{title} {author}',
            f'"{title}"',
            f'{author} {title}'
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
                                'extension': 'epub,pdf,mobi'
                            }
                            
                            logger.info(f"Searching: {title} by {author} (strategy {strategy_idx + 1}, attempt {attempt + 1}, URL: {base_url})")
                            
                            response = self._make_request(search_url, params)
                            
                            if response.status_code == 200:
                                # Parse results
                                result = self._parse_search_results(response.text, title, author)
                                
                                if result:
                                    self.stats['books_found'] += 1
                                    self.stats['md5s_found'] += 1
                                    logger.info(f"Found MD5: {result['md5_hash']} for {title} by {author}")
                                    return result
                            
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
        
        logger.info(f"No MD5 found: {title} by {author}")
        return None
    
    def _parse_search_results(self, html_content: str, original_title: str, original_author: str) -> Optional[Dict]:
        """Parse search results HTML to extract book info and MD5"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for search result items
            result_items = soup.find_all(['div', 'article'], class_=re.compile(r'item|result|book'))
            
            if not result_items:
                # Try alternative selectors
                result_items = soup.find_all('div', class_=re.compile(r'search|result'))
            
            for item in result_items:
                book_info = self._extract_book_info(item, original_title, original_author)
                if book_info:
                    return book_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return None
    
    def _extract_book_info(self, item_element, original_title: str, original_author: str) -> Optional[Dict]:
        """Extract book information from a search result item"""
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
            if not md5_hash:
                return None
            
            return {
                'original_title': original_title,
                'original_author': original_author,
                'found_title': title,
                'found_author': author,
                'md5_hash': md5_hash,
                'download_links': download_links,
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
                found = element.select_one(selector)
                if found:
                    text = found.get_text(strip=True)
                    if text:
                        return text
            except:
                continue
        return ""
    
    def _extract_title_from_text(self, text: str, original_title: str) -> str:
        """Extract title from text using heuristics"""
        # Simple heuristic: look for text that contains words from original title
        original_words = set(original_title.lower().split())
        text_words = text.split()
        
        for i, word in enumerate(text_words):
            if word.lower() in original_words:
                # Try to extract a title around this word
                start = max(0, i - 2)
                end = min(len(text_words), i + 5)
                return " ".join(text_words[start:end])
        
        return ""
    
    def _extract_author_from_text(self, text: str, original_author: str) -> str:
        """Extract author from text using heuristics"""
        # Simple heuristic: look for text that contains words from original author
        original_words = set(original_author.lower().split())
        text_words = text.split()
        
        for i, word in enumerate(text_words):
            if word.lower() in original_words:
                # Try to extract an author name around this word
                start = max(0, i - 1)
                end = min(len(text_words), i + 3)
                return " ".join(text_words[start:end])
        
        return ""
    
    def _is_good_match(self, original_title: str, original_author: str, found_title: str, found_author: str) -> bool:
        """Check if the found book is a good match"""
        if not found_title or not found_author:
            return False
        
        # Normalize text for comparison
        orig_title_norm = re.sub(r'[^a-z0-9]', '', original_title.lower())
        found_title_norm = re.sub(r'[^a-z0-9]', '', found_title.lower())
        orig_author_norm = re.sub(r'[^a-z0-9]', '', original_author.lower())
        found_author_norm = re.sub(r'[^a-z0-9]', '', found_author.lower())
        
        # Check for significant overlap
        title_overlap = len(set(orig_title_norm) & set(found_title_norm)) / max(len(orig_title_norm), 1)
        author_overlap = len(set(orig_author_norm) & set(found_author_norm)) / max(len(orig_author_norm), 1)
        
        # Require at least 60% overlap in title and 50% in author
        return title_overlap >= 0.6 and author_overlap >= 0.5
    
    def _extract_download_links(self, item_element) -> List[str]:
        """Extract download links from item element"""
        links = []
        
        # Look for links with common download patterns
        for link in item_element.find_all('a', href=True):
            href = link['href']
            if any(pattern in href.lower() for pattern in ['download', 'md5', 'file', 'epub', 'pdf']):
                links.append(href)
        
        return links
    
    def _extract_md5_from_links(self, links: List[str]) -> Optional[str]:
        """Extract MD5 hash from download links"""
        md5_pattern = re.compile(r'([a-fA-F0-9]{32})')
        
        for link in links:
            match = md5_pattern.search(link)
            if match:
                return match.group(1).lower()
        
        return None
    
    def _extract_file_formats(self, links: List[str]) -> List[str]:
        """Extract file formats from download links"""
        formats = []
        for link in links:
            if '.epub' in link.lower():
                formats.append('epub')
            elif '.pdf' in link.lower():
                formats.append('pdf')
            elif '.mobi' in link.lower():
                formats.append('mobi')
        
        return list(set(formats))
    
    def _calculate_match_quality(self, original_title: str, original_author: str, found_title: str, found_author: str) -> float:
        """Calculate match quality score (0-1)"""
        # Normalize text
        orig_title_norm = re.sub(r'[^a-z0-9]', '', original_title.lower())
        found_title_norm = re.sub(r'[^a-z0-9]', '', found_title.lower())
        orig_author_norm = re.sub(r'[^a-z0-9]', '', original_author.lower())
        found_author_norm = re.sub(r'[^a-z0-9]', '', found_author.lower())
        
        # Calculate overlaps
        title_overlap = len(set(orig_title_norm) & set(found_title_norm)) / max(len(orig_title_norm), 1)
        author_overlap = len(set(orig_author_norm) & set(found_author_norm)) / max(len(orig_author_norm), 1)
        
        # Weighted average (title is more important)
        return (title_overlap * 0.7 + author_overlap * 0.3)
    
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
                    'download_links': ';'.join(result['download_links'][:3])  # Limit to first 3 links
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
        logger.info(f"SSL errors: {self.stats['ssl_errors']}")
        logger.info(f"Timeout errors: {self.stats['timeout_errors']}")
        logger.info("=" * 50)


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract MD5 hashes from Anna's Archive for books in CSV")
    parser.add_argument("--csv", required=True, help="Input CSV file with books")
    parser.add_argument("--output", required=True, help="Output CSV file for results")
    parser.add_argument("--max-books", type=int, help="Maximum number of books to process")
    parser.add_argument("--delay-min", type=float, default=2.0, help="Minimum delay between requests")
    parser.add_argument("--delay-max", type=float, default=5.0, help="Maximum delay between requests")
    parser.add_argument("--no-tor", action="store_true", help="Disable Tor usage")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = AnnaArchiveMD5Extractor(
        delay_range=(args.delay_min, args.delay_max),
        use_tor=not args.no_tor
    )
    
    # Extract MD5s
    results = extractor.extract_md5s_from_csv(
        csv_path=args.csv,
        output_path=args.output,
        max_books=args.max_books
    )
    
    print(f"\nExtraction complete!")
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Found MD5s for {results['books_with_md5']} out of {results['total_books']} books")


if __name__ == "__main__":
    main()
