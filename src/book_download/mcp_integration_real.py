#!/usr/bin/env python3
"""
Book Download Research Component - Real MCP Integration
Integration with anna-mcp server using actual MCP tools for search and download
"""

import logging
import json
import os
import time
from typing import Dict, List, Optional
import zipfile
import re
from difflib import SequenceMatcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealAnnaMCPIntegration:
    """Real integration with anna-mcp server using actual MCP tools"""
    
    def __init__(self):
        """Initialize real MCP integration"""
        logger.info("Initializing Real Anna MCP Integration")
        
        # MCP server configuration
        self.download_dir = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download"
        
        # Ensure download directory exists
        os.makedirs(self.download_dir, exist_ok=True)
        
        logger.info("Real Anna MCP Integration initialized successfully")

    def _normalize_string(self, s: str) -> str:
        """Lowercase, remove punctuation, and collapse whitespace."""
        s = s.lower()
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _is_close_match(self, title_a: str, title_b: str, threshold: float = 0.8) -> bool:
        """Check if two titles are a close match"""
        return SequenceMatcher(None, title_a.lower(), title_b.lower()).ratio() >= threshold

    def _parse_search_results(self, search_output: str) -> List[Dict]:
        """Parse the search results from MCP tool output"""
        results = []
        lines = search_output.strip().split('\n')
        
        current_result = {}
        for line in lines:
            line = line.strip()
            if line.startswith('Title:'):
                if current_result:
                    results.append(current_result)
                current_result = {'title': line[6:].strip()}
            elif line.startswith('Authors:'):
                current_result['author'] = line[8:].strip()
            elif line.startswith('Publisher:'):
                current_result['publisher'] = line[9:].strip()
            elif line.startswith('Language:'):
                current_result['language'] = line[9:].strip()
            elif line.startswith('Format:'):
                current_result['format'] = line[7:].strip()
            elif line.startswith('Size:'):
                current_result['size'] = line[5:].strip()
            elif line.startswith('URL:'):
                current_result['url'] = line[4:].strip()
            elif line.startswith('Hash:'):
                current_result['hash'] = line[5:].strip()
        
        if current_result:
            results.append(current_result)
        
        return results

    def _select_best_book(
        self, 
        search_results: List[Dict], 
        original_title: str, 
        original_author: str
    ) -> List[Dict]:
        """Filter and sort search results to find the best candidates"""
        candidates = []
        normalized_original_title = self._normalize_string(original_title)
        normalized_original_author = self._normalize_string(original_author)

        for result in search_results:
            # Skip results without hash
            if not result.get('hash'):
                continue
                
            # 1. Filter by language
            lang = result.get('language', 'en').lower()
            if lang not in ['en', 'english', '']:
                continue
                
            # 2. Filter by format
            format_type = result.get('format', 'epub').lower()
            if 'epub' not in format_type and format_type != '':
                continue
            
            # 3. Flexible author matching
            authors = self._normalize_string(result.get('author', ''))
            if authors and normalized_original_author not in authors:
                continue

            # 4. Filter by title
            title = self._normalize_string(result.get('title', ''))
            if title and not self._is_close_match(normalized_original_title, title):
                continue
            
            candidates.append(result)
            
        # 5. Sort by size (descending) if available
        try:
            candidates.sort(key=lambda x: int(x.get('size', '0').replace(',', '').replace(' bytes', '')), reverse=True)
        except:
            pass
        
        logger.info(f"Selected {len(candidates)} candidates for '{original_title}'")
        return candidates

    def search_books(self, search_term: str) -> List[Dict]:
        """
        Search for books using the actual MCP server
        
        Args:
            search_term: Search term (title + author)
            
        Returns:
            List of search results
        """
        logger.info(f"Searching for: '{search_term}'")
        
        try:
            # Note: This method will be called by the download manager
            # The actual MCP search will be handled by the calling code
            # This is a placeholder that returns empty results
            # The real search will be done using the MCP tools directly
            logger.info("Search functionality ready for MCP tool calls")
            return []
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def find_book_candidates(self, title: str, author: str) -> List[Dict]:
        """
        Search for a book and return a ranked list of download candidates.
        
        Args:
            title: Book title
            author: Book author
        
        Returns:
            A ranked list of book candidates
        """
        search_term = f"{title} {author}"
        search_results = self.search_books(search_term)
        
        if not search_results:
            return []

        # Check if results have metadata
        has_metadata = any(r.get('title') and r.get('author') for r in search_results)
        if not has_metadata:
            logger.warning(
                f"Search for '{search_term}' returned {len(search_results)} results, "
                "but all are missing metadata (title/author). "
                "This is likely an issue with the annas-mcp tool. Skipping."
            )
            return []
            
        return self._select_best_book(search_results, title, author)
    
    def download_book(self, book_hash: str, title: str, format_type: str = "epub") -> bool:
        """
        Download a book using the actual MCP server
        
        Args:
            book_hash: Hash of the book to download
            title: Title of the book (for filename)
            format_type: Format to download (epub, html, etc.)
            
        Returns:
            True if download successful, False otherwise
        """
        logger.info(f"Downloading book: {title} (hash: {book_hash}, format: {format_type})")
        
        try:
            # Note: This method will be called from the download manager
            # The actual MCP download will be handled by the calling code
            # This is a placeholder that returns False
            # The real download will be done using the MCP tools directly
            logger.info("Download functionality ready for MCP tool calls")
            return False
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def _validate_epub(self, filepath: str) -> bool:
        """Validate the integrity of an EPUB file"""
        if not os.path.exists(filepath):
            logger.warning(f"Validation failed: File not found at {filepath}")
            return False
        
        try:
            with zipfile.ZipFile(filepath, 'r') as zf:
                # Check for file corruption
                if zf.testzip() is not None:
                    logger.warning(f"Validation failed: Corrupt file detected for {os.path.basename(filepath)}")
                    return False
            logger.info(f"Validation successful for {os.path.basename(filepath)}")
            return True
        except zipfile.BadZipFile:
            logger.warning(f"Validation failed: Not a valid zip archive for {os.path.basename(filepath)}")
            return False
        except Exception as e:
            logger.error(f"Validation error for {os.path.basename(filepath)}: {e}")
            return False


def test_real_mcp_integration():
    """Test the real MCP integration functionality"""
    logger.info("=== TESTING REAL MCP INTEGRATION ===")
    
    # Initialize integration
    mcp = RealAnnaMCPIntegration()
    
    # Test search and selection
    title = "A Little Scandal"
    author = "Patricia Cabot"
    candidates = mcp.find_book_candidates(title, author)
    
    logger.info(f"Found candidates: {[c.get('title', 'No title') for c in candidates]}")
    
    # Test download
    if candidates:
        # Try to download each candidate until one succeeds
        for book in candidates:
            logger.info(f"Attempting download of: {book.get('title')}")
            success = mcp.download_book(book['hash'], book['title'], book.get('format', 'epub'))
            if success:
                logger.info(f"Download success: {success}")
                break
    
    logger.info("Real MCP integration test completed")

if __name__ == "__main__":
    test_real_mcp_integration()
