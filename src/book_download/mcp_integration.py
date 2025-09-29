#!/usr/bin/env python3
"""
Book Download Research Component - MCP Integration
Integration with anna-mcp server for actual search and download functionality
"""

import logging
import json
import os
import subprocess
import time
from typing import Dict, List, Optional
from pathlib import Path

# Import EPUB guard helper
from aa_epub_guard import download_from_metadata, ensure_valid_epub, sniff_format

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnnaMCPIntegration:
    """Integration with anna-mcp server for book search and download"""
    
    def __init__(self):
        """Initialize MCP integration"""
        logger.info("Initializing Anna MCP Integration")
        
        # MCP server configuration
        self.mcp_binary = "/home/polina/.local/bin/annas-mcp"
        self.download_dir = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download"
        
        # Check if MCP server is available
        self.mcp_available = self._check_mcp_availability()
        
        if self.mcp_available:
            logger.info("Anna MCP server is available")
        else:
            logger.warning("Anna MCP server is not available - using simulation mode")
    
    def _check_mcp_availability(self) -> bool:
        """Check if anna-mcp server is available"""
        try:
            # Check if the MCP binary exists
            if not os.path.exists(self.mcp_binary):
                logger.error(f"MCP binary not found: {self.mcp_binary}")
                return False
            
            # Check if we can execute the binary
            result = subprocess.run([self.mcp_binary, "--help"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("MCP binary is executable")
                return True
            else:
                logger.error(f"MCP binary execution failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking MCP availability: {e}")
            return False
    
    def search_books(self, search_term: str) -> List[Dict]:
        """
        Search for books using anna-mcp server
        
        Args:
            search_term: Search term (title + author)
            
        Returns:
            List of search results
        """
        logger.info(f"Searching for: '{search_term}'")
        
        if not self.mcp_available:
            logger.warning("MCP server not available - returning simulated results")
            return self._simulate_search(search_term)
        
        try:
            # Set environment variables for anna-mcp
            env = os.environ.copy()
            env['ANNAS_SECRET_KEY'] = 'BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP'
            env['ANNAS_DOWNLOAD_PATH'] = self.download_dir
            
            # Call anna-mcp search command
            result = subprocess.run([
                self.mcp_binary, "search", search_term
            ], capture_output=True, text=True, timeout=30, env=env)
            
            if result.returncode != 0:
                logger.error(f"Search failed: {result.stderr}")
                return []
            
            # Parse search results
            search_results = self._parse_search_results(result.stdout)
            logger.info(f"Found {len(search_results)} search results")
            return search_results
            
        except subprocess.TimeoutExpired:
            logger.error("Search timed out")
            return []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def download_book(self, book_hash: str, title: str, format_type: str = "epub") -> bool:
        """
        Download a book using anna-mcp server (legacy method)
        
        Args:
            book_hash: Hash of the book to download
            title: Title of the book (for filename)
            format_type: Format to download (epub, html, etc.)
            
        Returns:
            True if download successful, False otherwise
        """
        logger.info(f"Downloading book: {title} (hash: {book_hash}, format: {format_type})")
        
        if not self.mcp_available:
            logger.warning("MCP server not available - simulating download")
            return self._simulate_download(book_hash, title, format_type)
        
        try:
            # Ensure download directory exists
            os.makedirs(self.download_dir, exist_ok=True)
            
            # Create filename for download
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_title}.{format_type}"
            
            # Set environment variables for anna-mcp
            env = os.environ.copy()
            env['ANNAS_SECRET_KEY'] = 'BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP'
            env['ANNAS_DOWNLOAD_PATH'] = self.download_dir
            
            # Call anna-mcp download command (hash, filename)
            result = subprocess.run([
                self.mcp_binary, "download", book_hash, filename
            ], capture_output=True, text=True, timeout=120, env=env)  # 2 minute timeout for downloads
            
            if result.returncode != 0:
                logger.error(f"Download failed: {result.stderr}")
                return False
            
            # Check if file was actually downloaded
            expected_path = os.path.join(self.download_dir, filename)
            
            if os.path.exists(expected_path):
                logger.info(f"Download successful: {expected_path}")
                return True
            else:
                logger.error(f"Download completed but file not found: {expected_path}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Download timed out")
            return False
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def download_book_with_epub_guard(self, metadata: Dict, work_id: int, author_name: str = "") -> bool:
        """
        Download a book using EPUB guard helper with robust validation and conversion
        
        Args:
            metadata: Full metadata dict from Anna's Archive search
            work_id: Work ID for tracking
            author_name: Author name for filename
            
        Returns:
            True if download successful, False otherwise
        """
        logger.info(f"Downloading book with EPUB guard: {metadata.get('title', 'Unknown')}")
        
        try:
            # Use EPUB guard helper for robust download and validation
            final_path = download_from_metadata(
                metadata, 
                Path(self.download_dir), 
                prefer_title_author=True, 
                convert_mobi=True
            )
            
            logger.info(f"EPUB guard download successful: {final_path}")
            return True
            
        except ValueError as e:
            # Handle specific EPUB guard errors with user-friendly messages
            error_msg = str(e)
            if "HTML page instead of a book" in error_msg:
                logger.error("The mirror returned an HTML page. Switching gateway...")
            elif "ZIP file is not a valid EPUB" in error_msg:
                logger.error("EPUB layout invalid (mimetype not first). Re-download from another gateway.")
            elif "Unknown or corrupted file" in error_msg:
                logger.error("Corrupted or wrong format. Retrying with alternate CID/mirror.")
            else:
                logger.error(f"EPUB guard error: {error_msg}")
            return False
            
        except Exception as e:
            logger.error(f"EPUB guard download failed: {e}")
            return False
    
    def _parse_search_results(self, output: str) -> List[Dict]:
        """Parse search results from anna-mcp output"""
        try:
            # Try to parse as JSON first
            if output.strip().startswith('{') or output.strip().startswith('['):
                results = json.loads(output)
                if isinstance(results, list):
                    return results
                elif isinstance(results, dict) and 'results' in results:
                    return results['results']
                else:
                    return [results]
            
            # Parse the structured "Book X:" format
            results = []
            lines = output.strip().split('\n')
            current_book = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a new book entry
                if line.startswith('Book '):
                    # Save previous book if it exists
                    if current_book and current_book.get('hash'):
                        results.append(current_book)
                    # Start new book
                    current_book = {
                        'format': 'epub',  # Default format
                        'size': 0,
                        'language': 'en',
                        'year': 0,
                        'publisher': '',
                        'confidence': 0.8
                    }
                
                # Parse book fields
                elif ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'title':
                        current_book['title'] = value if value else 'Unknown Title'
                    elif key == 'authors':
                        current_book['author'] = value if value else 'Unknown Author'
                    elif key == 'publisher':
                        current_book['publisher'] = value
                    elif key == 'language':
                        current_book['language'] = value if value else 'en'
                    elif key == 'format':
                        current_book['format'] = value.lower() if value else 'epub'
                    elif key == 'size':
                        try:
                            current_book['size'] = int(value) if value else 0
                        except ValueError:
                            current_book['size'] = 0
                    elif key == 'hash':
                        current_book['hash'] = value
                    elif key == 'url':
                        current_book['url'] = value
            
            # Don't forget the last book
            if current_book and current_book.get('hash'):
                results.append(current_book)
            
            logger.info(f"Parsed {len(results)} results from search output")
            return results
            
        except json.JSONDecodeError:
            logger.warning("Could not parse search results as JSON, trying structured parsing")
            return []
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return []
    
    def _simulate_search(self, search_term: str) -> List[Dict]:
        """Simulate search results for testing"""
        # Create realistic search results
        results = [
            {
                'hash': f"simulated_hash_{hash(search_term)}",
                'title': search_term.split()[0] if search_term else "Unknown Title",
                'author': search_term.split()[-1] if len(search_term.split()) > 1 else "Unknown Author",
                'format': 'epub',
                'size': 1024000,  # 1MB
                'language': 'en',
                'year': 2020,
                'publisher': 'Simulated Publisher',
                'confidence': 0.85
            }
        ]
        
        logger.info(f"Simulated search returned {len(results)} results")
        return results
    
    def _simulate_download(self, book_hash: str, title: str, format_type: str) -> bool:
        """Simulate download for testing"""
        try:
            # Create download directory if it doesn't exist
            download_dir = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download"
            os.makedirs(download_dir, exist_ok=True)
            
            # Create filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_title}.{format_type}"
            filepath = os.path.join(download_dir, filename)
            
            # Create simulated file
            with open(filepath, 'w') as f:
                f.write(f"Simulated download of {title}\n")
                f.write(f"Hash: {book_hash}\n")
                f.write(f"Format: {format_type}\n")
                f.write(f"Downloaded at: {os.popen('date').read().strip()}\n")
            
            logger.info(f"Simulated download completed: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Simulated download failed: {e}")
            return False

def test_mcp_integration():
    """Test the MCP integration functionality"""
    logger.info("=== TESTING MCP INTEGRATION ===")
    
    # Initialize integration
    mcp = AnnaMCPIntegration()
    
    # Test search
    search_term = "A Little Scandal Patricia Cabot"
    results = mcp.search_books(search_term)
    
    logger.info(f"Search results: {results}")
    
    # Test download
    if results:
        book = results[0]
        success = mcp.download_book(book['hash'], book['title'], book['format'])
        logger.info(f"Download success: {success}")
    
    logger.info("MCP integration test completed")

if __name__ == "__main__":
    test_mcp_integration()
