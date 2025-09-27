"""
Anna's Archive MCP Client Integration

This module integrates with the annas-mcp tool for targeted book searches
and downloads without requiring API keys.

Based on: https://github.com/iosifache/annas-mcp
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import tempfile
import os

@dataclass
class MCPBookResult:
    """Result from annas-mcp search"""
    id: str
    title: str
    author: str
    year: Optional[int]
    format: str
    size: Optional[int]
    download_url: Optional[str] = None

class AnnasMCPClient:
    """
    Client for using the annas-mcp tool for Anna's Archive access.

    This provides a Python interface to the annas-mcp CLI tool,
    allowing for targeted searches and downloads without API keys.
    """

    def __init__(self, mcp_binary_path: Optional[Path] = None, 
                 download_path: Optional[Path] = None):
        """
        Initialize the MCP client.

        Args:
            mcp_binary_path: Path to the annas-mcp binary (optional)
            download_path: Path for downloads (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Find the annas-mcp binary
        self.mcp_binary = self._find_mcp_binary(mcp_binary_path)
        
        # Set download path
        self.download_path = download_path or Path(tempfile.gettempdir()) / 'annas_downloads'
        self.download_path.mkdir(parents=True, exist_ok=True)

        # Set environment variables
        os.environ['ANNAS_DOWNLOAD_PATH'] = str(self.download_path)

    def _find_mcp_binary(self, custom_path: Optional[Path] = None) -> Optional[Path]:
        """Find the annas-mcp binary"""
        if custom_path and custom_path.exists():
            return custom_path

        # Try common locations
        possible_paths = [
            Path('./annas-mcp'),
            Path('./annas-mcp.exe'),
            Path('/usr/local/bin/annas-mcp'),
            Path('/usr/bin/annas-mcp'),
            Path.home() / 'bin' / 'annas-mcp'
        ]

        for path in possible_paths:
            if path.exists():
                self.logger.info(f"Found annas-mcp binary: {path}")
                return path

        # Try to find in PATH
        try:
            result = subprocess.run(['which', 'annas-mcp'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                binary_path = Path(result.stdout.strip())
                self.logger.info(f"Found annas-mcp in PATH: {binary_path}")
                return binary_path
        except FileNotFoundError:
            pass

        self.logger.warning("annas-mcp binary not found. Please install it first.")
        return None

    def _run_mcp_command(self, command: List[str]) -> Dict:
        """
        Run an annas-mcp command and return the result.

        Args:
            command: List of command arguments

        Returns:
            Dictionary with command result
        """
        if not self.mcp_binary:
            return {'error': 'annas-mcp binary not found'}

        try:
            full_command = [str(self.mcp_binary)] + command
            self.logger.debug(f"Running command: {' '.join(full_command)}")

            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {'output': result.stdout, 'error': None}
            else:
                return {
                    'error': result.stderr,
                    'output': result.stdout,
                    'returncode': result.returncode
                }

        except subprocess.TimeoutExpired:
            return {'error': 'Command timed out'}
        except Exception as e:
            return {'error': str(e)}

    def search_books(self, query: str, limit: int = 10) -> List[MCPBookResult]:
        """
        Search for books using annas-mcp.

        Args:
            query: Search query (title, author, or general terms)
            limit: Maximum number of results

        Returns:
            List of MCPBookResult objects
        """
        if not self.mcp_binary:
            self.logger.error("annas-mcp binary not available")
            return []

        # Run search command
        result = self._run_mcp_command(['search', query, '--limit', str(limit)])

        if 'error' in result:
            self.logger.error(f"Search failed: {result['error']}")
            return []

        # Parse results
        books = []
        if 'results' in result:
            for item in result['results']:
                book = MCPBookResult(
                    id=item.get('id', ''),
                    title=item.get('title', ''),
                    author=item.get('author', ''),
                    year=item.get('year'),
                    format=item.get('format', ''),
                    size=item.get('size'),
                    download_url=item.get('download_url')
                )
                books.append(book)

        self.logger.info(f"Found {len(books)} books for query: {query}")
        return books

    def download_book(self, book_id: str, format_preference: str = 'epub') -> Optional[Path]:
        """
        Download a book using annas-mcp.

        Args:
            book_id: ID of the book to download
            format_preference: Preferred format (epub, pdf, etc.)

        Returns:
            Path to downloaded file or None if failed
        """
        if not self.mcp_binary:
            self.logger.error("annas-mcp binary not available")
            return None

        # Run download command
        result = self._run_mcp_command(['download', book_id, '--format', format_preference])

        if 'error' in result:
            self.logger.error(f"Download failed: {result['error']}")
            return None

        # Find the downloaded file
        if 'downloaded_file' in result:
            file_path = Path(result['downloaded_file'])
            if file_path.exists():
                self.logger.info(f"Downloaded: {file_path}")
                return file_path

        # Try to find the file in download directory
        for file_path in self.download_path.glob('*'):
            if file_path.is_file():
                self.logger.info(f"Found downloaded file: {file_path}")
                return file_path

        self.logger.warning(f"No downloaded file found for book ID: {book_id}")
        return None

    def get_book_info(self, book_id: str) -> Optional[MCPBookResult]:
        """
        Get detailed information about a specific book.

        Args:
            book_id: ID of the book

        Returns:
            MCPBookResult object or None if not found
        """
        if not self.mcp_binary:
            return None

        result = self._run_mcp_command(['info', book_id])

        if 'error' in result:
            self.logger.error(f"Failed to get book info: {result['error']}")
            return None

        if 'book' in result:
            book_data = result['book']
            return MCPBookResult(
                id=book_data.get('id', book_id),
                title=book_data.get('title', ''),
                author=book_data.get('author', ''),
                year=book_data.get('year'),
                format=book_data.get('format', ''),
                size=book_data.get('size'),
                download_url=book_data.get('download_url')
            )

        return None

    def install_mcp_tool(self, install_dir: Optional[Path] = None) -> bool:
        """
        Download and install the annas-mcp tool.

        Args:
            install_dir: Directory to install the tool (optional)

        Returns:
            True if successful, False otherwise
        """
        if install_dir is None:
            install_dir = Path.home() / 'bin'
        
        install_dir.mkdir(parents=True, exist_ok=True)

        # This is a placeholder for the actual installation process
        # In a real implementation, you would:
        # 1. Download the appropriate binary for the user's system
        # 2. Make it executable
        # 3. Place it in the install directory

        self.logger.info(f"Installation would place annas-mcp in: {install_dir}")
        self.logger.info("Please download annas-mcp manually from: https://github.com/iosifache/annas-mcp/releases")
        
        return False

    def validate_installation(self) -> bool:
        """Validate that annas-mcp is properly installed and working"""
        if not self.mcp_binary:
            return False

        # Try to run a simple command
        result = self._run_mcp_command(['--version'])
        
        if 'error' not in result:
            self.logger.info("annas-mcp installation validated")
            return True
        else:
            self.logger.error(f"annas-mcp validation failed: {result['error']}")
            return False

    def get_available_formats(self, book_id: str) -> List[str]:
        """
        Get available formats for a specific book.

        Args:
            book_id: ID of the book

        Returns:
            List of available formats
        """
        book_info = self.get_book_info(book_id)
        if book_info and book_info.format:
            return [book_info.format]
        
        # Try common formats
        common_formats = ['epub', 'pdf', 'mobi', 'txt', 'html']
        available_formats = []

        for format_type in common_formats:
            # This would check if the format is available
            # For now, we'll return a mock list
            available_formats.append(format_type)

        return available_formats
