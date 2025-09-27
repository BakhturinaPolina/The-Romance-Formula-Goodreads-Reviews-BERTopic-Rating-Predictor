"""
Anna's Archive API Client

This module provides a client for interacting with Anna's Archive API
to search for and download books.

Note: Requires an API key from Anna's Archive (available through donation)
"""

import requests
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

@dataclass
class BookResult:
    """Represents a book found in Anna's Archive"""
    id: str
    title: str
    author: str
    year: Optional[int]
    formats: List[str]
    download_url: Optional[str] = None
    size: Optional[int] = None
    quality: Optional[str] = None

class AnnasArchiveClient:
    """
    Client for Anna's Archive API.

    Handles authentication, searching, and book retrieval.
    """

    def __init__(self, api_key: str, base_url: str = "https://annas-archive.org/api"):
        """
        Initialize the Anna's Archive client.

        Args:
            api_key: API key from Anna's Archive (required)
            base_url: Base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'User-Agent': 'RomanceNovelResearch/1.0'
        })

        self.logger = logging.getLogger(__name__)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds

    def _rate_limit(self):
        """Implement basic rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make an API request with error handling"""
        self.logger.debug(f"Making API request to endpoint: {endpoint}")
        self.logger.debug(f"Request params: {params}")
        
        self._rate_limit()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        self.logger.debug(f"Full request URL: {url}")

        try:
            response = self.session.get(url, params=params, timeout=30)
            self.logger.debug(f"Response status: {response.status_code}")
            self.logger.debug(f"Response headers: {dict(response.headers)}")

            if response.status_code == 200:
                data = response.json()
                self.logger.debug(f"Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                return data
            elif response.status_code == 401:
                self.logger.error("Invalid API key")
                return None
            elif response.status_code == 429:
                self.logger.warning("Rate limited, backing off...")
                time.sleep(5)
                return self._make_request(endpoint, params)
            else:
                self.logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {e}")
            return None

    def search_books(self, title: str, author: str, year: Optional[int] = None) -> List[BookResult]:
        """
        Search for books in Anna's Archive.

        Args:
            title: Book title to search for
            author: Book author to search for
            year: Publication year (optional)

        Returns:
            List of matching BookResult objects
        """
        self.logger.debug(f"Searching for book: '{title}' by '{author}' (year: {year})")
        
        params = {
            'title': title,
            'author': author,
        }

        if year:
            params['year'] = year

        # Add limit for performance
        params['limit'] = 10

        data = self._make_request('/search', params)

        if not data or 'results' not in data:
            self.logger.debug("No search results found or invalid response structure")
            return []

        results = []
        for i, item in enumerate(data['results']):
            self.logger.debug(f"Processing search result {i+1}: {item.get('title', 'Unknown')}")
            # Parse the result based on expected structure
            book = BookResult(
                id=item.get('id', ''),
                title=item.get('title', ''),
                author=item.get('author', ''),
                year=item.get('year'),
                formats=item.get('formats', []),
                download_url=item.get('download_url'),
                size=item.get('size'),
                quality=item.get('quality')
            )
            results.append(book)

        self.logger.debug(f"Found {len(results)} search results")
        return results

    def get_book_details(self, book_id: str) -> Optional[BookResult]:
        """
        Get detailed information about a specific book.

        Args:
            book_id: Anna's Archive book ID

        Returns:
            BookResult object or None if not found
        """
        data = self._make_request(f'/books/{book_id}')

        if not data:
            return None

        return BookResult(
            id=data.get('id', book_id),
            title=data.get('title', ''),
            author=data.get('author', ''),
            year=data.get('year'),
            formats=data.get('formats', []),
            download_url=data.get('download_url'),
            size=data.get('size'),
            quality=data.get('quality')
        )

    def download_book(self, book_id: str, format_type: str) -> Optional[bytes]:
        """
        Download a book in the specified format.

        Args:
            book_id: Anna's Archive book ID
            format_type: Format to download (epub, pdf, etc.)

        Returns:
            Book content as bytes or None if download fails
        """
        self.logger.debug(f"Attempting to download book {book_id} in format {format_type}")
        
        # Get book details first to find download URL
        book = self.get_book_details(book_id)
        if not book or not book.download_url:
            self.logger.error(f"No download URL found for book {book_id}")
            return None

        self.logger.debug(f"Download URL: {book.download_url}")

        try:
            self._rate_limit()
            response = self.session.get(book.download_url, timeout=60, stream=True)
            self.logger.debug(f"Download response status: {response.status_code}")

            if response.status_code == 200:
                # For large files, we might want to stream
                content = response.content
                self.logger.info(f"Downloaded {len(content)} bytes for book {book_id}")
                self.logger.debug(f"Content type: {response.headers.get('content-type', 'unknown')}")
                return content
            else:
                self.logger.error(f"Download failed: {response.status_code}")
                self.logger.debug(f"Response text: {response.text[:200]}...")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Download error: {e}")
            return None

    def validate_api_key(self) -> bool:
        """
        Validate that the API key is working.

        Returns:
            True if API key is valid, False otherwise
        """
        # Try a simple search to validate the key
        try:
            results = self.search_books("test", "test", limit=1)
            # If we get a response (even if no results), the key is likely valid
            return True
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False
