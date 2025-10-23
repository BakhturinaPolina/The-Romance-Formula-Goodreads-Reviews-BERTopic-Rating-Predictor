#!/usr/bin/env python3
"""
API Downloader for Anna's Archive

Downloads books using Anna's Archive fast download API with MD5 hashes.
Provides authentication and direct download functionality.

Usage:
    from api_downloader import AnnaArchiveDownloader
    
    downloader = AnnaArchiveDownloader(api_key="your_api_key")
    success = downloader.download_book("md5_hash", "output_directory")
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnnaArchiveDownloader:
    """Downloader for Anna's Archive using the fast download API."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://annas-archive.org",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the downloader.
        
        Args:
            api_key: Anna's Archive API key
            base_url: Base URL for Anna's Archive API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # API endpoints
        self.fast_download_url = f"{self.base_url}/dyn/api/fast_download.json"
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AnnaArchiveDownloader/1.0',
            'Accept': 'application/json'
        })
        
        logger.info(f"Initialized downloader for {self.base_url}")
    
    def get_download_url(self, md5: str) -> Optional[str]:
        """
        Get direct download URL for a book using its MD5 hash.
        
        Args:
            md5: MD5 hash of the book file
            
        Returns:
            Direct download URL or None if failed
        """
        params = {'md5': md5}
        
        # Try different authentication methods
        auth_methods = [
            {'Authorization': f'Bearer {self.api_key}'},
            {'X-API-Key': self.api_key},
            {'aa_session': self.api_key}
        ]
        
        for attempt in range(self.max_retries):
            for auth_header in auth_methods:
                try:
                    headers = self.session.headers.copy()
                    headers.update(auth_header)
                    
                    logger.debug(f"Requesting download URL for MD5: {md5}")
                    response = self.session.get(
                        self.fast_download_url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract download URL from response
                        download_url = self._extract_download_url(data)
                        if download_url:
                            logger.info(f"Got download URL for {md5}")
                            return download_url
                        else:
                            logger.warning(f"No download URL in response for {md5}")
                            logger.debug(f"Response: {data}")
                    
                    elif response.status_code == 401:
                        logger.warning(f"Authentication failed for {md5}")
                        continue  # Try next auth method
                    
                    elif response.status_code == 404:
                        logger.warning(f"Book not found: {md5}")
                        return None
                    
                    else:
                        logger.warning(f"HTTP {response.status_code} for {md5}: {response.text}")
                
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request failed for {md5} (attempt {attempt + 1}): {e}")
                
                # Small delay between attempts
                time.sleep(1)
        
        logger.error(f"Failed to get download URL for {md5} after {self.max_retries} attempts")
        return None
    
    def _extract_download_url(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Extract download URL from API response."""
        # Try different possible field names
        url_fields = ['url', 'download_url', 'direct_url', 'file_url', 'link']
        
        for field in url_fields:
            if field in response_data:
                url = response_data[field]
                if isinstance(url, str) and url.startswith('http'):
                    return url
        
        # If no direct URL field, look for nested structures
        if 'data' in response_data:
            return self._extract_download_url(response_data['data'])
        
        if 'result' in response_data:
            return self._extract_download_url(response_data['result'])
        
        return None
    
    def download_file(
        self,
        download_url: str,
        output_path: Path,
        chunk_size: int = 8192
    ) -> bool:
        """
        Download a file from URL to local path.
        
        Args:
            download_url: URL to download from
            output_path: Local path to save file
            chunk_size: Size of chunks for streaming download
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            logger.info(f"Downloading to {output_path}")
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get file size for progress bar
            head_response = self.session.head(download_url, timeout=self.timeout)
            total_size = int(head_response.headers.get('content-length', 0))
            
            # Download with progress bar
            with self.session.get(download_url, stream=True, timeout=self.timeout) as response:
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    if total_size > 0:
                        with tqdm(
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            desc=output_path.name
                        ) as pbar:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        # No content-length header, download without progress bar
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
            
            # Verify download
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Download complete: {output_path} ({output_path.stat().st_size:,} bytes)")
                return True
            else:
                logger.error(f"Download failed: {output_path} is empty or missing")
                return False
                
        except Exception as e:
            logger.error(f"Download failed for {download_url}: {e}")
            # Clean up partial file
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_book(
        self,
        md5: str,
        output_dir: str,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Download a book by its MD5 hash.
        
        Args:
            md5: MD5 hash of the book
            output_dir: Directory to save the file
            filename: Optional custom filename
            
        Returns:
            Path to downloaded file or None if failed
        """
        logger.info(f"Downloading book with MD5: {md5}")
        
        # Get download URL
        download_url = self.get_download_url(md5)
        if not download_url:
            return None
        
        # Determine output filename
        if not filename:
            # Extract filename from URL
            parsed_url = urlparse(download_url)
            filename = os.path.basename(parsed_url.path)
            
            # Fallback to MD5 if no filename in URL
            if not filename or '.' not in filename:
                filename = f"{md5}.epub"  # Default to EPUB
        
        # Create output path
        output_path = Path(output_dir) / filename
        
        # Download file
        if self.download_file(download_url, output_path):
            return output_path
        else:
            return None
    
    def download_books(
        self,
        md5_list: List[str],
        output_dir: str,
        delay: float = 1.0
    ) -> Dict[str, Optional[Path]]:
        """
        Download multiple books.
        
        Args:
            md5_list: List of MD5 hashes
            output_dir: Directory to save files
            delay: Delay between downloads in seconds
            
        Returns:
            Dictionary mapping MD5 to download path (or None if failed)
        """
        results = {}
        
        logger.info(f"Downloading {len(md5_list)} books to {output_dir}")
        
        for i, md5 in enumerate(md5_list, 1):
            logger.info(f"Downloading {i}/{len(md5_list)}: {md5}")
            
            result = self.download_book(md5, output_dir)
            results[md5] = result
            
            # Delay between downloads
            if i < len(md5_list) and delay > 0:
                time.sleep(delay)
        
        # Summary
        successful = sum(1 for path in results.values() if path is not None)
        logger.info(f"Download complete: {successful}/{len(md5_list)} successful")
        
        return results
    
    def close(self) -> None:
        """Close the session."""
        if self.session:
            self.session.close()


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Download books from Anna's Archive using MD5 hashes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a single book
  python api_downloader.py \\
    --api-key "your_api_key" \\
    --md5 "8336332bf5877e3adbfb60ac70720cd5" \\
    --output-dir ../../organized_outputs/epub_downloads/

  # Download multiple books from CSV
  python api_downloader.py \\
    --api-key "your_api_key" \\
    --csv-file results.csv \\
    --output-dir ../../organized_outputs/epub_downloads/ \\
    --delay 2.0

  # Test API connection
  python api_downloader.py \\
    --api-key "your_api_key" \\
    --test
        """
    )
    
    parser.add_argument(
        '--api-key',
        required=True,
        help='Anna\'s Archive API key'
    )
    
    parser.add_argument(
        '--md5',
        help='MD5 hash of book to download'
    )
    
    parser.add_argument(
        '--csv-file',
        help='CSV file with MD5 hashes (must have "md5" column)'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for downloaded files'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between downloads in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test API connection without downloading'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        downloader = AnnaArchiveDownloader(
            api_key=args.api_key,
            timeout=args.timeout
        )
        
        if args.test:
            print("🧪 Testing API connection...")
            # Try to get download URL for a test MD5
            test_md5 = "test123"
            url = downloader.get_download_url(test_md5)
            if url:
                print("✅ API connection successful")
            else:
                print("❌ API connection failed")
            return 0
        
        if args.md5:
            print(f"📥 Downloading book with MD5: {args.md5}")
            result = downloader.download_book(args.md5, args.output_dir)
            
            if result:
                print(f"✅ Download successful: {result}")
                return 0
            else:
                print("❌ Download failed")
                return 1
        
        elif args.csv_file:
            import pandas as pd
            
            print(f"📥 Downloading books from CSV: {args.csv_file}")
            
            # Read CSV file
            df = pd.read_csv(args.csv_file)
            if 'md5' not in df.columns:
                print("❌ Error: CSV file must have 'md5' column")
                return 1
            
            md5_list = df['md5'].dropna().tolist()
            print(f"Found {len(md5_list)} MD5 hashes to download")
            
            # Download books
            results = downloader.download_books(
                md5_list,
                args.output_dir,
                delay=args.delay
            )
            
            # Summary
            successful = sum(1 for path in results.values() if path is not None)
            print(f"✅ Download complete: {successful}/{len(md5_list)} successful")
            
            return 0 if successful > 0 else 1
        
        else:
            print("❌ No download target specified. Use --md5, --csv-file, or --test")
            return 1
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        if 'downloader' in locals():
            downloader.close()


if __name__ == "__main__":
    exit(main())
