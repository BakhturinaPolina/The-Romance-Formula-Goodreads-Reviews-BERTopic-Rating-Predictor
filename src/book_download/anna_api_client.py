#!/usr/bin/env python3
"""
Anna's Archive API Client
Robust client for the official Anna's Archive JSON API with mirror rotation and proper error handling
"""

import os
import re
import json
import logging
import requests
import subprocess
import shutil
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnnaAPIClient:
    """Robust client for Anna's Archive official JSON API"""
    
    # Current working mirrors
    MIRRORS = [
        "https://annas-archive.se",
        "https://annas-archive.li", 
        "https://annas-archive.org",
    ]
    
    # Regex to extract MD5 from URLs or strings
    MD5_RE = re.compile(r"(?:^|/)(?:md5/)?([a-fA-F0-9]{32})(?:[/?#]|$)")
    
    def __init__(self, api_key: Optional[str] = None, verify_tls: bool = True, timeout: int = 20, use_tor: bool = True):
        """
        Initialize the Anna's Archive API client
        
        Args:
            api_key: Member secret API key (or will use ANNAS_SECRET_KEY env var)
            verify_tls: Whether to verify TLS certificates
            timeout: HTTP timeout in seconds
            use_tor: Whether to use Tor for requests (required for Anna's Archive)
        """
        self.api_key = api_key or os.getenv('ANNAS_SECRET_KEY')
        self.verify_tls = verify_tls
        self.timeout = timeout
        self.use_tor = use_tor
        
        if not self.api_key:
            logger.warning("No API key provided - set ANNAS_SECRET_KEY environment variable")
        
        # Check for torsocks availability
        self.torsocks_available = shutil.which("torsocks") is not None
        if self.use_tor and not self.torsocks_available:
            logger.warning("torsocks not found - Anna's Archive requires Tor access")
        
        # Standard headers for API requests
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "anna-api-client/1.0",
        }
        
        logger.info(f"Anna API Client initialized with {len(self.MIRRORS)} mirrors, Tor: {self.use_tor and self.torsocks_available}")
    
    def extract_md5(self, input_str: str) -> Optional[str]:
        """
        Extract MD5 hash from various input formats
        
        Args:
            input_str: Anna's Archive URL, MD5 hash, or string containing MD5
            
        Returns:
            Lowercased MD5 hash or None if not found
        """
        input_str = input_str.strip()
        
        # Check if it's a bare 32-char hex MD5
        if re.fullmatch(r"[a-fA-F0-9]{32}", input_str):
            return input_str.lower()
        
        # Try to extract from URL
        try:
            parsed = urlparse(input_str)
            hay = parsed.path or ""
            match = self.MD5_RE.search(hay)
            if match:
                return match.group(1).lower()
        except Exception:
            pass
        
        # Search the whole string as fallback
        match = self.MD5_RE.search(input_str)
        return match.group(1).lower() if match else None
    
    def _make_tor_request(self, url: str, params: Dict = None, headers: Dict = None) -> requests.Response:
        """
        Make a request through Tor using torsocks
        """
        if not self.use_tor or not self.torsocks_available:
            # Fallback to regular requests
            return requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_tls
            )
        
        # Build curl command with torsocks
        cmd = ["torsocks", "curl", "-s", "-L"]
        
        # Add headers
        if headers:
            for key, value in headers.items():
                cmd.extend(["-H", f"{key}: {value}"])
        
        # Add parameters
        if params:
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url_with_params = f"{url}?{param_str}"
        else:
            url_with_params = url
        
        cmd.append(url_with_params)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            
            # Create a mock response object
            response = requests.Response()
            response.status_code = 200 if result.returncode == 0 else 500
            response._content = result.stdout.encode('utf-8')
            response.headers = {'content-type': 'application/json'}
            response.url = url_with_params
            
            return response
            
        except subprocess.TimeoutExpired:
            response = requests.Response()
            response.status_code = 408  # Request timeout
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
    
    def get_fast_download(self, md5_or_url: str) -> Dict:
        """
        Get fast download URL for a book using its MD5 hash
        
        Args:
            md5_or_url: MD5 hash or Anna's Archive URL containing MD5
            
        Returns:
            Dictionary with download info or error details
        """
        if not self.api_key:
            return {
                "error": True,
                "message": "No API key provided - set ANNAS_SECRET_KEY environment variable"
            }
        
        # Extract MD5 from input
        md5 = self.extract_md5(md5_or_url)
        if not md5:
            return {
                "error": True,
                "message": f"Could not extract MD5 from input: {md5_or_url}"
            }
        
        logger.info(f"Requesting fast download for MD5: {md5}")
        
        # Try each mirror until we get valid JSON
        params = {"md5": md5, "key": self.api_key}
        errors = []
        
        for mirror in self.MIRRORS:
            url = f"{mirror}/dyn/api/fast_download.json"
            
            try:
                logger.debug(f"Trying mirror: {mirror}")
                response = self._make_tor_request(url, params, self.headers)
                
                # Try to parse as JSON (robust detection)
                try:
                    data = response.json()
                    logger.info(f"Successfully got JSON response from {mirror}")
                    return {
                        "success": True,
                        "domain": mirror,
                        "status_code": response.status_code,
                        "data": data,
                        "md5": md5
                    }
                except (json.JSONDecodeError, ValueError):
                    # Not JSON - capture sample for debugging
                    sample = (response.text or "")[:300].replace("\n", " ").replace("\r", " ")
                    errors.append({
                        "mirror": mirror,
                        "status_code": response.status_code,
                        "content_type": response.headers.get("Content-Type", ""),
                        "sample": sample
                    })
                    logger.warning(f"Non-JSON response from {mirror}: {response.status_code} - {sample[:100]}...")
                    
            except requests.exceptions.RequestException as e:
                errors.append({
                    "mirror": mirror,
                    "error": f"Request error: {e}"
                })
                logger.warning(f"Request failed for {mirror}: {e}")
        
        # All mirrors failed
        return {
            "error": True,
            "message": "All mirrors failed",
            "md5": md5,
            "errors": errors
        }
    
    def download_book(self, md5_or_url: str, filename: Optional[str] = None, download_dir: str = None) -> Dict:
        """
        Download a book using its MD5 hash or URL
        
        Args:
            md5_or_url: MD5 hash or Anna's Archive URL
            filename: Optional filename (will be extracted from download URL if not provided)
            download_dir: Directory to save the file (defaults to organized_outputs/anna_archive_download)
            
        Returns:
            Dictionary with download result
        """
        if download_dir is None:
            download_dir = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download"
        
        # Ensure download directory exists
        os.makedirs(download_dir, exist_ok=True)
        
        # Get fast download URL
        download_info = self.get_fast_download(md5_or_url)
        if download_info.get("error"):
            return download_info
        
        # Extract download URL from response
        data = download_info["data"]
        download_url = data.get("download_url")
        
        if not download_url:
            return {
                "error": True,
                "message": "No download URL in API response",
                "response": data
            }
        
        # Determine filename
        if not filename:
            # Extract filename from URL
            from urllib.parse import unquote
            filename = download_url.split('/')[-1]
            filename = unquote(filename)
        
        # Create full file path
        filepath = os.path.join(download_dir, filename)
        
        try:
            logger.info(f"Downloading to: {filepath}")
            
            # Download the file through Tor
            if self.use_tor and self.torsocks_available:
                # Use torsocks for download
                cmd = ["torsocks", "curl", "-L", "-o", filepath, download_url]
                result = subprocess.run(cmd, timeout=self.timeout * 3)
                
                if result.returncode == 0 and os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    logger.info(f"Download completed: {filepath} ({file_size} bytes)")
                    return {
                        "success": True,
                        "filepath": filepath,
                        "filename": filename,
                        "file_size": file_size,
                        "download_url": download_url,
                        "md5": download_info["md5"]
                    }
                else:
                    return {
                        "error": True,
                        "message": f"Download failed with return code: {result.returncode}",
                        "download_url": download_url,
                        "md5": download_info["md5"]
                    }
            else:
                # Fallback to regular requests
                response = requests.get(
                    download_url,
                    headers=self.headers,
                    timeout=self.timeout * 3,  # Longer timeout for downloads
                    verify=self.verify_tls,
                    stream=True
                )
                response.raise_for_status()
                
                # Save file
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Get file size
                file_size = os.path.getsize(filepath)
                
                logger.info(f"Download completed: {filepath} ({file_size} bytes)")
                
                return {
                    "success": True,
                    "filepath": filepath,
                    "filename": filename,
                    "file_size": file_size,
                    "download_url": download_url,
                    "md5": download_info["md5"]
                }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"Download failed: {e}",
                "download_url": download_url,
                "md5": download_info["md5"]
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"File save failed: {e}",
                "download_url": download_url,
                "md5": download_info["md5"]
            }
    
    def check_account_info(self, test_md5: str = "00000000000000000000000000000000") -> Dict:
        """
        Check account information by making a test request
        
        Args:
            test_md5: MD5 to use for testing (defaults to dummy MD5)
        
        Returns:
            Account info or error details
        """
        result = self.get_fast_download(test_md5)
        
        if result.get("success"):
            data = result["data"]
            account_info = data.get("account_fast_download_info", {})
            return {
                "success": True,
                "account_info": account_info,
                "response": data
            }
        else:
            return {
                "error": True,
                "message": "Could not retrieve account information",
                "details": result
            }

    def _normalize_text(self, text: str) -> str:
        """Return *text* in lowercase with non-alnum replaced by spaces."""
        import re
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def search_md5(self, title: str, author: str | None = None, prefer_exts: tuple[str, ...] = ("epub",)) -> Optional[str]:
        """Search Anna's Archive member API for *title*/*author* and return a preferred MD5.

        The method queries the same mirrors used by :py:meth:`get_fast_download` but hits
        the member search endpoint (``/dyn/api/search.json``).  The response schema is a
        list of result objects that include *md5*, *extension*, *title*, and *author*.

        Selection heuristic (best-to-worst):
        1. extension in *prefer_exts* (earlier items have higher priority)
        2. exact (normalized) title match;
        3. author substring match (if *author* given);
        4. highest reported relevance score (if ``score`` field present).
        """
        if not self.api_key:
            logger.error("Member secret (ANNAS_SECRET_KEY) not configured – cannot search.")
            return None

        search_term = f"{title} {author}".strip()
        params = {"query": search_term, "key": self.api_key, "page": 1, "type": "books"}
        errors = []

        norm_title = self._normalize_text(title)
        norm_author = self._normalize_text(author) if author else None

        best_candidate: dict | None = None
        best_score = -1  # higher is better

        for mirror in self.MIRRORS:
            url = f"{mirror}/dyn/api/search.json"
            try:
                response = self._make_tor_request(url, params, self.headers)
                data = response.json()
                results = data.get("results") or data  # some versions return list directly
                if not isinstance(results, list):
                    logger.debug("Unexpected search payload from %s", mirror)
                    continue

                for res in results:
                    md5 = res.get("md5")
                    ext = (res.get("extension") or "").lower()
                    title_res = res.get("title", "")
                    author_res = res.get("author", "")
                    score_res = res.get("score", 0)

                    if not md5 or len(md5) != 32:
                        continue

                    # heuristic score
                    score = score_res
                    # preferred extension bonus
                    if ext in prefer_exts:
                        score += 30 + (len(prefer_exts) - prefer_exts.index(ext))
                    # exact title match bonus
                    if self._normalize_text(title_res) == norm_title:
                        score += 50
                    # author substring bonus
                    if norm_author and norm_author in self._normalize_text(author_res):
                        score += 10

                    if score > best_score:
                        best_score = score
                        best_candidate = res

                # if we already found a very good match, break early
                if best_score >= 90:
                    break
            except Exception as exc:  # pylint: disable=broad-except
                errors.append({"mirror": mirror, "error": str(exc)})
                logger.debug("Search request failed for %s: %s", mirror, exc)

        if best_candidate:
            logger.info(
                "Selected MD5 %s for '%s' by '%s' (ext=%s, score=%.1f)",
                best_candidate.get("md5"),
                title,
                author,
                best_candidate.get("extension"),
                best_score,
            )
            return best_candidate.get("md5")

        logger.warning("No MD5 found for '%s' by '%s'", title, author)
        if errors:
            logger.debug("Search errors: %s", errors)
        return None

def main():
    """Test the Anna API client"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Anna's Archive API client"
    )
    parser.add_argument("input", help="MD5 hash or Anna's Archive URL")
    parser.add_argument("--download", action="store_true", help="Download the file")
    parser.add_argument("--filename", help="Filename for download")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification")
    parser.add_argument("--check-account", action="store_true", help="Check account info only")
    
    args = parser.parse_args()
    
    # Initialize client
    client = AnnaAPIClient(verify_tls=not args.insecure)
    
    if args.check_account:
        # Check account info
        result = client.check_account_info()
        print(json.dumps(result, indent=2))
        return
    
    if args.download:
        # Download the file
        result = client.download_book(args.input, args.filename)
    else:
        # Just get download info
        result = client.get_fast_download(args.input)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
