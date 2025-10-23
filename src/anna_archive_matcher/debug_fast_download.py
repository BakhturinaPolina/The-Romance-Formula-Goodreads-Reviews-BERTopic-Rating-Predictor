#!/usr/bin/env python3
"""
Debug the fast download endpoint to see the response format
"""

import sys
import logging
import json
from pathlib import Path

# Add the book_download directory to the path
sys.path.append(str((Path(__file__).parent.parent / "book_download").resolve()))

from anna_api_client import AnnaAPIClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_fast_download")

# Your API key
API_KEY = "BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP"

def debug_fast_download():
    """Debug the fast download endpoint"""
    
    logger.info("Debug Anna's Archive Fast Download")
    logger.info("=" * 40)
    
    # Initialize the API client
    client = AnnaAPIClient(
        api_key=API_KEY,
        use_tor=True,
        timeout=30
    )
    
    # Test with a known MD5
    known_md5 = "4aaa4f9b53b20a0d31aa28fb8c74b7c4"
    
    logger.info(f"Testing fast download for MD5: {known_md5}")
    
    # Manually make the request to see the raw response
    params = {"md5": known_md5, "key": API_KEY}
    
    for mirror in client.MIRRORS:
        url = f"{mirror}/dyn/api/fast_download.json"
        logger.info(f"Trying mirror: {url}")
        
        try:
            response = client._make_tor_request(url, params, client.headers)
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"Response data: {json.dumps(data, indent=2)}")
                    
                    # Check if there's a download URL
                    if isinstance(data, dict):
                        download_url = data.get("url") or data.get("download_url") or data.get("fast_download_url")
                        if download_url:
                            logger.info(f"✅ Download URL found: {download_url}")
                        else:
                            logger.info("❌ No download URL in response")
                    else:
                        logger.info(f"Unexpected response format: {type(data)}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    logger.info(f"Raw response text: {response.text[:500]}")
            else:
                logger.error(f"HTTP error: {response.status_code}")
                logger.info(f"Response text: {response.text[:500]}")
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
        
        logger.info("-" * 40)

if __name__ == "__main__":
    debug_fast_download()
