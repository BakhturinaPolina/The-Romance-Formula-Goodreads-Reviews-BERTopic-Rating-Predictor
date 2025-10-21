#!/usr/bin/env python3
"""
Test MCP tools directly to investigate search issues
"""

import os
import sys
import logging
import subprocess
import json
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_mcp_search_direct():
    """Test MCP search functionality directly"""
    logger.info("=== TESTING MCP SEARCH DIRECTLY ===")
    
    # Set environment variables
    env = os.environ.copy()
    env['ANNAS_SECRET_KEY'] = 'BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP'
    env['ANNAS_DOWNLOAD_PATH'] = '/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download'
    
    # Test different search terms
    test_terms = [
        "romance",
        "fiction", 
        "book",
        "Jane Austen",
        "Pride and Prejudice",
        "Harry Potter",
        "The Great Gatsby",
        "1984",
        "To Kill a Mockingbird",
        "The Catcher in the Rye"
    ]
    
    results = {}
    
    for term in test_terms:
        logger.info(f"Testing search term: '{term}'")
        
        try:
            # Run MCP search command
            result = subprocess.run([
                '/home/polina/.local/bin/annas-mcp', 'search', term
            ], capture_output=True, text=True, timeout=30, env=env)
            
            results[term] = {
                'returncode': result.returncode,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'success': result.returncode == 0 and result.stdout.strip() != "No books found."
            }
            
            logger.info(f"  Return code: {result.returncode}")
            logger.info(f"  Output: {result.stdout.strip()}")
            if result.stderr:
                logger.info(f"  Error: {result.stderr.strip()}")
            
        except subprocess.TimeoutExpired:
            logger.error(f"  Timeout for term: {term}")
            results[term] = {'error': 'timeout'}
        except Exception as e:
            logger.error(f"  Exception for term {term}: {e}")
            results[term] = {'error': str(e)}
    
    return results

def test_mcp_server_mode():
    """Test if MCP server mode works differently"""
    logger.info("=== TESTING MCP SERVER MODE ===")
    
    try:
        # Start MCP server in background
        process = subprocess.Popen([
            '/home/polina/.local/bin/annas-mcp', 'mcp'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for server to start
        import time
        time.sleep(2)
        
        # Check if server is running
        if process.poll() is None:
            logger.info("MCP server started successfully")
            
            # Try to send a search request via MCP protocol
            # This would require MCP client implementation
            logger.info("MCP server is running, but direct MCP protocol testing requires client implementation")
            
            # Terminate server
            process.terminate()
            process.wait()
            logger.info("MCP server terminated")
        else:
            logger.error("MCP server failed to start")
            stdout, stderr = process.communicate()
            logger.error(f"Server output: {stdout}")
            logger.error(f"Server error: {stderr}")
            
    except Exception as e:
        logger.error(f"Error testing MCP server mode: {e}")

def test_network_connectivity():
    """Test network connectivity to Anna's Archive"""
    logger.info("=== TESTING NETWORK CONNECTIVITY ===")
    
    import urllib.request
    import urllib.error
    
    test_urls = [
        "https://annas-archive.org",
        "https://annas-archive.se",
        "https://archive.org",
        "https://google.com"
    ]
    
    for url in test_urls:
        try:
            logger.info(f"Testing connectivity to: {url}")
            response = urllib.request.urlopen(url, timeout=10)
            logger.info(f"  Status: {response.status}")
            logger.info(f"  Headers: {dict(response.headers)}")
        except urllib.error.URLError as e:
            logger.error(f"  Connection failed: {e}")
        except Exception as e:
            logger.error(f"  Error: {e}")

def test_environment_variables():
    """Test environment variable configuration"""
    logger.info("=== TESTING ENVIRONMENT VARIABLES ===")
    
    required_vars = ['ANNAS_SECRET_KEY', 'ANNAS_DOWNLOAD_PATH']
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"{var}: {'*' * (len(value) - 4) + value[-4:] if len(value) > 4 else '***'}")
        else:
            logger.error(f"{var}: NOT SET")
    
    # Test download directory
    download_path = os.environ.get('ANNAS_DOWNLOAD_PATH')
    if download_path:
        if os.path.exists(download_path):
            logger.info(f"Download path exists: {download_path}")
        else:
            logger.warning(f"Download path does not exist: {download_path}")
            try:
                os.makedirs(download_path, exist_ok=True)
                logger.info(f"Created download path: {download_path}")
            except Exception as e:
                logger.error(f"Failed to create download path: {e}")

def main():
    """Main test function"""
    logger.info("Starting MCP investigation tests...")
    
    # Test environment variables
    test_environment_variables()
    print()
    
    # Test network connectivity
    test_network_connectivity()
    print()
    
    # Test MCP search directly
    search_results = test_mcp_search_direct()
    print()
    
    # Test MCP server mode
    test_mcp_server_mode()
    print()
    
    # Summary
    logger.info("=== INVESTIGATION SUMMARY ===")
    
    successful_searches = sum(1 for r in search_results.values() if r.get('success', False))
    total_searches = len(search_results)
    
    logger.info(f"Successful searches: {successful_searches}/{total_searches}")
    
    if successful_searches == 0:
        logger.warning("No successful searches found. Possible issues:")
        logger.warning("1. Anna's Archive database not accessible")
        logger.warning("2. API key invalid or expired")
        logger.warning("3. Network connectivity issues")
        logger.warning("4. MCP server configuration problems")
    else:
        logger.info("Some searches successful - investigating patterns...")
        for term, result in search_results.items():
            if result.get('success'):
                logger.info(f"  Successful: {term}")
    
    # Save results
    results_file = "mcp_investigation_results.json"
    with open(results_file, 'w') as f:
        json.dump(search_results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
