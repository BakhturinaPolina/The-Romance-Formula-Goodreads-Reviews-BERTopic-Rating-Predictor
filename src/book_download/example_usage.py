#!/usr/bin/env python3
"""
Example usage of the updated Anna's Archive API integration
Shows how to use the new MD5-based download functionality
"""

import os
import sys
import logging

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from anna_api_client import AnnaAPIClient
from mcp_integration import AnnaMCPIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def example_direct_api_usage():
    """Example of using the Anna API client directly"""
    logger.info("=== DIRECT API USAGE EXAMPLE ===")
    
    # Initialize the API client
    client = AnnaAPIClient()
    
    # Example 1: Download by MD5 hash
    md5_hash = "9fbeb1ac79a509bcc8a17d6137b929e7"  # Replace with real MD5
    logger.info(f"Downloading book with MD5: {md5_hash}")
    
    result = client.download_book(md5_hash, "example_book.epub")
    if result.get("success"):
        logger.info(f"Download successful: {result['filepath']}")
    else:
        logger.warning(f"Download failed: {result.get('message')}")
    
    # Example 2: Download by Anna's Archive URL
    aa_url = "https://annas-archive.se/md5/9fbeb1ac79a509bcc8a17d6137b929e7"
    logger.info(f"Downloading book from URL: {aa_url}")
    
    result = client.download_book(aa_url, "example_book_from_url.epub")
    if result.get("success"):
        logger.info(f"Download successful: {result['filepath']}")
    else:
        logger.warning(f"Download failed: {result.get('message')}")

def example_mcp_integration_usage():
    """Example of using the MCP integration"""
    logger.info("\n=== MCP INTEGRATION USAGE EXAMPLE ===")
    
    # Initialize MCP integration
    mcp = AnnaMCPIntegration()
    
    # Example 1: Download by MD5 using MCP integration
    md5_hash = "9fbeb1ac79a509bcc8a17d6137b929e7"  # Replace with real MD5
    logger.info(f"Downloading book with MD5 using MCP: {md5_hash}")
    
    result = mcp.download_by_md5(md5_hash, "example_book_mcp.epub")
    if result.get("success"):
        logger.info(f"Download successful: {result['filepath']}")
    else:
        logger.warning(f"Download failed: {result.get('message')}")
    
    # Example 2: Traditional search + download (if MCP server available)
    if mcp.mcp_available:
        logger.info("MCP server available - testing search functionality")
        search_results = mcp.search_books("romance novel")
        if search_results:
            logger.info(f"Found {len(search_results)} search results")
            # Download first result
            first_result = search_results[0]
            success = mcp.download_book(
                first_result.get('hash', ''),
                first_result.get('title', 'unknown'),
                first_result.get('format', 'epub')
            )
            logger.info(f"Download success: {success}")
        else:
            logger.warning("No search results found")
    else:
        logger.info("MCP server not available - using Anna API client only")

def example_batch_processing():
    """Example of batch processing with download manager"""
    logger.info("\n=== BATCH PROCESSING EXAMPLE ===")
    
    # This would require a CSV file with MD5 hashes
    # For demonstration, we'll show the structure
    
    logger.info("To use batch processing:")
    logger.info("1. Create a CSV file with columns: work_id, title, author_name, md5_hash")
    logger.info("2. Use BookDownloadManager.run_md5_download_batch()")
    logger.info("3. Example CSV structure:")
    logger.info("   work_id,title,author_name,publication_year,md5_hash")
    logger.info("   1,Example Book,Example Author,2020,9fbeb1ac79a509bcc8a17d6137b929e7")
    logger.info("   2,Another Book,Another Author,2021,1234567890abcdef1234567890abcdef")
    
    # Example code (commented out since we don't have a real CSV):
    """
    from download_manager import BookDownloadManager
    
    manager = BookDownloadManager(
        csv_path="books_with_md5.csv",
        daily_limit=10
    )
    
    summary = manager.run_md5_download_batch(max_books=5)
    print(f"Processed: {summary['processed']}, Downloaded: {summary['downloaded']}")
    """

def main():
    """Main example function"""
    logger.info("Anna's Archive API Integration Examples")
    logger.info("=====================================")
    
    # Check if API key is set
    api_key = os.getenv('ANNAS_SECRET_KEY')
    if not api_key:
        logger.warning("ANNAS_SECRET_KEY environment variable not set")
        logger.warning("Set it with: export ANNAS_SECRET_KEY='your_secret_key'")
        logger.warning("Continuing with examples (downloads will fail without valid key)")
    else:
        logger.info(f"API key found: {api_key[:10]}...")
    
    # Run examples
    example_direct_api_usage()
    example_mcp_integration_usage()
    example_batch_processing()
    
    logger.info("\n=== EXAMPLES COMPLETED ===")
    logger.info("For more information, see:")
    logger.info("- anna_api_client.py: Direct API client")
    logger.info("- mcp_integration.py: MCP server integration")
    logger.info("- download_manager.py: Batch download management")

if __name__ == "__main__":
    main()
