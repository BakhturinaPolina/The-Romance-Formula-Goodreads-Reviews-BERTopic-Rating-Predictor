#!/usr/bin/env python3
"""
Book Download Research Component - Test Real MCP Integration
Test script for real anna-mcp server integration
"""

import logging
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp_integration import AnnaMCPIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_real_mcp_integration():
    """Test the real MCP integration with actual anna-mcp server calls"""
    
    logger.info("=== TESTING REAL MCP INTEGRATION ===")
    
    # Initialize MCP integration
    mcp = AnnaMCPIntegration()
    
    if not mcp.mcp_available:
        logger.error("MCP server is not available - cannot test real integration")
        return False
    
    # Test search with a simple term
    test_search_terms = [
        "romance novel",
        "Jane Austen",
        "Pride and Prejudice"
    ]
    
    for search_term in test_search_terms:
        logger.info(f"\n--- Testing search: '{search_term}' ---")
        
        try:
            results = mcp.search_books(search_term)
            logger.info(f"Search results: {len(results)} found")
            
            if results:
                # Show first result
                first_result = results[0]
                logger.info(f"First result: {first_result}")
                
                # Test download with first result
                logger.info(f"Testing download of: {first_result.get('title', 'Unknown')}")
                download_success = mcp.download_book(
                    first_result.get('hash', ''),
                    first_result.get('title', 'test_book'),
                    first_result.get('format', 'epub')
                )
                
                logger.info(f"Download success: {download_success}")
                
                # Only test one download per search term to avoid rate limiting
                break
            else:
                logger.warning(f"No results found for: {search_term}")
                
        except Exception as e:
            logger.error(f"Error testing search '{search_term}': {e}")
    
    logger.info("\n=== REAL MCP INTEGRATION TEST COMPLETED ===")
    return True

def test_mcp_availability():
    """Test MCP server availability and basic functionality"""
    
    logger.info("=== TESTING MCP AVAILABILITY ===")
    
    mcp = AnnaMCPIntegration()
    
    logger.info(f"MCP binary path: {mcp.mcp_binary}")
    logger.info(f"Download directory: {mcp.download_dir}")
    logger.info(f"MCP available: {mcp.mcp_available}")
    
    if mcp.mcp_available:
        logger.info("✓ MCP server is available and ready for testing")
        return True
    else:
        logger.error("✗ MCP server is not available")
        return False

def main():
    """Main test function"""
    
    logger.info("Starting MCP integration tests...")
    
    # Test availability first
    if not test_mcp_availability():
        logger.error("MCP server not available - exiting")
        sys.exit(1)
    
    # Test real integration
    try:
        test_real_mcp_integration()
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
