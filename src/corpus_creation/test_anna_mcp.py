"""
Test script for Anna's Archive MCP tool integration

This script tests the annas-mcp tool functionality and helps understand
the output format for proper parsing.

Usage:
    python test_anna_mcp.py --test-search
    python test_anna_mcp.py --test-with-sample
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.corpus_creation.anna_archive_client import setup_logging, load_environment_config


def test_annas_mcp_installation(annas_mcp_path: str) -> bool:
    """Test if annas-mcp is properly installed and accessible"""
    try:
        result = subprocess.run([annas_mcp_path, "--help"], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ annas-mcp tool is working correctly")
            print(f"Version info: {result.stdout.split('USAGE')[0].strip()}")
            return True
        else:
            print("✗ annas-mcp tool failed to run")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error testing annas-mcp: {str(e)}")
        return False


def test_search_functionality(annas_mcp_path: str, api_key: str) -> None:
    """Test search functionality with a simple query"""
    print("\n" + "="*50)
    print("TESTING SEARCH FUNCTIONALITY")
    print("="*50)
    
    # Test with a well-known book
    test_query = "Patricia Cabot A Little Scandal"
    print(f"Testing search with query: '{test_query}'")
    
    try:
        env = os.environ.copy()
        env['ANNAS_SECRET_KEY'] = api_key
        
        cmd = [annas_mcp_path, "search", test_query]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
        
        print(f"\nReturn code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        if result.returncode == 0:
            print("✓ Search test successful")
        else:
            print("✗ Search test failed")
            
    except subprocess.TimeoutExpired:
        print("✗ Search test timed out")
    except Exception as e:
        print(f"✗ Search test error: {str(e)}")


def test_with_sample_csv(annas_mcp_path: str, api_key: str, csv_path: str) -> None:
    """Test search with books from sample CSV"""
    print("\n" + "="*50)
    print("TESTING WITH SAMPLE CSV")
    print("="*50)
    
    try:
        import csv
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            books = list(reader)
        
        print(f"Loaded {len(books)} books from CSV")
        
        # Test with first 2 books only to conserve API quota
        test_books = books[:2]
        
        for i, book in enumerate(test_books, 1):
            if not book['title'] or not book['author_name']:
                continue
                
            print(f"\n--- Testing book {i}: {book['title']} by {book['author_name']} ---")
            
            query = f"{book['author_name']} {book['title']}"
            print(f"Query: '{query}'")
            
            try:
                env = os.environ.copy()
                env['ANNAS_SECRET_KEY'] = api_key
                
                cmd = [annas_mcp_path, "search", query]
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
                
                print(f"Return code: {result.returncode}")
                if result.stdout:
                    print(f"Results found: {len(result.stdout.split('\\n')) - 1} lines")
                    # Show first few lines of output
                    lines = result.stdout.strip().split('\n')[:5]
                    for line in lines:
                        print(f"  {line}")
                    if len(result.stdout.strip().split('\n')) > 5:
                        print("  ... (output truncated)")
                        
                if result.stderr:
                    print(f"Errors: {result.stderr}")
                    
            except Exception as e:
                print(f"Error testing {book['title']}: {str(e)}")
                
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Test Anna's MCP tool integration")
    parser.add_argument("--test-search", action="store_true", 
                       help="Test basic search functionality")
    parser.add_argument("--test-with-sample", action="store_true",
                       help="Test with sample books from CSV")
    parser.add_argument("--config", default="anna_archive_config.env",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    annas_mcp_path = str(project_root / "annas-mcp")
    sample_csv_path = str(project_root / "data" / "processed" / "sample_books_for_download.csv")
    
    print("Anna's MCP Tool Test Suite")
    print("="*50)
    print(f"Project root: {project_root}")
    print(f"annas-mcp path: {annas_mcp_path}")
    print(f"Sample CSV: {sample_csv_path}")
    
    # Test installation
    if not test_annas_mcp_installation(annas_mcp_path):
        print("Installation test failed. Please check annas-mcp binary.")
        return 1
    
    # Load configuration
    config_path = project_root / args.config
    if config_path.exists():
        try:
            config = load_environment_config(str(config_path))
            api_key = config.get('ANNAS_SECRET_KEY', '')
            if not api_key or api_key == 'your_secret_key_here':
                print("\\nWarning: No valid API key found in configuration.")
                print(f"Please set ANNAS_SECRET_KEY in {config_path}")
                api_key = input("Enter your API key for testing (or press Enter to skip): ").strip()
                if not api_key:
                    print("Skipping tests that require API key")
                    return 0
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            api_key = input("Enter your API key for testing: ").strip()
    else:
        print(f"Config file not found: {config_path}")
        api_key = input("Enter your API key for testing: ").strip()
    
    # Run tests
    if args.test_search or not (args.test_search or args.test_with_sample):
        test_search_functionality(annas_mcp_path, api_key)
    
    if args.test_with_sample:
        if Path(sample_csv_path).exists():
            test_with_sample_csv(annas_mcp_path, api_key, sample_csv_path)
        else:
            print(f"Sample CSV not found: {sample_csv_path}")
    
    print("\n" + "="*50)
    print("Test completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
