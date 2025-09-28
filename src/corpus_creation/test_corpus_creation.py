"""
Test script for romance novel corpus creation

This script tests the corpus creation pipeline with the sample books
from the CSV file to validate the approach before scaling to the full dataset.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.corpus_creation.simple_anna_client import SimpleAnnaClient, CorpusCreator, setup_logging


def test_simple_search():
    """Test basic search functionality"""
    print("="*60)
    print("TESTING SIMPLE SEARCH FUNCTIONALITY")
    print("="*60)
    
    # Setup
    download_path = project_root / "data" / "raw" / "anna_archive_corpus"
    logger = setup_logging(str(project_root / "logs" / "corpus_creation"))
    
    client = SimpleAnnaClient(str(download_path), logger)
    
    # Test with a well-known book
    test_query = "Patricia Cabot A Little Scandal"
    print(f"Testing search with query: '{test_query}'")
    
    results = client.search_books(test_query, max_results=3)
    
    if results:
        print(f"✓ Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.title} by {result.author}")
            print(f"     Type: {result.file_type}, Size: {result.file_size}")
            print(f"     URL: {result.url}")
    else:
        print("✗ No results found")
    
    return len(results) > 0


def test_with_sample_csv(max_books: int = 2):
    """Test corpus creation with sample books from CSV"""
    print("\n" + "="*60)
    print(f"TESTING CORPUS CREATION WITH {max_books} SAMPLE BOOKS")
    print("="*60)
    
    # Setup paths
    download_path = project_root / "data" / "raw" / "anna_archive_corpus"
    sample_csv_path = project_root / "data" / "processed" / "sample_books_for_download.csv"
    output_path = project_root / "data" / "intermediate" / "anna_archive_metadata" / "test_results.json"
    
    # Setup logging
    logger = setup_logging(str(project_root / "logs" / "corpus_creation"))
    
    # Create client and corpus creator
    client = SimpleAnnaClient(str(download_path), logger)
    corpus_creator = CorpusCreator(client, str(sample_csv_path), logger)
    
    # Load and process books
    try:
        books = corpus_creator.load_book_metadata()
        print(f"Loaded {len(books)} books from CSV")
        
        # Process first few books
        results = corpus_creator.process_books(books, max_books=max_books)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_creator.save_results(results, str(output_path))
        
        # Print summary
        print(f"\nResults Summary:")
        print(f"  Total books processed: {results['total_books']}")
        print(f"  Successful downloads: {results['successful_downloads']}")
        print(f"  Failed searches: {results['failed_searches']}")
        print(f"  Failed downloads: {results['failed_downloads']}")
        print(f"  Results saved to: {output_path}")
        
        if results['downloaded_files']:
            print(f"\nDownloaded files:")
            for file_info in results['downloaded_files']:
                print(f"  - {file_info['downloaded_file']}")
        
        return results['successful_downloads'] > 0
        
    except Exception as e:
        print(f"Error in corpus creation test: {str(e)}")
        logger.error(f"Corpus creation test failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Anna's Archive corpus creation")
    parser.add_argument("--test-search", action="store_true", 
                       help="Test basic search functionality")
    parser.add_argument("--test-sample", action="store_true",
                       help="Test with sample books from CSV")
    parser.add_argument("--max-books", type=int, default=2,
                       help="Maximum number of books to process in sample test")
    
    args = parser.parse_args()
    
    print("Anna's Archive Corpus Creation Test Suite")
    print("="*60)
    print(f"Project root: {project_root}")
    
    success = True
    
    # Run tests
    if args.test_search or not (args.test_search or args.test_sample):
        success &= test_simple_search()
    
    if args.test_sample:
        success &= test_with_sample_csv(args.max_books)
    
    print("\n" + "="*60)
    if success:
        print("✓ All tests completed successfully!")
        print("\nNext steps:")
        print("1. Review the downloaded files in data/raw/anna_archive_corpus/")
        print("2. Check the results in data/intermediate/anna_archive_metadata/")
        print("3. If satisfied, run the full corpus creation pipeline")
    else:
        print("✗ Some tests failed. Check the logs for details.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
