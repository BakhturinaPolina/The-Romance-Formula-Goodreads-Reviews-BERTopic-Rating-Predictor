#!/usr/bin/env python3
"""
Anna's Archive Book Matcher - Main Runner Script
Automated book matching system for romance novels
"""

import sys
import logging
from pathlib import Path
import argparse
from typing import Optional

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from anna_archive_matcher.core.book_matcher import BookMatcher
from anna_archive_matcher.utils.data_processor import AnnaArchiveDataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anna_archive_matcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the book matching process
    """
    parser = argparse.ArgumentParser(description='Anna Archive Book Matcher')
    parser.add_argument('--romance-csv', required=True, 
                       help='Path to romance books CSV file')
    parser.add_argument('--data-dir', default='data',
                       help='Path to Anna Archive data directory')
    parser.add_argument('--output-dir', default='outputs',
                       help='Path to output directory')
    parser.add_argument('--process-data', action='store_true',
                       help='Process raw Anna Archive data files')
    parser.add_argument('--similarity-threshold', type=float, default=0.8,
                       help='Similarity threshold for matching (0.0-1.0)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Process only a sample of books (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Process Anna Archive data if requested
        if args.process_data:
            logger.info("Processing Anna Archive data files...")
            processor = AnnaArchiveDataProcessor(args.data_dir)
            
            # Process all datasets
            processor.process_elasticsearch_data()
            processor.process_aac_data()
            processor.process_mariadb_data()
            
            logger.info("Data processing completed")
        
        # Step 2: Initialize book matcher
        logger.info("Initializing book matcher...")
        matcher = BookMatcher(args.data_dir)
        
        # Step 3: Load romance dataset
        logger.info(f"Loading romance dataset from {args.romance_csv}")
        romance_df = matcher.load_romance_dataset(args.romance_csv)
        
        # Apply sample size if specified
        if args.sample_size:
            romance_df = romance_df.head(args.sample_size)
            logger.info(f"Using sample of {len(romance_df)} books")
        
        # Step 4: Set up Anna Archive tables
        logger.info("Setting up Anna Archive tables...")
        matcher.setup_anna_archive_tables()
        
        # Step 5: Find matches
        logger.info("Finding matches...")
        matches_df = matcher.find_matches(romance_df, args.similarity_threshold)
        
        # Step 6: Extract MD5 hashes
        logger.info("Extracting MD5 hashes...")
        download_df = matcher.extract_md5_hashes(matches_df)
        
        # Step 7: Save results
        matches_output = output_dir / "anna_archive_matches.csv"
        download_output = output_dir / "download_ready_books.csv"
        
        matcher.save_matches(matches_df, matches_output)
        download_df.to_csv(download_output, index=False)
        
        # Step 8: Generate summary report
        generate_summary_report(romance_df, matches_df, download_df, output_dir)
        
        logger.info("Book matching completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during book matching: {e}")
        raise
    
    finally:
        # Clean up
        if 'matcher' in locals():
            matcher.close()


def generate_summary_report(romance_df, matches_df, download_df, output_dir):
    """
    Generate a summary report of the matching process
    """
    report_path = output_dir / "matching_summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("Anna Archive Book Matching Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total romance books processed: {len(romance_df)}\n")
        f.write(f"Total matches found: {len(matches_df)}\n")
        f.write(f"Books ready for download: {len(download_df)}\n")
        f.write(f"Match rate: {len(matches_df)/len(romance_df)*100:.2f}%\n")
        f.write(f"Download rate: {len(download_df)/len(romance_df)*100:.2f}%\n\n")
        
        if len(matches_df) > 0:
            f.write("Match Type Distribution:\n")
            match_types = matches_df['match_type'].value_counts()
            for match_type, count in match_types.items():
                f.write(f"  {match_type}: {count} ({count/len(matches_df)*100:.1f}%)\n")
            f.write("\n")
            
            f.write("Source Distribution:\n")
            sources = matches_df['source'].value_counts()
            for source, count in sources.items():
                f.write(f"  {source}: {count} ({count/len(matches_df)*100:.1f}%)\n")
            f.write("\n")
        
        f.write("Top 10 Matched Books:\n")
        if len(matches_df) > 0:
            top_matches = matches_df.nlargest(10, 'similarity_score')
            for idx, match in top_matches.iterrows():
                f.write(f"  {match['title']} by {match['author_name']} "
                       f"(similarity: {match['similarity_score']:.3f})\n")
    
    logger.info(f"Summary report saved to {report_path}")


if __name__ == "__main__":
    main()
