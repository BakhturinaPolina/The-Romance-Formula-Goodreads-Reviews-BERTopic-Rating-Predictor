"""
Runner script for fuzzy text matching functionality.

This script provides an easy way to run the fuzzy matching between
Goodreads books and external text datasets with predefined configurations.

Author: Research Assistant
Date: 2025-01-09
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.text_matching.match_goodreads_to_texts import main, MatchConfig


def run_matching_with_hf_dataset(
    goodreads_csv: str,
    output_dir: str = "data/processed",
    use_custom_config: bool = False,
    config: Optional[MatchConfig] = None
):
    """
    Run matching using the Hugging Face AlekseyKorshuk/romance-books dataset.
    
    Args:
        goodreads_csv: Path to the Goodreads CSV file
        output_dir: Directory to save output files
        use_custom_config: Whether to use custom configuration
        config: Custom MatchConfig if use_custom_config is True
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output file paths
    matches_file = os.path.join(output_dir, "goodreads_to_hf_matches_definitive.csv")
    review_file = os.path.join(output_dir, "goodreads_to_hf_matches_needs_review.csv")
    
    print("=" * 60)
    print("FUZZY TEXT MATCHING - Hugging Face Dataset")
    print("=" * 60)
    print(f"Goodreads CSV: {goodreads_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Matches file: {matches_file}")
    print(f"Review file: {review_file}")
    
    if use_custom_config and config:
        print(f"Using custom configuration:")
        print(f"  - Title accept threshold: {config.title_threshold_accept}")
        print(f"  - Title review threshold: {config.title_threshold_review}")
        print(f"  - Author weight: {config.author_weight}")
        print(f"  - Title weight: {config.title_weight}")
    
    print("=" * 60)
    
    try:
        main(
            goodreads_csv=goodreads_csv,
            output_matches_csv=matches_file,
            output_review_csv=review_file,
            use_hf_texts=True,
            local_texts_csv=None
        )
        print("\n✅ Matching completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during matching: {str(e)}")
        raise


def run_matching_with_local_dataset(
    goodreads_csv: str,
    local_texts_csv: str,
    output_dir: str = "data/processed",
    use_custom_config: bool = False,
    config: Optional[MatchConfig] = None
):
    """
    Run matching using a local text dataset CSV.
    
    Args:
        goodreads_csv: Path to the Goodreads CSV file
        local_texts_csv: Path to the local text dataset CSV
        output_dir: Directory to save output files
        use_custom_config: Whether to use custom configuration
        config: Custom MatchConfig if use_custom_config is True
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output file paths
    matches_file = os.path.join(output_dir, "goodreads_to_local_matches_definitive.csv")
    review_file = os.path.join(output_dir, "goodreads_to_local_matches_needs_review.csv")
    
    print("=" * 60)
    print("FUZZY TEXT MATCHING - Local Dataset")
    print("=" * 60)
    print(f"Goodreads CSV: {goodreads_csv}")
    print(f"Local texts CSV: {local_texts_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Matches file: {matches_file}")
    print(f"Review file: {review_file}")
    
    if use_custom_config and config:
        print(f"Using custom configuration:")
        print(f"  - Title accept threshold: {config.title_threshold_accept}")
        print(f"  - Title review threshold: {config.title_threshold_review}")
        print(f"  - Author weight: {config.author_weight}")
        print(f"  - Title weight: {config.title_weight}")
    
    print("=" * 60)
    
    try:
        main(
            goodreads_csv=goodreads_csv,
            output_matches_csv=matches_file,
            output_review_csv=review_file,
            use_hf_texts=False,
            local_texts_csv=local_texts_csv
        )
        print("\n✅ Matching completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during matching: {str(e)}")
        raise


def create_relaxed_config() -> MatchConfig:
    """Create a more relaxed configuration for better recall."""
    return MatchConfig(
        title_threshold_accept=85,      # Lower threshold for acceptance
        title_threshold_review=75,      # Lower threshold for review
        author_weight=0.20,             # Lower author weight
        title_weight=0.75,              # Higher title weight
        year_bonus_per_match=3,         # Lower year bonus
        year_tolerance=3,               # More year tolerance
        year_bonus_close=1,
        max_candidates_per_block=100,   # More candidates
        block_title_prefix=12,          # Shorter blocking prefix
        top_k=5                         # More top candidates
    )


def create_strict_config() -> MatchConfig:
    """Create a stricter configuration for better precision."""
    return MatchConfig(
        title_threshold_accept=95,      # Higher threshold for acceptance
        title_threshold_review=85,      # Higher threshold for review
        author_weight=0.30,             # Higher author weight
        title_weight=0.65,              # Lower title weight
        year_bonus_per_match=8,         # Higher year bonus
        year_tolerance=1,               # Less year tolerance
        year_bonus_close=3,
        max_candidates_per_block=30,    # Fewer candidates
        block_title_prefix=16,          # Longer blocking prefix
        top_k=2                         # Fewer top candidates
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fuzzy text matching with predefined configurations")
    parser.add_argument("--goodreads_csv", type=str, required=True, 
                       help="Path to Goodreads CSV file")
    parser.add_argument("--local_texts_csv", type=str, default=None,
                       help="Path to local text dataset CSV (if not using HF dataset)")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Output directory for results")
    parser.add_argument("--config", type=str, choices=["default", "relaxed", "strict"], 
                       default="default", help="Configuration preset to use")
    
    args = parser.parse_args()
    
    # Select configuration
    if args.config == "relaxed":
        config = create_relaxed_config()
        use_custom_config = True
    elif args.config == "strict":
        config = create_strict_config()
        use_custom_config = True
    else:
        config = None
        use_custom_config = False
    
    # Run matching
    if args.local_texts_csv:
        run_matching_with_local_dataset(
            goodreads_csv=args.goodreads_csv,
            local_texts_csv=args.local_texts_csv,
            output_dir=args.output_dir,
            use_custom_config=use_custom_config,
            config=config
        )
    else:
        run_matching_with_hf_dataset(
            goodreads_csv=args.goodreads_csv,
            output_dir=args.output_dir,
            use_custom_config=use_custom_config,
            config=config
        )
