"""
Runner script for the sub-dataset sampling functionality.

This script provides a simple interface to create the 6,000 book sub-dataset
with equal representation across popularity tiers.

Usage:
    python run_subdataset_sampling.py [input_csv] [output_csv]

Author: Research Assistant
Date: September 2025
"""

import os
import sys
from create_subdataset_6000 import create_subdataset_6000

def main():
    """Main execution function."""
    print("Romance Novel NLP Research - Sub-dataset Sampling")
    print("=" * 60)
    
    # Set up paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Default paths
    input_path = os.path.join(project_root, "data", "processed", "romance_books_main_final.csv")
    output_path = os.path.join(project_root, "data", "processed", "romance_subdataset_6000.csv")
    
    # Allow command line overrides
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    print(f"Input CSV: {input_path}")
    print(f"Output CSV: {output_path}")
    print()
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Run the sampling
    try:
        sample_df = create_subdataset_6000(input_path, output_path)
        print(f"\n✅ Sampling completed successfully!")
        print(f"📊 Final dataset: {len(sample_df)} books")
        print(f"📁 Saved to: {output_path}")
        
        # Show tier distribution
        tier_counts = sample_df['pop_tier'].value_counts().sort_index()
        print("\nTier distribution:")
        for tier, count in tier_counts.items():
            print(f"  {tier}: {count:,} books")
            
    except Exception as e:
        print(f"❌ Error during sampling: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
