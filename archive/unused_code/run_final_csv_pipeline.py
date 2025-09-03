#!/usr/bin/env python3
"""
Final CSV Pipeline Runner

This script runs the final CSV pipeline that builds the CSV exactly as specified:
- Build master table first with all required columns
- Apply English language filtering
- Implement proper publication year logic
- Add author selection with tie-breakers
- Add weighted aggregations
- Apply final filters (2000-2020, English editions)
- Output to specified path
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from data_processing.pipeline_runner import PipelineRunner


def main():
    """Run the final CSV pipeline."""
    print("🚀 Starting Final CSV Pipeline (2000-2020 English Editions)")
    print("=" * 60)
    
    try:
        # Initialize pipeline runner
        pipeline = PipelineRunner()
        
        # Run the final CSV pipeline
        pipeline.run_final_csv_pipeline()
        
        print("=" * 60)
        print("✅ Final CSV Pipeline completed successfully!")
        print("📁 Output file: data/processed/final_books_2000_2020_en.csv")
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ Final CSV Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
