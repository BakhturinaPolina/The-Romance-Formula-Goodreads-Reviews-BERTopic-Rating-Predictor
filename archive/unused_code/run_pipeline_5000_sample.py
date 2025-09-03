#!/usr/bin/env python3
"""
Run Pipeline with 5,000 Book Sample
===================================

Simple script to run the existing pipeline with a 5,000 book sample.
This reuses the existing pipeline code instead of creating new code.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_processing.pipeline_runner import PipelineRunner

def main():
    """Run the pipeline with 5,000 book sample."""
    print("🚀 Running Pipeline with 5,000 Book Sample")
    print("=" * 50)
    
    try:
        # Initialize pipeline runner
        pipeline = PipelineRunner()
        
        # Run pipeline with 5,000 book sample
        pipeline.run_pipeline(sample_size=5000)
        
        print("\n" + "=" * 50)
        print("✅ Pipeline completed successfully!")
        print("📁 Check the following directories for outputs:")
        print("   - data/processed/ - Final datasets")
        print("   - logs/ - Execution logs and validation reports")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
