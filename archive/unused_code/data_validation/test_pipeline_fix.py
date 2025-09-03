#!/usr/bin/env python3
"""
Test script to verify pipeline fixes work correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.pipeline_runner import PipelineRunner


def test_pipeline_fix():
    """Test the pipeline with the fixes applied."""
    print("=== TESTING PIPELINE FIXES ===")
    
    try:
        # Initialize pipeline runner
        print("Initializing pipeline runner...")
        runner = PipelineRunner()
        print("✅ Pipeline runner initialized successfully")
        
        # Test with a small sample to see if it works
        print("\nTesting pipeline with small sample...")
        runner.run_pipeline(sample_size=100)
        
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_pipeline_fix()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
