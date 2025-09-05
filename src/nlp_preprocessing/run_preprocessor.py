#!/usr/bin/env python3
"""
Run the Text Preprocessor for NLP Analysis
Executes comprehensive text preprocessing on romance novel dataset.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import using relative import
from .text_preprocessor import TextPreprocessor


def main():
    """Run the text preprocessor."""
    print("🔤 Running Text Preprocessor for NLP Analysis...")
    print("📖 Features:")
    print("  - HTML cleaning and text normalization")
    print("  - Popular shelves format standardization")
    print("  - Genre normalization and categorization")
    print("  - Comprehensive validation and reporting")
    
    # Initialize the preprocessor
    preprocessor = TextPreprocessor()
    
    # Run complete preprocessing
    results = preprocessor.run_complete_preprocessing()
    
    # Print summary
    preprocessor.print_preprocessing_summary()
    
    print(f"\n✅ Text preprocessing completed successfully!")
    print(f"📄 Processed dataset: {results['dataset_path']}")
    print(f"📄 Detailed report: {results['report_path']}")
    
    return results


if __name__ == "__main__":
    main()
