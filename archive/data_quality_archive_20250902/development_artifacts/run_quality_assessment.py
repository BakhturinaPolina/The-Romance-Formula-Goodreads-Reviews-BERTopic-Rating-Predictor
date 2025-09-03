#!/usr/bin/env python3
"""
Runner script for Data Quality Assessment
Step 1 of the EDA plan: Data Quality Assessment and Missing Values Handling

Usage:
    python run_quality_assessment.py [dataset_filename]
    
If no filename is provided, the script will automatically find the latest processed dataset.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from data_quality.data_quality_assessment import DataQualityAssessment


def main():
    """Run the data quality assessment."""
    print("🔍 Romance Novel Dataset - Data Quality Assessment")
    print("=" * 60)
    print("Step 1: Data Quality Assessment and Missing Values Handling")
    print("=" * 60)
    
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print("\nUsage:")
        print("  python run_quality_assessment.py [dataset_filename]")
        print("\nArguments:")
        print("  dataset_filename  Optional: Specific CSV file to analyze")
        print("                    If not provided, will find the latest processed dataset")
        print("\nExamples:")
        print("  python run_quality_assessment.py")
        print("  python run_quality_assessment.py final_books_2000_2020_en_enhanced_titles_20250902_001152.csv")
        return 0
    
    # Get dataset filename from command line if provided
    dataset_filename = None
    if len(sys.argv) > 1:
        dataset_filename = sys.argv[1]
        print(f"📚 Using specified dataset: {dataset_filename}")
    else:
        print("📚 Will automatically find the latest processed dataset")
    
    # Initialize and run assessment
    assessor = DataQualityAssessment()
    
    try:
        results = assessor.run_full_assessment(dataset_filename)
        
        if results:
            print(f"\n✅ Assessment completed successfully!")
            print(f"📋 Report saved to: {results.get('report_path', 'Unknown')}")
            
            # Show key findings
            if 'missing_values' in results:
                print(f"\n📊 Key Findings:")
                for field, stats in results['missing_values'].items():
                    print(f"  - {field}: {stats['missing_count']:,} missing ({stats['missing_percentage']:.1f}%)")
            
            if 'nlp_readiness' in results:
                nlp_stats = results['nlp_readiness']
                print(f"  - NLP Ready: {nlp_stats['ready_for_nlp']:,} books")
                print(f"  - Excluded: {nlp_stats['excluded_from_nlp']:,} books")
            
        else:
            print("\n❌ Assessment failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during assessment: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
