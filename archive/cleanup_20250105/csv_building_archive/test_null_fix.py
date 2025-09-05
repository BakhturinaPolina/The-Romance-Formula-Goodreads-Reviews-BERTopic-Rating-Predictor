#!/usr/bin/env python3
"""
Test script to compare original and enhanced CSV builders on a small sample.
This script will demonstrate the null value fix and show before/after results.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from csv_building.final_csv_builder_working import OptimizedFinalCSVBuilder
from csv_building.final_csv_builder_fixed import EnhancedFinalCSVBuilder


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> dict:
    """Compare two dataframes and return comparison metrics."""
    comparison = {
        'name1': name1,
        'name2': name2,
        'rows1': len(df1),
        'rows2': len(df2),
        'columns1': len(df1.columns),
        'columns2': len(df2.columns),
        'null_comparison': {}
    }
    
    # Compare null values in common columns
    common_columns = set(df1.columns) & set(df2.columns)
    
    for col in common_columns:
        if col in df1.columns and col in df2.columns:
            nulls1 = df1[col].isnull().sum()
            nulls2 = df2[col].isnull().sum()
            comparison['null_comparison'][col] = {
                'nulls1': nulls1,
                'nulls2': nulls2,
                'difference': nulls2 - nulls1,
                'improvement': nulls1 - nulls2  # Positive means improvement
            }
    
    return comparison


def print_comparison_report(comparison: dict):
    """Print a detailed comparison report."""
    print(f"\n📊 COMPARISON REPORT: {comparison['name1']} vs {comparison['name2']}")
    print("=" * 80)
    
    print(f"📈 Dataset Size:")
    print(f"  - {comparison['name1']}: {comparison['rows1']:,} rows, {comparison['columns1']} columns")
    print(f"  - {comparison['name2']}: {comparison['rows2']:,} rows, {comparison['columns2']} columns")
    
    print(f"\n🔍 Null Value Comparison:")
    print(f"{'Column':<30} {'Original':<10} {'Enhanced':<10} {'Difference':<12} {'Improvement':<12}")
    print("-" * 80)
    
    total_improvement = 0
    for col, metrics in comparison['null_comparison'].items():
        improvement = metrics['improvement']
        total_improvement += improvement
        
        status = "✅" if improvement > 0 else "❌" if improvement < 0 else "➖"
        print(f"{col:<30} {metrics['nulls1']:<10} {metrics['nulls2']:<10} {metrics['difference']:<12} {improvement:<12} {status}")
    
    print("-" * 80)
    print(f"{'TOTAL IMPROVEMENT':<30} {'':<10} {'':<10} {'':<12} {total_improvement:<12}")
    
    if total_improvement > 0:
        print(f"\n🎉 SUCCESS: Enhanced version reduced nulls by {total_improvement}")
    elif total_improvement < 0:
        print(f"\n⚠️  WARNING: Enhanced version increased nulls by {abs(total_improvement)}")
    else:
        print(f"\n➖ NO CHANGE: Both versions have the same null count")


def test_original_builder(sample_size: int = 100) -> tuple:
    """Test the original CSV builder."""
    print(f"\n🔧 Testing ORIGINAL CSV Builder (sample size: {sample_size})")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        builder = OptimizedFinalCSVBuilder()
        output_path = builder.build_final_csv_optimized(sample_size=sample_size)
        
        # Load the result
        df = pd.read_csv(output_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Original builder completed in {processing_time:.2f} seconds")
        print(f"📁 Output: {output_path}")
        print(f"📊 Result: {len(df)} rows, {len(df.columns)} columns")
        
        return df, output_path, processing_time
        
    except Exception as e:
        print(f"❌ Original builder failed: {e}")
        return None, None, 0


def test_enhanced_builder(sample_size: int = 100) -> tuple:
    """Test the enhanced CSV builder."""
    print(f"\n🚀 Testing ENHANCED CSV Builder (sample size: {sample_size})")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        builder = EnhancedFinalCSVBuilder()
        output_path = builder.build_final_csv_enhanced(sample_size=sample_size)
        
        # Load the result
        df = pd.read_csv(output_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Enhanced builder completed in {processing_time:.2f} seconds")
        print(f"📁 Output: {output_path}")
        print(f"📊 Result: {len(df)} rows, {len(df.columns)} columns")
        
        return df, output_path, processing_time
        
    except Exception as e:
        print(f"❌ Enhanced builder failed: {e}")
        return None, None, 0


def analyze_data_quality(df: pd.DataFrame, builder_name: str):
    """Analyze data quality of a dataframe."""
    print(f"\n🔍 Data Quality Analysis - {builder_name}")
    print("-" * 50)
    
    # Basic stats
    print(f"📊 Basic Statistics:")
    print(f"  - Total rows: {len(df):,}")
    print(f"  - Total columns: {len(df.columns)}")
    
    # Null analysis
    print(f"\n📋 Null Value Analysis:")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls > 0:
        print(f"  - Total null values: {total_nulls:,}")
        print(f"  - Columns with nulls:")
        for col, count in null_counts[null_counts > 0].items():
            percentage = (count / len(df)) * 100
            print(f"    • {col}: {count:,} ({percentage:.1f}%)")
    else:
        print(f"  - No null values found! 🎉")
    
    # Key field analysis
    key_fields = ['work_id', 'title', 'publication_year', 'num_pages_median', 'author_name']
    print(f"\n🔑 Key Field Analysis:")
    for field in key_fields:
        if field in df.columns:
            null_count = df[field].isnull().sum()
            percentage = (null_count / len(df)) * 100
            status = "✅" if null_count == 0 else "⚠️"
            print(f"  - {field}: {null_count:,} nulls ({percentage:.1f}%) {status}")


def main():
    """Run the complete test comparing original and enhanced builders."""
    print("🧪 NULL VALUE FIX TEST")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    sample_size = 100
    
    # Test original builder
    original_df, original_path, original_time = test_original_builder(sample_size)
    
    # Test enhanced builder
    enhanced_df, enhanced_path, enhanced_time = test_enhanced_builder(sample_size)
    
    # Compare results
    if original_df is not None and enhanced_df is not None:
        print(f"\n📊 PERFORMANCE COMPARISON")
        print("-" * 40)
        print(f"Original builder:  {original_time:.2f} seconds")
        print(f"Enhanced builder:  {enhanced_time:.2f} seconds")
        print(f"Time difference:   {enhanced_time - original_time:+.2f} seconds")
        
        # Detailed comparison
        comparison = compare_dataframes(original_df, enhanced_df, "Original", "Enhanced")
        print_comparison_report(comparison)
        
        # Individual analysis
        analyze_data_quality(original_df, "Original Builder")
        analyze_data_quality(enhanced_df, "Enhanced Builder")
        
        # Save comparison report
        save_comparison_report(comparison, original_df, enhanced_df)
        
    else:
        print("\n❌ Test failed - could not compare results")
        if original_df is None:
            print("  - Original builder failed")
        if enhanced_df is None:
            print("  - Enhanced builder failed")


def save_comparison_report(comparison: dict, original_df: pd.DataFrame, enhanced_df: pd.DataFrame):
    """Save detailed comparison report to file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path("outputs") / f"null_fix_comparison_report_{timestamp}.md"
    
    # Ensure outputs directory exists
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Null Value Fix Comparison Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Sample Size**: 100 books\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Original Builder**: {comparison['rows1']} rows, {comparison['columns1']} columns\n")
        f.write(f"- **Enhanced Builder**: {comparison['rows2']} rows, {comparison['columns2']} columns\n\n")
        
        f.write("## Null Value Comparison\n\n")
        f.write("| Column | Original | Enhanced | Difference | Improvement |\n")
        f.write("|--------|----------|----------|------------|-------------|\n")
        
        total_improvement = 0
        for col, metrics in comparison['null_comparison'].items():
            improvement = metrics['improvement']
            total_improvement += improvement
            status = "✅" if improvement > 0 else "❌" if improvement < 0 else "➖"
            f.write(f"| {col} | {metrics['nulls1']} | {metrics['nulls2']} | {metrics['difference']} | {improvement} {status} |\n")
        
        f.write(f"\n**Total Improvement**: {total_improvement}\n\n")
        
        if total_improvement > 0:
            f.write("## Result\n\n")
            f.write("✅ **SUCCESS**: The enhanced builder successfully reduced null values!\n\n")
        else:
            f.write("## Result\n\n")
            f.write("⚠️ **WARNING**: The enhanced builder did not improve null values.\n\n")
        
        f.write("## Detailed Analysis\n\n")
        f.write("### Original Builder Data Quality\n\n")
        f.write("```\n")
        f.write(f"Total rows: {len(original_df):,}\n")
        f.write(f"Total columns: {len(original_df.columns)}\n")
        f.write(f"Total nulls: {original_df.isnull().sum().sum():,}\n")
        f.write("```\n\n")
        
        f.write("### Enhanced Builder Data Quality\n\n")
        f.write("```\n")
        f.write(f"Total rows: {len(enhanced_df):,}\n")
        f.write(f"Total columns: {len(enhanced_df.columns)}\n")
        f.write(f"Total nulls: {enhanced_df.isnull().sum().sum():,}\n")
        f.write("```\n\n")
    
    print(f"\n📋 Comparison report saved to: {report_path}")


if __name__ == "__main__":
    main()
