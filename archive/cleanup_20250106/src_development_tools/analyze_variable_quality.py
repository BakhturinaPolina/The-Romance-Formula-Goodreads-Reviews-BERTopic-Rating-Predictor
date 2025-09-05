#!/usr/bin/env python3
"""
Variable-Specific Data Quality Analysis
Analyzes the actual variables/columns in the dataset to identify specific quality issues.
"""

import pandas as pd
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def analyze_variable_quality(input_file: str, sample_size: int = 50000):
    """
    Analyze quality issues for specific variables in the dataset.
    
    Args:
        input_file: Path to the input CSV file
        sample_size: Number of records to analyze (for performance)
    """
    print("🔍 Variable-Specific Data Quality Analysis")
    print("=" * 60)
    
    # Read a sample of the data
    print(f"📊 Loading sample of {sample_size:,} records...")
    df = pd.read_csv(input_file, nrows=sample_size)
    
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
    print(f"📋 Columns: {list(df.columns)}")
    print()
    
    # Initialize results
    quality_report = {
        'analysis_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'sample_size': len(df),
        'total_columns': len(df.columns),
        'variable_quality': {}
    }
    
    # Analyze each variable
    print("🔍 Analyzing each variable...")
    print("-" * 60)
    
    for column in df.columns:
        print(f"\n📊 Variable: {column}")
        print(f"   Data Type: {df[column].dtype}")
        
        # Missing values
        missing_count = df[column].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        # Unique values
        unique_count = df[column].nunique()
        unique_pct = (unique_count / len(df)) * 100
        
        # Basic statistics
        if df[column].dtype in ['int64', 'float64']:
            min_val = df[column].min()
            max_val = df[column].max()
            mean_val = df[column].mean()
            std_val = df[column].std()
            
            # Outlier detection (IQR method)
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            outlier_pct = (outliers / len(df)) * 100
            
            print(f"   📈 Range: {min_val:.2f} to {max_val:.2f}")
            print(f"   📊 Mean: {mean_val:.2f}, Std: {std_val:.2f}")
            print(f"   🎯 Outliers: {outliers:,} ({outlier_pct:.1f}%)")
        else:
            # For text columns, show most common values
            top_values = df[column].value_counts().head(3)
            print(f"   📝 Top values:")
            for val, count in top_values.items():
                if pd.isna(val):
                    print(f"      - NULL: {count:,}")
                else:
                    print(f"      - '{str(val)[:50]}{'...' if len(str(val)) > 50 else ''}': {count:,}")
        
        print(f"   ❌ Missing: {missing_count:,} ({missing_pct:.1f}%)")
        print(f"   🔢 Unique: {unique_count:,} ({unique_pct:.1f}%)")
        
        # Store results
        quality_report['variable_quality'][column] = {
            'data_type': str(df[column].dtype),
            'missing_count': int(missing_count),
            'missing_percentage': float(missing_pct),
            'unique_count': int(unique_count),
            'unique_percentage': float(unique_pct)
        }
        
        # Add numeric-specific metrics
        if df[column].dtype in ['int64', 'float64']:
            quality_report['variable_quality'][column].update({
                'min_value': float(min_val) if not pd.isna(min_val) else None,
                'max_value': float(max_val) if not pd.isna(max_val) else None,
                'mean_value': float(mean_val) if not pd.isna(mean_val) else None,
                'std_value': float(std_val) if not pd.isna(std_val) else None,
                'outlier_count': int(outliers),
                'outlier_percentage': float(outlier_pct)
            })
    
    # Summary analysis
    print("\n" + "=" * 60)
    print("📋 VARIABLE QUALITY SUMMARY")
    print("=" * 60)
    
    # Sort variables by missing percentage
    missing_issues = [(col, data['missing_percentage']) for col, data in quality_report['variable_quality'].items()]
    missing_issues.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🔴 Variables with Missing Values (sorted by severity):")
    for col, missing_pct in missing_issues:
        if missing_pct > 0:
            print(f"   • {col}: {missing_pct:.1f}% missing")
    
    # Sort variables by outlier percentage
    outlier_issues = [(col, data.get('outlier_percentage', 0)) for col, data in quality_report['variable_quality'].items() 
                     if 'outlier_percentage' in data]
    outlier_issues.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🟡 Variables with Outliers (sorted by severity):")
    for col, outlier_pct in outlier_issues[:10]:  # Top 10
        if outlier_pct > 0:
            print(f"   • {col}: {outlier_pct:.1f}% outliers")
    
    # Data type optimization opportunities
    print("\n🔧 Data Type Optimization Opportunities:")
    for col, data in quality_report['variable_quality'].items():
        if data['data_type'] == 'int64':
            max_val = data.get('max_value', 0)
            min_val = data.get('min_value', 0)
            if max_val <= 2147483647 and min_val >= -2147483648:
                print(f"   • {col}: int64 → int32 (saves 4 bytes per value)")
            elif max_val <= 32767 and min_val >= -32768:
                print(f"   • {col}: int64 → int16 (saves 6 bytes per value)")
        elif data['data_type'] == 'float64':
            print(f"   • {col}: float64 → float32 (saves 4 bytes per value)")
    
    # Save detailed report
    report_path = f"../../outputs/variable_quality_analysis_{quality_report['analysis_timestamp']}.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(quality_report, f, indent=2)
    
    print(f"\n📄 Detailed report saved: {report_path}")
    
    return quality_report

def main():
    """Main function to run variable quality analysis."""
    input_file = "../../data/processed/final_books_2000_2020_en_enhanced_20250904_215835.csv"
    
    # Ask user for sample size
    print("📊 Variable Quality Analysis")
    print("Choose analysis scope:")
    print("1. Quick analysis (10,000 records)")
    print("2. Standard analysis (50,000 records)")
    print("3. Comprehensive analysis (100,000 records)")
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        sample_size = 10000
    elif choice == "2":
        sample_size = 50000
    elif choice == "3":
        sample_size = 100000
    else:
        print("Invalid choice. Using standard analysis (50,000 records)")
        sample_size = 50000
    
    # Run analysis
    results = analyze_variable_quality(input_file, sample_size)
    
    print(f"\n✅ Analysis complete! Analyzed {results['sample_size']:,} records across {results['total_columns']} variables.")

if __name__ == "__main__":
    main()
