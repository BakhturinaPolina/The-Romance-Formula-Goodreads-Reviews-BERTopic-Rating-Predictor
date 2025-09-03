#!/usr/bin/env python3
"""
Basic Data Cleaning for Romance Novel Dataset
Simple title cleaning only - no complex transformations.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Optional

class BasicRomanceCleaner:
    """
    Basic data cleaner that only performs simple title cleaning.
    """
    
    def __init__(self, input_path: str, output_dir: str = "data/processed"):
        """
        Initialize the basic cleaner.
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory for output files
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple bracket pattern for title cleaning
        self.bracket_pattern = re.compile(r'\s*\([^)]*\)|\s*\[[^\]]*\]')
    
    def clean_title_simple(self, title: str) -> tuple[str, str]:
        """
        Simple title cleaning: just strip everything within brackets/parentheses.
        
        Args:
            title: Original title string
            
        Returns:
            Tuple of (cleaned_title, removed_content)
        """
        if pd.isna(title) or not isinstance(title, str):
            return title, ""
        
        original_title = title.strip()
        cleaned_title = original_title
        removed_parts = []
        
        # Find and remove all bracket content
        for match in self.bracket_pattern.finditer(original_title):
            removed_parts.append(match.group(0))
            cleaned_title = cleaned_title.replace(match.group(0), '')
        
        # Clean up extra whitespace
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title)
        cleaned_title = cleaned_title.strip()
        
        # If cleaning resulted in empty title, return original
        if not cleaned_title:
            return original_title, ""
        
        removed_content = '; '.join(removed_parts) if removed_parts else ""
        return cleaned_title, removed_content
    
    def clean_dataset(self, output_filename: Optional[str] = None) -> str:
        """
        Clean the dataset with basic title cleaning only.
        
        Args:
            output_filename: Optional custom output filename
            
        Returns:
            Path to the cleaned dataset
        """
        print("🧹 Starting basic data cleaning (title cleaning only)...")
        
        # Load dataset
        print(f"📚 Loading dataset from {self.input_path}")
        df = pd.read_csv(self.input_path)
        print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
        
        # Create backup columns
        df['title_original'] = df['title'].copy()
        df['title_cleaned'] = df['title'].copy()
        df['title_removed_content'] = ""
        
        # Apply title cleaning
        print("📖 Cleaning titles...")
        titles_cleaned = 0
        
        for idx, row in df.iterrows():
            if pd.notna(row['title']):
                cleaned_title, removed_content = self.clean_title_simple(row['title'])
                df.loc[idx, 'title_cleaned'] = cleaned_title
                df.loc[idx, 'title_removed_content'] = removed_content
                
                if removed_content:
                    titles_cleaned += 1
        
        print(f"✅ Title cleaning completed: {titles_cleaned:,} titles cleaned ({titles_cleaned/len(df)*100:.1f}%)")
        
        # Generate output filename
        if output_filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"romance_novels_basic_cleaned_{timestamp}.csv"
        
        output_path = self.output_dir / output_filename
        
        # Save cleaned dataset
        print(f"💾 Saving cleaned dataset to {output_path}")
        df.to_csv(output_path, index=False)
        
        # Print summary
        print(f"\n📊 Basic Cleaning Summary:")
        print(f"  - Total records: {len(df):,}")
        print(f"  - Titles cleaned: {titles_cleaned:,}")
        print(f"  - Output file: {output_path}")
        
        return str(output_path)

def main():
    """Run basic cleaning on the sample dataset."""
    print("🚀 Running Basic Romance Novel Cleaner")
    print("=" * 50)
    
    # Use the 500-book sample dataset that was just created
    input_path = "data/processed/final_books_2000_2020_en_cleaned_titles_sampled_500_20250901_221322.csv"
    
    if not Path(input_path).exists():
        print(f"❌ Input file not found: {input_path}")
        print("Please run the CSV builder first to generate the sample dataset.")
        return
    
    # Initialize and run cleaner
    cleaner = BasicRomanceCleaner(input_path)
    output_path = cleaner.clean_dataset()
    
    print(f"\n✅ Basic cleaning completed successfully!")
    print(f"📁 Output: {output_path}")

if __name__ == "__main__":
    main()
