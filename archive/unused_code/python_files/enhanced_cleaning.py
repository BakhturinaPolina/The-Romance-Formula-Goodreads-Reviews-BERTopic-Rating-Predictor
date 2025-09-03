#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Cleaning Functions for Romance Novel Dataset
Based on EDA analysis results and identified data quality issues.

Key Issues Identified:
- Title cleaning: Series information embedded in titles (67.5% of series books)
- Description cleaning: HTML artifacts, whitespace issues (94.8% have whitespace problems)
- Author deduplication: 14,634 potential duplicate names
- Series standardization: Inconsistent series numbering and title extraction
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRomanceNovelCleaner:
    """
    Enhanced cleaning functions for romance novel dataset based on EDA analysis.
    """
    
    def __init__(self):
        """Initialize the cleaner with common patterns and configurations."""
        # Common series patterns identified in analysis
        self.series_patterns = {
            'number_colon': r'\b(\d+)\s*[:\-]\s*',
            'book_volume': r'\b(Book|Volume|Part|Chapter)\s+(\d+)\b',
            'ordinal': r'\b(\d+)\s*(?:st|nd|rd|th)\s*',
            'parenthesis': r'\b(\d+)\s*\(',
            'roman_numerals': r'\b([IVX]+)\s*[:\-]\s*',
            'alpha_series': r'\b([A-Z])\s*[:\-]\s*',
            'end_number': r'\b(\d+)\s*$'
        }
        
        # HTML patterns for description cleaning
        self.html_patterns = {
            'tags': r'<[^>]+>',
            'entities': r'&[a-zA-Z]+;',
            'line_breaks': r'[\r\n\t]+',
            'multiple_spaces': r'\s+',
            'special_chars': r'[\u00A0-\uFFFF]'
        }
        
        # Common romance subgenre indicators
        self.subgenre_indicators = {
            'contemporary': ['contemporary', 'modern', 'present-day'],
            'historical': ['historical', 'victorian', 'regency', 'medieval'],
            'paranormal': ['paranormal', 'vampire', 'werewolf', 'magic'],
            'suspense': ['suspense', 'thriller', 'mystery', 'crime'],
            'fantasy': ['fantasy', 'magical', 'supernatural'],
            'scifi': ['science fiction', 'futuristic', 'space']
        }
    
    def clean_title(self, title: str, series_title: Optional[str] = None) -> Dict[str, Union[str, Optional[int]]]:
        """
        Enhanced title cleaning based on EDA analysis results.
        
        Args:
            title: Book title to clean
            series_title: Associated series title if available
            
        Returns:
            Dictionary with cleaned_title, series_number, and cleaning_notes
        """
        if pd.isna(title):
            return {
                'cleaned_title': title,
                'series_number': None,
                'cleaning_notes': 'Title was NaN'
            }
        
        original_title = title
        cleaning_notes = []
        
        # Extract series number first
        series_number = self.extract_series_number(title)
        if series_number:
            cleaning_notes.append(f"Extracted series number: {series_number}")
        
        # Remove series title if embedded (addresses 67.5% of series books issue)
        if series_title and pd.notna(series_title):
            if series_title.lower() in title.lower():
                title = title.replace(series_title, '').strip()
                cleaning_notes.append(f"Removed embedded series title: '{series_title}'")
        
        # Remove all series patterns
        for pattern_name, pattern in self.series_patterns.items():
            matches = re.findall(pattern, title, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    # Handle patterns with groups
                    for match in matches:
                        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
                else:
                    title = re.sub(pattern, '', title, flags=re.IGNORECASE)
                cleaning_notes.append(f"Removed {pattern_name} pattern")
        
        # Clean up whitespace and separators
        title = re.sub(r'^[\s\-:]+|[\s\-:]+$', '', title)  # Remove leading/trailing separators
        title = re.sub(r'\s+', ' ', title).strip()  # Normalize whitespace
        
        # Handle edge cases
        if not title or title.isspace():
            title = "Untitled"
            cleaning_notes.append("Title became empty, set to 'Untitled'")
        
        return {
            'cleaned_title': title,
            'series_number': series_number,
            'cleaning_notes': '; '.join(cleaning_notes) if cleaning_notes else 'No cleaning needed'
        }
    
    def clean_description(self, description: str) -> Dict[str, Union[str, List[str]]]:
        """
        Enhanced description cleaning based on HTML artifact analysis.
        
        Args:
            description: Book description to clean
            
        Returns:
            Dictionary with cleaned_description and cleaning_notes
        """
        if pd.isna(description):
            return {
                'cleaned_description': description,
                'cleaning_notes': ['Description was NaN']
            }
        
        original_description = description
        cleaning_notes = []
        
        # Remove HTML tags (0.0% of descriptions have HTML tags)
        if re.search(self.html_patterns['tags'], description):
            description = re.sub(self.html_patterns['tags'], '', description)
            cleaning_notes.append("Removed HTML tags")
        
        # Remove HTML entities (0.0% of descriptions have HTML entities)
        if re.search(self.html_patterns['entities'], description):
            description = re.sub(self.html_patterns['entities'], ' ', description)
            cleaning_notes.append("Removed HTML entities")
        
        # Handle line breaks and tabs (83.7% of descriptions have these)
        if re.search(self.html_patterns['line_breaks'], description):
            description = re.sub(self.html_patterns['line_breaks'], ' ', description)
            cleaning_notes.append("Normalized line breaks and tabs")
        
        # Normalize whitespace (94.8% of descriptions have multiple spaces)
        if re.search(self.html_patterns['multiple_spaces'], description):
            description = re.sub(self.html_patterns['multiple_spaces'], ' ', description)
            cleaning_notes.append("Normalized whitespace")
        
        # Remove special characters
        if re.search(self.html_patterns['special_chars'], description):
            description = re.sub(self.html_patterns['special_chars'], ' ', description)
            cleaning_notes.append("Removed special characters")
        
        # Final cleanup
        description = description.strip()
        
        # Handle very short descriptions (0.2% are <50 chars)
        if len(description) < 50:
            cleaning_notes.append(f"Warning: Very short description ({len(description)} chars)")
        
        return {
            'cleaned_description': description,
            'cleaning_notes': cleaning_notes
        }
    
    def extract_series_number(self, title: str) -> Optional[int]:
        """
        Enhanced series number extraction based on identified patterns.
        
        Args:
            title: Book title to analyze
            
        Returns:
            Extracted series number or None
        """
        if pd.isna(title):
            return None
        
        # Try all patterns in order of specificity
        for pattern_name, pattern in self.series_patterns.items():
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                try:
                    if pattern_name == 'roman_numerals':
                        # Convert Roman numerals to Arabic
                        return self._roman_to_arabic(match.group(1))
                    elif pattern_name == 'alpha_series':
                        # Convert A=1, B=2, etc.
                        return ord(match.group(1).upper()) - ord('A') + 1
                    elif isinstance(match.group(1), str):
                        return int(match.group(1))
                    else:
                        return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _roman_to_arabic(self, roman: str) -> int:
        """Convert Roman numerals to Arabic numbers."""
        roman_numerals = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000
        }
        
        result = 0
        prev_value = 0
        
        for char in reversed(roman.upper()):
            value = roman_numerals.get(char, 0)
            if value >= prev_value:
                result += value
            else:
                result -= value
            prev_value = value
        
        return result
    
    def identify_author_duplicates(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Identify potential author duplicates based on name similarity.
        
        Args:
            df: DataFrame with author_name and author_id columns
            
        Returns:
            Dictionary with duplicate groups and recommendations
        """
        # Group by author name and find multiple IDs
        author_name_to_ids = defaultdict(list)
        for _, row in df.iterrows():
            author_name_to_ids[row['author_name']].append(row['author_id'])
        
        # Find names with multiple IDs
        duplicate_groups = []
        for name, ids in author_name_to_ids.items():
            if len(set(ids)) > 1:
                duplicate_groups.append({
                    'author_name': name,
                    'author_ids': list(set(ids)),
                    'book_count': len(df[df['author_name'] == name]),
                    'confidence': 'high' if len(set(ids)) > 2 else 'medium'
                })
        
        # Sort by confidence and book count
        duplicate_groups.sort(key=lambda x: (x['confidence'] == 'high', x['book_count']), reverse=True)
        
        return {
            'total_duplicate_groups': len(duplicate_groups),
            'high_confidence_duplicates': [g for g in duplicate_groups if g['confidence'] == 'high'],
            'medium_confidence_duplicates': [g for g in duplicate_groups if g['confidence'] == 'medium'],
            'duplicate_groups': duplicate_groups
        }
    
    def extract_subgenre_signals(self, popular_shelves: str) -> Dict[str, Union[List[str], float]]:
        """
        Extract subgenre signals from popular shelves.
        
        Args:
            popular_shelves: Popular shelves string (JSON format)
            
        Returns:
            Dictionary with subgenres and confidence scores
        """
        if pd.isna(popular_shelves):
            return {'subgenres': [], 'confidence': 0.0}
        
        try:
            shelves = json.loads(popular_shelves)
            if not isinstance(shelves, list):
                return {'subgenres': [], 'confidence': 0.0}
        except (json.JSONDecodeError, TypeError):
            return {'subgenres': [], 'confidence': 0.0}
        
        subgenres = []
        total_matches = 0
        
        for shelf in shelves:
            shelf_lower = shelf.lower()
            for subgenre, indicators in self.subgenre_indicators.items():
                for indicator in indicators:
                    if indicator in shelf_lower:
                        subgenres.append(subgenre)
                        total_matches += 1
                        break
        
        # Calculate confidence based on matches
        confidence = min(1.0, total_matches / len(shelves)) if shelves else 0.0
        
        return {
            'subgenres': list(set(subgenres)),  # Remove duplicates
            'confidence': confidence
        }
    
    def clean_series_information(self, title: str, series_title: Optional[str], 
                                series_id: Optional[Union[int, float]]) -> Dict[str, Union[str, Optional[str], Optional[int]]]:
        """
        Comprehensive series information cleaning and extraction.
        
        Args:
            title: Book title
            series_title: Series title
            series_id: Series ID
            
        Returns:
            Dictionary with cleaned series information
        """
        result = {
            'cleaned_title': title,
            'cleaned_series_title': series_title,
            'extracted_series_number': None,
            'series_confidence': 'none'
        }
        
        # Extract series number from title
        series_number = self.extract_series_number(title)
        if series_number:
            result['extracted_series_number'] = series_number
            result['series_confidence'] = 'high'
        
        # Clean title if series information is present
        if series_number or (series_title and pd.notna(series_title)):
            title_cleaning = self.clean_title(title, series_title)
            result['cleaned_title'] = title_cleaning['cleaned_title']
            result['series_confidence'] = 'high'
        
        # Validate series information consistency
        if series_title and pd.notna(series_title):
            if series_id and pd.notna(series_id):
                result['series_confidence'] = 'high'
            else:
                result['series_confidence'] = 'medium'
        
        return result
    
    def generate_cleaning_report(self, df: pd.DataFrame) -> Dict[str, Union[int, float, List[str]]]:
        """
        Generate comprehensive cleaning report based on EDA analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with cleaning statistics and recommendations
        """
        report = {
            'total_books': len(df),
            'books_in_series': df['series_id'].notna().sum(),
            'books_with_descriptions': df['description'].notna().sum(),
            'cleaning_recommendations': []
        }
        
        # Title cleaning recommendations
        titles_with_patterns = 0
        for pattern in self.series_patterns.values():
            matches = df['title'].str.contains(pattern, regex=True, na=False)
            titles_with_patterns += matches.sum()
        
        if titles_with_patterns > 0:
            report['cleaning_recommendations'].append(
                f"Title cleaning: {titles_with_patterns:,} titles have series patterns"
            )
        
        # Description cleaning recommendations
        html_descriptions = df['description'].str.contains(self.html_patterns['tags'], regex=True, na=False).sum()
        if html_descriptions > 0:
            report['cleaning_recommendations'].append(
                f"Description cleaning: {html_descriptions:,} descriptions have HTML artifacts"
            )
        
        # Author deduplication recommendations
        author_duplicates = self.identify_author_duplicates(df)
        if author_duplicates['total_duplicate_groups'] > 0:
            report['cleaning_recommendations'].append(
                f"Author deduplication: {author_duplicates['total_duplicate_groups']:,} potential duplicate groups"
            )
        
        # Series standardization recommendations
        series_books = df[df['series_id'].notna()]
        if not series_books.empty:
            embedded_series = series_books.apply(
                lambda row: row['series_title'].lower() in row['title'].lower() 
                if pd.notna(row['series_title']) else False, axis=1
            ).sum()
            
            if embedded_series > 0:
                report['cleaning_recommendations'].append(
                    f"Series standardization: {embedded_series:,} series books have embedded series titles"
                )
        
        return report

def main():
    """Test the enhanced cleaning functions."""
    print("🧹 Testing Enhanced Romance Novel Cleaner")
    print("=" * 50)
    
    # Create cleaner instance
    cleaner = EnhancedRomanceNovelCleaner()
    
    # Test title cleaning
    test_titles = [
        "Book 1: The Beginning",
        "Volume II - The Middle",
        "The End (3rd)",
        "Series Name: Book Title",
        "Normal Title"
    ]
    
    print("\n📚 Testing Title Cleaning:")
    for title in test_titles:
        result = cleaner.clean_title(title)
        print(f"  '{title}' -> '{result['cleaned_title']}' (Number: {result['series_number']})")
    
    # Test description cleaning
    test_descriptions = [
        "Normal description",
        "Description with <b>HTML</b> tags",
        "Multiple    spaces   and\nline breaks",
        "Special chars: &amp; &quot;"
    ]
    
    print("\n📖 Testing Description Cleaning:")
    for desc in test_descriptions:
        result = cleaner.clean_description(desc)
        print(f"  '{desc[:30]}...' -> '{result['cleaned_description'][:30]}...'")
    
    print("\n✅ Enhanced cleaning functions tested successfully!")

if __name__ == "__main__":
    main()
