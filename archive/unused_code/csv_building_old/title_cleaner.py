"""
Title Cleaner Utility for Romance Novel Data
Removes series information and other metadata from book titles.
"""

import re
from typing import Optional, Tuple

class TitleCleaner:
    """
    Utility class for cleaning book titles by removing series information.
    """
    
    def __init__(self):
        # Patterns to identify and remove series information
        self.series_patterns = [
            # (Series Name, #1) or (Series Name #1)
            r'\s*\([^)]*#[0-9]+[^)]*\)',
            r'\s*\([^)]*[Ss]eries[^)]*\)',
            r'\s*\([^)]*[Bb]ook\s+[0-9]+[^)]*\)',
            r'\s*\([^)]*[Vv]olume\s+[0-9]+[^)]*\)',
            r'\s*\([^)]*[Pp]art\s+[0-9]+[^)]*\)',
            
            # Standalone series indicators
            r'\s*#[0-9]+(?:\s*$|\s*[^a-zA-Z])',  # #1, #2, etc. (but not #1st, #2nd)
            r'\s*[Bb]ook\s+[0-9]+(?:\s*$|\s*[^a-zA-Z])',
            r'\s*[Vv]olume\s+[0-9]+(?:\s*$|\s*[^a-zA-Z])',
            r'\s*[Pp]art\s+[0-9]+(?:\s*$|\s*[^a-zA-Z])',
            
            # Common series suffixes
            r'\s*[Nn]ovella(?:\s*$|\s*[^a-zA-Z])',
            r'\s*[Ss]eries(?:\s*$|\s*[^a-zA-Z])',
            
            # Publisher series indicators
            r'\s*\([^)]*[Pp]ublisher[^)]*\)',
            r'\s*\([^)]*[Ii]mprint[^)]*\)',
            
            # Edition indicators that aren't part of the title
            r'\s*\([^)]*[Ee]dition[^)]*\)',
            r'\s*\([^)]*[Aa]bridged[^)]*\)',
            r'\s*\([^)]*[Uu]nabridged[^)]*\)',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.series_patterns]
    
    def clean_title(self, title: str) -> Tuple[str, str]:
        """
        Clean a title by removing series information.
        
        Args:
            title: The original title
            
        Returns:
            Tuple of (cleaned_title, removed_series_info)
        """
        if not title or not isinstance(title, str):
            return title, ""
        
        original_title = title.strip()
        cleaned_title = original_title
        removed_parts = []
        
        # Apply each pattern to remove series information
        for pattern in self.compiled_patterns:
            matches = pattern.findall(cleaned_title)
            if matches:
                for match in matches:
                    cleaned_title = cleaned_title.replace(match, '')
                    removed_parts.append(match.strip())
        
        # Clean up extra whitespace and punctuation
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title)  # Multiple spaces to single
        cleaned_title = re.sub(r'\s*[,\s]*$', '', cleaned_title)  # Remove trailing commas/spaces
        cleaned_title = re.sub(r'^\s*[,\s]*', '', cleaned_title)  # Remove leading commas/spaces
        cleaned_title = cleaned_title.strip()
        
        # If cleaning resulted in empty title, return original
        if not cleaned_title:
            return original_title, ""
        
        removed_series_info = '; '.join(removed_parts) if removed_parts else ""
        
        return cleaned_title, removed_series_info
    
    def is_series_title(self, title: str) -> bool:
        """
        Check if a title contains series information.
        
        Args:
            title: The title to check
            
        Returns:
            True if the title contains series information
        """
        if not title or not isinstance(title, str):
            return False
        
        for pattern in self.compiled_patterns:
            if pattern.search(title):
                return True
        
        return False
    
    def extract_series_info(self, title: str) -> dict:
        """
        Extract series information from a title.
        
        Args:
            title: The title to analyze
            
        Returns:
            Dictionary with extracted series information
        """
        if not title or not isinstance(title, str):
            return {}
        
        series_info = {
            'has_series': False,
            'series_name': None,
            'series_number': None,
            'series_type': None,
            'cleaned_title': title,
            'removed_parts': []
        }
        
        # Check for series patterns
        if self.is_series_title(title):
            series_info['has_series'] = True
            
            # Extract series number
            number_match = re.search(r'#([0-9]+)', title, re.IGNORECASE)
            if number_match:
                series_info['series_number'] = int(number_match.group(1))
            
            # Extract series type
            if re.search(r'[Bb]ook\s+[0-9]+', title):
                series_info['series_type'] = 'book'
            elif re.search(r'[Vv]olume\s+[0-9]+', title):
                series_info['series_type'] = 'volume'
            elif re.search(r'[Pp]art\s+[0-9]+', title):
                series_info['series_type'] = 'part'
            elif re.search(r'[Nn]ovella', title):
                series_info['series_type'] = 'novella'
            
            # Clean the title
            cleaned_title, removed_parts = self.clean_title(title)
            series_info['cleaned_title'] = cleaned_title
            series_info['removed_parts'] = [p for p in removed_parts if p]
            
            # Try to extract series name from parentheses
            series_name_match = re.search(r'\(([^)]*#[0-9]+[^)]*)\)', title)
            if series_name_match:
                series_name = series_name_match.group(1)
                # Remove the number part
                series_name = re.sub(r'#[0-9]+', '', series_name).strip()
                if series_name:
                    series_info['series_name'] = series_name
        
        return series_info
    
    def batch_clean_titles(self, titles: list) -> list:
        """
        Clean a batch of titles.
        
        Args:
            titles: List of titles to clean
            
        Returns:
            List of tuples (cleaned_title, removed_series_info)
        """
        return [self.clean_title(title) for title in titles]
    
    def get_cleaning_stats(self, titles: list) -> dict:
        """
        Get statistics about title cleaning.
        
        Args:
            titles: List of titles to analyze
            
        Returns:
            Dictionary with cleaning statistics
        """
        if not titles:
            return {}
        
        total_titles = len(titles)
        titles_with_series = sum(1 for title in titles if self.is_series_title(title))
        cleaned_titles = [self.clean_title(title)[0] for title in titles]
        unique_cleaned = len(set(cleaned_titles))
        
        return {
            'total_titles': total_titles,
            'titles_with_series': titles_with_series,
            'titles_without_series': total_titles - titles_with_series,
            'series_percentage': (titles_with_series / total_titles) * 100 if total_titles > 0 else 0,
            'unique_cleaned_titles': unique_cleaned,
            'duplicate_titles_after_cleaning': total_titles - unique_cleaned
        }


def test_title_cleaner():
    """Test the title cleaner with sample titles."""
    
    cleaner = TitleCleaner()
    
    test_titles = [
        "Prowled Darkness (Dante's Circle, #7)",
        "Guardian Cougar (Finding Fatherhood, #2)",
        "A Kitty in the Lion's Den (Sweet Water, #3)",
        "Healer's Touch (Hearts And Thrones, #4)",
        "Emma",
        "Pride and Prejudice",
        "The Hunger Games #1",
        "Harry Potter and the Sorcerer's Stone (Book 1)",
        "The Fellowship of the Ring (Volume 1)",
        "Game of Thrones: Part 1",
        "Playmaker: A Venom Series Novella",
        "Twilight (Twilight Series, #1)"
    ]
    
    print("🧪 Testing Title Cleaner...")
    print("=" * 60)
    
    for title in test_titles:
        cleaned_title, removed_info = cleaner.clean_title(title)
        series_info = cleaner.extract_series_info(title)
        
        print(f"Original: {title}")
        print(f"Cleaned:  {cleaned_title}")
        if removed_info:
            print(f"Removed:  {removed_info}")
        if series_info['has_series']:
            print(f"Series:   {series_info['series_name']} #{series_info['series_number']} ({series_info['series_type']})")
        print("-" * 40)
    
    # Get batch statistics
    stats = cleaner.get_cleaning_stats(test_titles)
    print(f"\n📊 Batch Cleaning Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_title_cleaner()
