"""
Simple Title Cleaner for Romance Novel Data
Only strips brackets and parentheses from titles - minimal cleaning.
"""

import re
from typing import Tuple

class SimpleTitleCleaner:
    """
    Simple utility class for cleaning book titles by removing bracket content.
    """
    
    def __init__(self):
        # Simple pattern to remove content within brackets/parentheses
        self.bracket_pattern = re.compile(r'\s*\([^)]*\)|\s*\[[^\]]*\]')
    
    def clean_title(self, title: str) -> Tuple[str, str]:
        """
        Clean a title by removing content within brackets/parentheses.
        
        Args:
            title: The original title
            
        Returns:
            Tuple of (cleaned_title, removed_content)
        """
        if not title or not isinstance(title, str):
            return title, ""
        
        original_title = title.strip()
        cleaned_title = original_title
        
        # Find all bracket content
        removed_parts = []
        for match in self.bracket_pattern.finditer(original_title):
            removed_parts.append(match.group(0))
            cleaned_title = cleaned_title.replace(match.group(0), '')
        
        # Clean up extra whitespace
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title)  # Multiple spaces to single
        cleaned_title = cleaned_title.strip()
        
        # If cleaning resulted in empty title, return original
        if not cleaned_title:
            return original_title, ""
        
        removed_content = '; '.join(removed_parts) if removed_parts else ""
        
        return cleaned_title, removed_content
    
    def has_brackets(self, title: str) -> bool:
        """
        Check if a title contains brackets/parentheses.
        
        Args:
            title: The title to check
            
        Returns:
            True if the title contains brackets
        """
        if not title or not isinstance(title, str):
            return False
        
        return bool(self.bracket_pattern.search(title))


def test_simple_title_cleaner():
    """Test the simple title cleaner with sample titles."""
    
    cleaner = SimpleTitleCleaner()
    
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
        "Twilight (Twilight Series, #1)",
        "Book Title [Special Edition]",
        "Another Book (Hardcover) [Limited]"
    ]
    
    print("🧪 Testing Simple Title Cleaner...")
    print("=" * 60)
    
    for title in test_titles:
        cleaned_title, removed_content = cleaner.clean_title(title)
        has_brackets = cleaner.has_brackets(title)
        
        print(f"Original: {title}")
        print(f"Cleaned:  {cleaned_title}")
        if removed_content:
            print(f"Removed:  {removed_content}")
        print(f"Has brackets: {has_brackets}")
        print("-" * 40)


if __name__ == "__main__":
    test_simple_title_cleaner()
