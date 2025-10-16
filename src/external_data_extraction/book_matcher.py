"""
Book Matching Module

Matches author names and titles from external datasets (like BookRix) to existing
Goodreads metadata using both exact and fuzzy matching techniques.

Usage:
    python -m src.external_data_extraction.book_matcher
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import re
from difflib import SequenceMatcher
from rapidfuzz import fuzz, process
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a book matching operation."""
    bookrix_idx: int
    goodreads_idx: int
    match_type: str  # 'exact', 'fuzzy_author', 'fuzzy_title', 'fuzzy_both'
    confidence: float  # 0.0 to 1.0
    author_similarity: float
    title_similarity: float
    bookrix_author: str
    bookrix_title: str
    goodreads_author: str
    goodreads_title: str
    goodreads_work_id: int


class BookMatcher:
    """Matches books between external datasets and Goodreads metadata."""
    
    def __init__(
        self,
        goodreads_path: str,
        bookrix_path: str,
        fuzzy_threshold: float = 0.8,
        author_weight: float = 0.6,
        title_weight: float = 0.4
    ):
        """
        Initialize the book matcher.
        
        Args:
            goodreads_path: Path to Goodreads CSV file
            bookrix_path: Path to BookRix CSV file
            fuzzy_threshold: Minimum similarity for fuzzy matches
            author_weight: Weight for author similarity in combined score
            title_weight: Weight for title similarity in combined score
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.author_weight = author_weight
        self.title_weight = title_weight
        
        # Load datasets
        logger.info("Loading datasets...")
        self.goodreads_df = pd.read_csv(goodreads_path)
        self.bookrix_df = pd.read_csv(bookrix_path)
        
        logger.info(f"Loaded {len(self.goodreads_df)} Goodreads books")
        logger.info(f"Loaded {len(self.bookrix_df)} BookRix books")
        
        # Preprocess data
        self._preprocess_data()
    
    def _preprocess_data(self) -> None:
        """Preprocess data for better matching."""
        logger.info("Preprocessing data...")
        
        # Clean and normalize Goodreads data
        self.goodreads_df['author_clean'] = self.goodreads_df['author_name'].apply(self._clean_text)
        self.goodreads_df['title_clean'] = self.goodreads_df['title'].apply(self._clean_text)
        
        # Clean and normalize BookRix data
        self.bookrix_df['author_clean'] = self.bookrix_df['author_from_url'].apply(self._clean_text)
        self.bookrix_df['title_clean'] = self.bookrix_df['title_from_url'].apply(self._clean_text)
        
        # Create lookup dictionaries for faster matching
        self.goodreads_lookup = {
            'author': self.goodreads_df['author_clean'].tolist(),
            'title': self.goodreads_df['title_clean'].tolist(),
            'author_title': [
                f"{author}|||{title}" 
                for author, title in zip(
                    self.goodreads_df['author_clean'], 
                    self.goodreads_df['title_clean']
                )
            ]
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for matching."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common punctuation that might vary
        text = re.sub(r'[.,;:!?\'"()\[\]{}]', '', text)
        
        # Remove common words that might be inconsistent
        text = re.sub(r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', '', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Use rapidfuzz for better performance
        return fuzz.ratio(text1, text2) / 100.0
    
    def find_exact_matches(self) -> List[MatchResult]:
        """Find exact matches between datasets."""
        logger.info("Finding exact matches...")
        matches = []
        
        for bookrix_idx, bookrix_row in self.bookrix_df.iterrows():
            bookrix_author = bookrix_row['author_clean']
            bookrix_title = bookrix_row['title_clean']
            
            if not bookrix_author or not bookrix_title:
                continue
            
            # Look for exact matches
            exact_matches = self.goodreads_df[
                (self.goodreads_df['author_clean'] == bookrix_author) &
                (self.goodreads_df['title_clean'] == bookrix_title)
            ]
            
            for goodreads_idx, goodreads_row in exact_matches.iterrows():
                match = MatchResult(
                    bookrix_idx=bookrix_idx,
                    goodreads_idx=goodreads_idx,
                    match_type='exact',
                    confidence=1.0,
                    author_similarity=1.0,
                    title_similarity=1.0,
                    bookrix_author=bookrix_row['author_from_url'],
                    bookrix_title=bookrix_row['title_from_url'],
                    goodreads_author=goodreads_row['author_name'],
                    goodreads_title=goodreads_row['title'],
                    goodreads_work_id=goodreads_row['work_id']
                )
                matches.append(match)
        
        logger.info(f"Found {len(matches)} exact matches")
        return matches
    
    def find_fuzzy_matches(self, exclude_exact: bool = True) -> List[MatchResult]:
        """Find fuzzy matches between datasets."""
        logger.info("Finding fuzzy matches...")
        matches = []
        
        # Get exact matches to exclude if requested
        exact_matches = set()
        if exclude_exact:
            exact_results = self.find_exact_matches()
            exact_matches = {(r.bookrix_idx, r.goodreads_idx) for r in exact_results}
        
        for bookrix_idx, bookrix_row in self.bookrix_df.iterrows():
            bookrix_author = bookrix_row['author_clean']
            bookrix_title = bookrix_row['title_clean']
            
            if not bookrix_author or not bookrix_title:
                continue
            
            # Find best fuzzy matches for author
            author_matches = process.extract(
                bookrix_author, 
                self.goodreads_lookup['author'], 
                limit=10,
                scorer=fuzz.ratio
            )
            
            # Find best fuzzy matches for title
            title_matches = process.extract(
                bookrix_title, 
                self.goodreads_lookup['title'], 
                limit=10,
                scorer=fuzz.ratio
            )
            
            # Combine and score matches
            candidate_scores = {}
            
            for author_match, author_score, _ in author_matches:
                if author_score / 100.0 < self.fuzzy_threshold:
                    continue
                    
                goodreads_indices = self.goodreads_df[
                    self.goodreads_df['author_clean'] == author_match
                ].index.tolist()
                
                for goodreads_idx in goodreads_indices:
                    if exclude_exact and (bookrix_idx, goodreads_idx) in exact_matches:
                        continue
                    
                    goodreads_row = self.goodreads_df.loc[goodreads_idx]
                    title_score = self._calculate_similarity(
                        bookrix_title, 
                        goodreads_row['title_clean']
                    )
                    
                    if title_score < self.fuzzy_threshold:
                        continue
                    
                    # Calculate combined score
                    combined_score = (
                        (author_score / 100.0) * self.author_weight +
                        title_score * self.title_weight
                    )
                    
                    if combined_score >= self.fuzzy_threshold:
                        candidate_scores[(bookrix_idx, goodreads_idx)] = {
                            'author_score': author_score / 100.0,
                            'title_score': title_score,
                            'combined_score': combined_score
                        }
            
            # Add best matches
            for (b_idx, g_idx), scores in candidate_scores.items():
                goodreads_row = self.goodreads_df.loc[g_idx]
                bookrix_row = self.bookrix_df.loc[b_idx]
                
                # Determine match type
                if scores['author_score'] >= 0.9 and scores['title_score'] >= 0.9:
                    match_type = 'fuzzy_both'
                elif scores['author_score'] >= 0.9:
                    match_type = 'fuzzy_author'
                else:
                    match_type = 'fuzzy_title'
                
                match = MatchResult(
                    bookrix_idx=b_idx,
                    goodreads_idx=g_idx,
                    match_type=match_type,
                    confidence=scores['combined_score'],
                    author_similarity=scores['author_score'],
                    title_similarity=scores['title_score'],
                    bookrix_author=bookrix_row['author_from_url'],
                    bookrix_title=bookrix_row['title_from_url'],
                    goodreads_author=goodreads_row['author_name'],
                    goodreads_title=goodreads_row['title'],
                    goodreads_work_id=goodreads_row['work_id']
                )
                matches.append(match)
        
        # Remove duplicates and keep best matches
        matches = self._deduplicate_matches(matches)
        logger.info(f"Found {len(matches)} fuzzy matches")
        return matches
    
    def _deduplicate_matches(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Remove duplicate matches, keeping the best one for each BookRix book."""
        # Group by BookRix index
        matches_by_bookrix = {}
        for match in matches:
            if match.bookrix_idx not in matches_by_bookrix:
                matches_by_bookrix[match.bookrix_idx] = []
            matches_by_bookrix[match.bookrix_idx].append(match)
        
        # Keep best match for each BookRix book
        deduplicated = []
        for bookrix_idx, book_matches in matches_by_bookrix.items():
            # Sort by confidence and take the best
            best_match = max(book_matches, key=lambda x: x.confidence)
            deduplicated.append(best_match)
        
        return deduplicated
    
    def find_all_matches(self) -> List[MatchResult]:
        """Find all matches (exact + fuzzy)."""
        logger.info("Finding all matches...")
        
        # Get exact matches
        exact_matches = self.find_exact_matches()
        
        # Get fuzzy matches (excluding exact ones)
        fuzzy_matches = self.find_fuzzy_matches(exclude_exact=True)
        
        # Combine and sort by confidence
        all_matches = exact_matches + fuzzy_matches
        all_matches.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Total matches found: {len(all_matches)}")
        return all_matches
    
    def generate_report(self, matches: List[MatchResult]) -> Dict[str, Any]:
        """Generate a detailed matching report."""
        if not matches:
            return {
                "total_matches": 0,
                "exact_matches": 0,
                "fuzzy_matches": 0,
                "match_type_distribution": {},
                "confidence_stats": {
                    "average": 0.0,
                    "minimum": 0.0,
                    "maximum": 0.0
                },
                "coverage": {
                    "matched_bookrix_books": 0,
                    "total_bookrix_books": len(self.bookrix_df),
                    "coverage_percentage": 0.0
                }
            }
        
        # Basic statistics
        total_matches = len(matches)
        exact_matches = len([m for m in matches if m.match_type == 'exact'])
        fuzzy_matches = total_matches - exact_matches
        
        # Match type distribution
        match_types = {}
        for match in matches:
            match_types[match.match_type] = match_types.get(match.match_type, 0) + 1
        
        # Confidence distribution
        confidences = [m.confidence for m in matches]
        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        
        # Coverage statistics
        matched_bookrix = len(set(m.bookrix_idx for m in matches))
        total_bookrix = len(self.bookrix_df)
        coverage = matched_bookrix / total_bookrix if total_bookrix > 0 else 0
        
        report = {
            "total_matches": total_matches,
            "exact_matches": exact_matches,
            "fuzzy_matches": fuzzy_matches,
            "match_type_distribution": match_types,
            "confidence_stats": {
                "average": avg_confidence,
                "minimum": min_confidence,
                "maximum": max_confidence
            },
            "coverage": {
                "matched_bookrix_books": matched_bookrix,
                "total_bookrix_books": total_bookrix,
                "coverage_percentage": coverage * 100
            }
        }
        
        return report
    
    def save_matches(self, matches: List[MatchResult], output_path: str) -> None:
        """Save matches to CSV file."""
        logger.info(f"Saving matches to {output_path}")
        
        # Convert matches to DataFrame
        match_data = []
        for match in matches:
            match_data.append({
                'bookrix_idx': match.bookrix_idx,
                'goodreads_idx': match.goodreads_idx,
                'match_type': match.match_type,
                'confidence': match.confidence,
                'author_similarity': match.author_similarity,
                'title_similarity': match.title_similarity,
                'bookrix_author': match.bookrix_author,
                'bookrix_title': match.bookrix_title,
                'goodreads_author': match.goodreads_author,
                'goodreads_title': match.goodreads_title,
                'goodreads_work_id': match.goodreads_work_id,
                'bookrix_url': self.bookrix_df.iloc[match.bookrix_idx]['url']
            })
        
        matches_df = pd.DataFrame(match_data)
        matches_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(matches)} matches to {output_path}")


def test_subsample(
    goodreads_path: str,
    bookrix_path: str,
    goodreads_sample_size: int = 1000,
    bookrix_sample_size: int = 100,
    fuzzy_threshold: float = 0.8
) -> None:
    """Test matching on a subsample for faster iteration."""
    print(f"=== TESTING ON SUBSAMPLE ===")
    print(f"Goodreads sample size: {goodreads_sample_size}")
    print(f"BookRix sample size: {bookrix_sample_size}")
    print(f"Fuzzy threshold: {fuzzy_threshold}")
    print()
    
    # Load and sample datasets
    print("Loading datasets...")
    goodreads_df = pd.read_csv(goodreads_path)
    bookrix_df = pd.read_csv(bookrix_path)
    
    print(f"Original Goodreads size: {len(goodreads_df)}")
    print(f"Original BookRix size: {len(bookrix_df)}")
    
    # Sample datasets
    goodreads_sample = goodreads_df.sample(n=min(goodreads_sample_size, len(goodreads_df)), random_state=42)
    bookrix_sample = bookrix_df.sample(n=min(bookrix_sample_size, len(bookrix_df)), random_state=42)
    
    print(f"Sampled Goodreads size: {len(goodreads_sample)}")
    print(f"Sampled BookRix size: {len(bookrix_sample)}")
    print()
    
    # Save samples for testing
    goodreads_sample_path = "data/processed/goodreads_sample.csv"
    bookrix_sample_path = "data/processed/bookrix_sample.csv"
    
    goodreads_sample.to_csv(goodreads_sample_path, index=False)
    bookrix_sample.to_csv(bookrix_sample_path, index=False)
    
    print(f"Saved samples to:")
    print(f"  {goodreads_sample_path}")
    print(f"  {bookrix_sample_path}")
    print()
    
    # Initialize matcher with samples
    matcher = BookMatcher(
        goodreads_path=goodreads_sample_path,
        bookrix_path=bookrix_sample_path,
        fuzzy_threshold=fuzzy_threshold,
        author_weight=0.6,
        title_weight=0.4
    )
    
    # Debug: Show sample data after preprocessing
    print("\n=== SAMPLE DATA DEBUG ===")
    print("Sample BookRix authors:")
    for i, (_, row) in enumerate(matcher.bookrix_df.head(5).iterrows()):
        print(f"  {i+1}. \"{row['author_from_url']}\" -> \"{row['author_clean']}\"")
    
    print("\nSample BookRix titles:")
    for i, (_, row) in enumerate(matcher.bookrix_df.head(5).iterrows()):
        print(f"  {i+1}. \"{row['title_from_url']}\" -> \"{row['title_clean']}\"")
    
    print("\nSample Goodreads authors:")
    for i, (_, row) in enumerate(matcher.goodreads_df.head(5).iterrows()):
        print(f"  {i+1}. \"{row['author_name']}\" -> \"{row['author_clean']}\"")
    
    print("\nSample Goodreads titles:")
    for i, (_, row) in enumerate(matcher.goodreads_df.head(5).iterrows()):
        print(f"  {i+1}. \"{row['title']}\" -> \"{row['title_clean']}\"")
    
    # Find all matches
    print("\nFinding matches...")
    matches = matcher.find_all_matches()
    
    # Generate and print detailed report
    report = matcher.generate_report(matches)
    print("\n=== DETAILED MATCHING REPORT ===")
    print(f"Total matches: {report['total_matches']}")
    print(f"Exact matches: {report['exact_matches']}")
    print(f"Fuzzy matches: {report['fuzzy_matches']}")
    print(f"Coverage: {report['coverage']['coverage_percentage']:.1f}% ({report['coverage']['matched_bookrix_books']}/{report['coverage']['total_bookrix_books']})")
    print(f"Average confidence: {report['confidence_stats']['average']:.3f}")
    print(f"Confidence range: {report['confidence_stats']['minimum']:.3f} - {report['confidence_stats']['maximum']:.3f}")
    
    print("\nMatch type distribution:")
    for match_type, count in report['match_type_distribution'].items():
        percentage = (count / report['total_matches']) * 100 if report['total_matches'] > 0 else 0
        print(f"  {match_type}: {count} ({percentage:.1f}%)")
    
    # Show all matches with details
    print(f"\n=== ALL MATCHES ({len(matches)}) ===")
    for i, match in enumerate(matches, 1):
        print(f"{i:2d}. {match.match_type.upper()} (confidence: {match.confidence:.3f})")
        print(f"    Author similarity: {match.author_similarity:.3f}")
        print(f"    Title similarity: {match.title_similarity:.3f}")
        print(f"    BookRix: \"{match.bookrix_author}\" - \"{match.bookrix_title}\"")
        print(f"    Goodreads: \"{match.goodreads_author}\" - \"{match.goodreads_title}\"")
        print(f"    Goodreads ID: {match.goodreads_work_id}")
        print()
    
    # Show unmatched BookRix books
    matched_bookrix_indices = set(match.bookrix_idx for match in matches)
    unmatched_bookrix = bookrix_sample[~bookrix_sample.index.isin(matched_bookrix_indices)]
    
    print(f"=== UNMATCHED BOOKRIX BOOKS ({len(unmatched_bookrix)}) ===")
    for i, (_, row) in enumerate(unmatched_bookrix.head(10).iterrows(), 1):
        print(f"{i:2d}. \"{row['author_from_url']}\" - \"{row['title_from_url']}\"")
        print(f"    URL: {row['url']}")
        print()
    
    if len(unmatched_bookrix) > 10:
        print(f"... and {len(unmatched_bookrix) - 10} more unmatched books")
    
    # Save sample matches
    sample_output_path = "data/processed/book_matches_sample.csv"
    matcher.save_matches(matches, sample_output_path)
    print(f"Saved sample matches to: {sample_output_path}")


def main():
    """Main function to run the book matching process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Match BookRix books to Goodreads metadata")
    parser.add_argument("--test", action="store_true", help="Run on subsample first")
    parser.add_argument("--goodreads-sample", type=int, default=1000, help="Goodreads sample size for testing")
    parser.add_argument("--bookrix-sample", type=int, default=100, help="BookRix sample size for testing")
    parser.add_argument("--threshold", type=float, default=0.8, help="Fuzzy matching threshold")
    args = parser.parse_args()
    
    # File paths
    goodreads_path = "data/processed/romance_books_main_final_canonicalized.csv"
    bookrix_path = "data/processed/romance-books.with_author_title.csv"
    output_path = "data/processed/book_matches.csv"
    
    if args.test:
        # Run on subsample
        test_subsample(
            goodreads_path=goodreads_path,
            bookrix_path=bookrix_path,
            goodreads_sample_size=args.goodreads_sample,
            bookrix_sample_size=args.bookrix_sample,
            fuzzy_threshold=args.threshold
        )
    else:
        # Run on full dataset
        print("=== RUNNING ON FULL DATASET ===")
        print("This may take several minutes...")
        print("Use --test flag to run on subsample first")
        print()
        
        # Initialize matcher
        matcher = BookMatcher(
            goodreads_path=goodreads_path,
            bookrix_path=bookrix_path,
            fuzzy_threshold=args.threshold,
            author_weight=0.6,
            title_weight=0.4
        )
        
        # Find all matches
        matches = matcher.find_all_matches()
        
        # Generate and print report
        report = matcher.generate_report(matches)
        print("\n=== MATCHING REPORT ===")
        print(f"Total matches: {report['total_matches']}")
        print(f"Exact matches: {report['exact_matches']}")
        print(f"Fuzzy matches: {report['fuzzy_matches']}")
        print(f"Coverage: {report['coverage']['coverage_percentage']:.1f}%")
        print(f"Average confidence: {report['confidence_stats']['average']:.3f}")
        
        print("\nMatch type distribution:")
        for match_type, count in report['match_type_distribution'].items():
            print(f"  {match_type}: {count}")
        
        # Save matches
        matcher.save_matches(matches, output_path)
        
        # Show sample matches
        print(f"\n=== SAMPLE MATCHES ===")
        for i, match in enumerate(matches[:5], 1):
            print(f"{i}. {match.match_type.upper()} (confidence: {match.confidence:.3f})")
            print(f"   BookRix: \"{match.bookrix_author}\" - \"{match.bookrix_title}\"")
            print(f"   Goodreads: \"{match.goodreads_author}\" - \"{match.goodreads_title}\"")
            print()


if __name__ == "__main__":
    main()
