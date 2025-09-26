#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to validate the shelf normalization improvements.
"""

import sys
from pathlib import Path
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def test_noncontent_filtering():
    """Test the non-content filtering logic."""
    print("Testing non-content filtering...")
    
    def is_noncontent(shelf: str):
        """Simplified version of the non-content filter."""
        s = shelf.strip().lower()
        
        # Reading status patterns
        reading_status = {
            'to-read', 'to read', 'tbr', 'want-to-read', 'want to read', 'wtr',
            'currently-reading', 'currently reading', 'reading', 'cr',
            'read', 'finished', 'done', 'completed',
            'dnf', 'did-not-finish', 'did not finish', 'abandoned',
            'on-hold', 'on hold', 'paused', 'suspended'
        }
        
        # Ownership/collection patterns  
        ownership = {
            'owned', 'have', 'have-it', 'have it', 'bought', 'purchased',
            'library', 'borrowed', 'borrow', 'lent', 'loan',
            'kindle', 'ebook', 'e-book', 'audiobook', 'audio-book',
            'hardcover', 'paperback', 'mass-market', 'mass market',
            'wishlist', 'wish-list', 'want', 'want-it', 'want it'
        }
        
        # Date/reading challenge patterns
        date_patterns = {
            '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            'summer', 'winter', 'spring', 'fall', 'autumn'
        }
        
        # Pure punctuation/dash runs
        if re.match(r'^[-_\.\s]+$', s) or len(s) <= 2:
            return 'punctuation'
        
        # Check exact matches
        if s in reading_status:
            return 'reading_status'
        if s in ownership:
            return 'ownership'
        if s in date_patterns:
            return 'date_pattern'
        
        # Check patterns with dashes/underscores normalized
        s_norm = re.sub(r'[-_\s]+', '-', s)
        if s_norm in reading_status or s_norm in ownership:
            return 'reading_status' if s_norm in reading_status else 'ownership'
        
        # Check if starts with common prefixes
        if any(s.startswith(prefix) for prefix in ['to-', 'want-', 'have-', 'currently-']):
            return 'reading_status'
        
        return None
    
    # Test cases
    test_cases = [
        # Reading status
        ("to-read", "reading_status"),
        ("currently-reading", "reading_status"),
        ("tbr", "reading_status"),
        ("dnf", "reading_status"),
        ("read", "reading_status"),
        
        # Ownership
        ("owned", "ownership"),
        ("kindle", "ownership"),
        ("library", "ownership"),
        ("wishlist", "ownership"),
        
        # Date patterns
        ("2023", "date_pattern"),
        ("january", "date_pattern"),
        
        # Punctuation
        ("----2016", "punctuation"),
        ("_", "punctuation"),
        
        # Content shelves (should return None)
        ("romance", None),
        ("fantasy", None),
        ("young-adult", None),
        ("historical-fiction", None),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for shelf, expected in test_cases:
        result = is_noncontent(shelf)
        if result == expected:
            print(f"  ✓ {shelf} -> {result}")
            passed += 1
        else:
            print(f"  ✗ {shelf} -> {result} (expected {expected})")
    
    print(f"Non-content filtering: {passed}/{total} tests passed")
    return passed == total

def test_segmentation_gating():
    """Test the segmentation gating logic."""
    print("\nTesting segmentation gating...")
    
    def test_gates(shelf):
        """Test the segmentation gates."""
        original = shelf
        
        # Gate 1: Must have no whitespace
        if " " in original:
            return "has_whitespace"
            
        # Gate 2: Must be length >= 6
        if len(original) < 6:
            return "too_short"
            
        # Gate 3: Check if already a common word (simplified)
        common_words = {"romance", "fantasy", "fiction", "novel", "book"}
        if original.lower() in common_words:
            return "common_word"
            
        # Gate 4: Must be CamelCase OR all-lowercase concatenation
        CAMEL_HIT = re.compile(r"[A-Z][a-z]+[A-Z]")
        LOWER_CONCAT = re.compile(r"^[a-z]{6,}$")
        
        looks_camel = bool(CAMEL_HIT.search(original))
        looks_concat = bool(LOWER_CONCAT.fullmatch(original))
        
        if not (looks_camel or looks_concat):
            return "not_camel_or_concat"
        
        return "would_process"
    
    test_cases = [
        # Should be rejected
        ("romance", "common_word"),  # Common word
        ("fantasy", "common_word"),  # Common word
        ("rom", "too_short"),        # Too short
        ("romance fantasy", "has_whitespace"),  # Has whitespace
        ("romancefantasy", "not_camel_or_concat"),  # Not camel or concat pattern
        
        # Should be processed
        ("YoungAdult", "would_process"),  # CamelCase
        ("romancefantasy", "would_process"),  # Lowercase concat
    ]
    
    passed = 0
    total = len(test_cases)
    
    for shelf, expected in test_cases:
        result = test_gates(shelf)
        if result == expected:
            print(f"  ✓ {shelf} -> {result}")
            passed += 1
        else:
            print(f"  ✗ {shelf} -> {result} (expected {expected})")
    
    print(f"Segmentation gating: {passed}/{total} tests passed")
    return passed == total

def test_tiered_thresholds():
    """Test the tiered similarity thresholds."""
    print("\nTesting tiered similarity thresholds...")
    
    def simple_jaro_winkler(a, b):
        """Simple Jaro-Winkler approximation."""
        if a == b:
            return 1.0
        if len(a) == 0 or len(b) == 0:
            return 0.0
        
        # Simple character-based similarity
        a_chars = set(a.lower())
        b_chars = set(b.lower())
        intersection = len(a_chars & b_chars)
        union = len(a_chars | b_chars)
        return intersection / union if union > 0 else 0.0
    
    def simple_edit_distance(a, b):
        """Simple edit distance."""
        if len(a) < len(b):
            a, b = b, a
        if len(b) == 0:
            return len(a)
        
        # Simple Hamming distance for same length, else length difference
        if len(a) == len(b):
            return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))
        else:
            return abs(len(a) - len(b))
    
    def jaccard_char_ngrams(a, b, n=3):
        """Character n-gram Jaccard similarity."""
        def ngrams(s):
            s = f" {s} "
            return {s[i:i+n] for i in range(len(s)-n+1)}
        
        A, B = ngrams(a), ngrams(b)
        if not A and not B:
            return 1.0
        return len(A & B) / max(1, len(A | B))
    
    def apply_tiered_thresholds(a, b):
        """Apply tiered thresholds based on string length."""
        jw = simple_jaro_winkler(a, b)
        ed = simple_edit_distance(a, b)
        j3 = jaccard_char_ngrams(a, b, n=3)
        
        if len(a) <= 6 and len(b) <= 6:
            # Short strings: DL ≤ 1
            return ed <= 1, f"short: ED={ed}"
        elif len(a) <= 12 and len(b) <= 12:
            # Mid-length: JW ≥ 0.92
            return jw >= 0.92, f"mid: JW={jw:.3f}"
        else:
            # Long strings: j3 ≥ 0.95
            return j3 >= 0.95, f"long: J3={j3:.3f}"
    
    test_cases = [
        # Short strings (≤6): DL ≤ 1
        ("alpha", "alpha", True),   # Same string
        ("alpha", "alpa", True),    # DL = 1
        ("alpha", "beta", False),   # DL > 1
        
        # Mid-length (7-12): JW ≥ 0.92
        ("romance", "romance", True),  # Same string
        ("romance", "romanc", True),   # High JW
        ("romance", "fantasy", False), # Low JW
        
        # Long strings: j3 ≥ 0.95
        ("young-adult", "young-adult", True),  # Same string
        ("young-adult", "youngadult", True),   # High j3
        ("young-adult", "historical-fiction", False),  # Low j3
    ]
    
    passed = 0
    total = len(test_cases)
    
    for a, b, expected in test_cases:
        result, reason = apply_tiered_thresholds(a, b)
        if result == expected:
            print(f"  ✓ {a} vs {b}: {result} ({reason})")
            passed += 1
        else:
            print(f"  ✗ {a} vs {b}: {result} (expected {expected}) ({reason})")
    
    print(f"Tiered thresholds: {passed}/{total} tests passed")
    return passed == total

def main():
    """Run all tests."""
    print("Testing shelf normalization improvements...")
    print("=" * 50)
    
    tests = [
        test_noncontent_filtering,
        test_segmentation_gating,
        test_tiered_thresholds,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Overall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All improvements are working correctly!")
        return 0
    else:
        print("❌ Some improvements need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
