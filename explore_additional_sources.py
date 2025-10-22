#!/usr/bin/env python3
"""
Script to explore and integrate additional data sources:
- LibGen Fiction
- Project Gutenberg
- Other book databases
"""

import requests
import json
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import time

def explore_project_gutenberg():
    """Explore Project Gutenberg catalog and metadata"""
    print("=== EXPLORING PROJECT GUTENBERG ===")
    
    # Project Gutenberg catalog URLs
    catalog_urls = [
        "https://www.gutenberg.org/ebooks/",
        "https://www.gutenberg.org/ebooks/search/?sort_order=downloads",
        "https://www.gutenberg.org/ebooks/search/?query=romance",
        "https://www.gutenberg.org/ebooks/search/?query=fiction"
    ]
    
    print("Project Gutenberg provides:")
    print("- Over 60,000 free eBooks")
    print("- Primarily public domain works")
    print("- Multiple download formats (HTML, EPUB, Kindle, etc.)")
    print("- RDF/XML metadata available")
    print("- FTP access for bulk downloads")
    
    # Check if we can access their catalog
    try:
        response = requests.get("https://www.gutenberg.org/ebooks/", timeout=10)
        if response.status_code == 200:
            print("✅ Project Gutenberg catalog is accessible")
        else:
            print(f"❌ Project Gutenberg catalog returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Error accessing Project Gutenberg: {e}")
    
    return {
        "name": "Project Gutenberg",
        "size": "60,000+ books",
        "focus": "Public domain literature",
        "formats": ["HTML", "EPUB", "Kindle", "Plain text"],
        "access": "Free, open access",
        "metadata": "RDF/XML format available"
    }

def explore_libgen_fiction():
    """Explore LibGen Fiction database"""
    print("\n=== EXPLORING LIBGEN FICTION ===")
    
    print("LibGen Fiction provides:")
    print("- Large collection of fiction books")
    print("- Multiple mirrors available")
    print("- Various formats (PDF, EPUB, etc.)")
    print("- Metadata includes titles, authors, ISBNs")
    
    # Note: LibGen access requires careful consideration of legal issues
    print("⚠️  Note: LibGen access requires compliance with local laws")
    print("⚠️  Consider legal implications before integration")
    
    return {
        "name": "LibGen Fiction",
        "size": "Large collection (exact size varies)",
        "focus": "Fiction books",
        "formats": ["PDF", "EPUB", "MOBI", "DJVU"],
        "access": "Mirror sites",
        "metadata": "Titles, authors, ISBNs, file hashes",
        "legal_note": "Requires legal compliance"
    }

def explore_other_sources():
    """Explore other potential book data sources"""
    print("\n=== EXPLORING OTHER SOURCES ===")
    
    sources = [
        {
            "name": "Internet Archive Books",
            "description": "Digital library with millions of books",
            "access": "https://archive.org/details/texts",
            "metadata": "Rich metadata, multiple formats"
        },
        {
            "name": "HathiTrust",
            "description": "Academic library consortium",
            "access": "https://www.hathitrust.org/",
            "metadata": "Academic metadata, limited access"
        },
        {
            "name": "Google Books",
            "description": "Large book database with metadata",
            "access": "Google Books API",
            "metadata": "Comprehensive metadata, API access"
        },
        {
            "name": "WorldCat",
            "description": "Global library catalog",
            "access": "WorldCat API",
            "metadata": "Library holdings, comprehensive metadata"
        }
    ]
    
    for source in sources:
        print(f"\n📚 {source['name']}:")
        print(f"   Description: {source['description']}")
        print(f"   Access: {source['access']}")
        print(f"   Metadata: {source['metadata']}")
    
    return sources

def create_integration_plan():
    """Create a plan for integrating additional data sources"""
    print("\n=== INTEGRATION PLAN ===")
    
    plan = {
        "phase_1": {
            "name": "Complete OpenLibrary Loading",
            "description": "Load full 11.8M record OpenLibrary dataset",
            "estimated_time": "4-6 hours",
            "priority": "High"
        },
        "phase_2": {
            "name": "Project Gutenberg Integration",
            "description": "Download and integrate Project Gutenberg catalog",
            "estimated_time": "2-3 hours",
            "priority": "High",
            "legal_status": "Safe (public domain)"
        },
        "phase_3": {
            "name": "LibGen Fiction Research",
            "description": "Research legal access to LibGen Fiction data",
            "estimated_time": "1-2 hours",
            "priority": "Medium",
            "legal_status": "Requires review"
        },
        "phase_4": {
            "name": "Unified Search Interface",
            "description": "Create unified search across all sources",
            "estimated_time": "3-4 hours",
            "priority": "High"
        }
    }
    
    for phase, details in plan.items():
        print(f"\n{phase.upper().replace('_', ' ')}:")
        print(f"   Name: {details['name']}")
        print(f"   Description: {details['description']}")
        print(f"   Time: {details['estimated_time']}")
        print(f"   Priority: {details['priority']}")
        if 'legal_status' in details:
            print(f"   Legal: {details['legal_status']}")
    
    return plan

def main():
    """Main function to explore all additional sources"""
    print("🔍 EXPLORING ADDITIONAL DATA SOURCES FOR TITLE MATCHING SYSTEM")
    print("=" * 70)
    
    # Explore each source
    gutenberg_info = explore_project_gutenberg()
    libgen_info = explore_libgen_fiction()
    other_sources = explore_other_sources()
    
    # Create integration plan
    integration_plan = create_integration_plan()
    
    print("\n" + "=" * 70)
    print("📋 SUMMARY OF FINDINGS")
    print("=" * 70)
    
    print(f"\n✅ RECOMMENDED NEXT STEPS:")
    print("1. Complete OpenLibrary loading (11.8M records)")
    print("2. Integrate Project Gutenberg (60K+ public domain books)")
    print("3. Research LibGen Fiction legal access")
    print("4. Create unified search interface")
    
    print(f"\n📊 POTENTIAL SYSTEM SIZE:")
    print(f"- Current: 18,758+ books")
    print(f"- + OpenLibrary: 11,800,000+ books")
    print(f"- + Project Gutenberg: 60,000+ books")
    print(f"- + LibGen Fiction: TBD (research needed)")
    print(f"- Total Potential: 11,878,758+ books")

if __name__ == "__main__":
    main()
