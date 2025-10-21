#!/usr/bin/env python3
"""
Book Download Research Component - Download Monitor
Monitor and analyze download progress, success rates, and system status
"""

import pandas as pd
import os
import sys
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DownloadMonitor:
    """Monitor and analyze download progress"""
    
    def __init__(self, download_dir: str):
        """
        Initialize download monitor
        
        Args:
            download_dir: Directory containing downloaded books and progress files
        """
        self.download_dir = Path(download_dir)
        self.progress_file = self.download_dir / "download_progress.json"
        self.results_dir = self.download_dir / "production_results"
        
        logger.info(f"Download Monitor initialized for: {download_dir}")
    
    def get_current_status(self) -> Dict:
        """Get current download status and statistics"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'progress_file_exists': self.progress_file.exists(),
            'downloads_directory': str(self.download_dir),
            'total_files': 0,
            'epub_files': 0,
            'mobi_files': 0,
            'other_files': 0,
            'progress_data': None,
            'recent_downloads': [],
            'success_rate': 0.0,
            'daily_stats': {}
        }
        
        # Count files in download directory
        if self.download_dir.exists():
            for file_path in self.download_dir.iterdir():
                if file_path.is_file():
                    status['total_files'] += 1
                    if file_path.suffix.lower() == '.epub':
                        status['epub_files'] += 1
                    elif file_path.suffix.lower() == '.mobi':
                        status['mobi_files'] += 1
                    else:
                        status['other_files'] += 1
        
        # Load progress data
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                status['progress_data'] = progress_data
                
                # Calculate success rate
                total_processed = progress_data.get('total_processed', 0)
                total_downloaded = progress_data.get('total_downloaded', 0)
                if total_processed > 0:
                    status['success_rate'] = (total_downloaded / total_processed) * 100
                
                # Get recent downloads
                download_history = progress_data.get('download_history', [])
                status['recent_downloads'] = download_history[-10:]  # Last 10 downloads
                
                # Daily stats
                status['daily_stats'] = {
                    'daily_downloads': progress_data.get('daily_downloads', 0),
                    'last_run_date': progress_data.get('last_run_date', 'Never'),
                    'total_processed': total_processed,
                    'total_downloaded': total_downloaded,
                    'total_failed': progress_data.get('total_failed', 0)
                }
                
            except Exception as e:
                logger.error(f"Error loading progress data: {e}")
        
        return status
    
    def analyze_download_patterns(self) -> Dict:
        """Analyze download patterns and success rates"""
        if not self.progress_file.exists():
            return {'error': 'No progress file found'}
        
        try:
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            download_history = progress_data.get('download_history', [])
            if not download_history:
                return {'error': 'No download history found'}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(download_history)
            
            # Basic statistics
            total_books = len(df)
            successful_downloads = len(df[df['status'] == 'downloaded'])
            failed_downloads = len(df[df['status'] == 'failed'])
            
            # Success rate by author
            author_success = df.groupby('author_name')['status'].apply(
                lambda x: (x == 'downloaded').sum() / len(x) * 100
            ).sort_values(ascending=False)
            
            # Success rate by publication year
            df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
            year_success = df.groupby('publication_year')['status'].apply(
                lambda x: (x == 'downloaded').sum() / len(x) * 100
            ).sort_values(ascending=False)
            
            # Error analysis
            error_analysis = df[df['status'] == 'failed']['error'].value_counts()
            
            # Time analysis
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            hourly_success = df.groupby('hour')['status'].apply(
                lambda x: (x == 'downloaded').sum() / len(x) * 100
            )
            
            analysis = {
                'total_books_processed': total_books,
                'successful_downloads': successful_downloads,
                'failed_downloads': failed_downloads,
                'overall_success_rate': (successful_downloads / total_books * 100) if total_books > 0 else 0,
                'top_successful_authors': author_success.head(10).to_dict(),
                'top_successful_years': year_success.head(10).to_dict(),
                'common_errors': error_analysis.head(5).to_dict(),
                'hourly_success_rate': hourly_success.to_dict(),
                'recent_trend': self._calculate_recent_trend(df)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing download patterns: {e}")
            return {'error': str(e)}
    
    def _calculate_recent_trend(self, df: pd.DataFrame) -> Dict:
        """Calculate recent download trend"""
        if len(df) < 10:
            return {'trend': 'insufficient_data'}
        
        # Get last 10 downloads
        recent_df = df.tail(10)
        recent_success_rate = (recent_df['status'] == 'downloaded').sum() / len(recent_df) * 100
        
        # Get previous 10 downloads
        if len(df) >= 20:
            previous_df = df.iloc[-20:-10]
            previous_success_rate = (previous_df['status'] == 'downloaded').sum() / len(previous_df) * 100
            
            trend_direction = 'improving' if recent_success_rate > previous_success_rate else 'declining'
            trend_magnitude = abs(recent_success_rate - previous_success_rate)
        else:
            trend_direction = 'insufficient_data'
            trend_magnitude = 0
        
        return {
            'recent_success_rate': recent_success_rate,
            'trend_direction': trend_direction,
            'trend_magnitude': trend_magnitude
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive download report"""
        status = self.get_current_status()
        analysis = self.analyze_download_patterns()
        
        report = []
        report.append("=" * 80)
        report.append("BOOK DOWNLOAD SYSTEM - COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Current Status
        report.append("CURRENT STATUS")
        report.append("-" * 40)
        report.append(f"Download Directory: {status['downloads_directory']}")
        report.append(f"Total Files: {status['total_files']}")
        report.append(f"  - EPUB Files: {status['epub_files']}")
        report.append(f"  - MOBI Files: {status['mobi_files']}")
        report.append(f"  - Other Files: {status['other_files']}")
        report.append(f"Progress File: {'✓ Found' if status['progress_file_exists'] else '✗ Missing'}")
        report.append("")
        
        # Progress Statistics
        if status['progress_data']:
            daily_stats = status['daily_stats']
            report.append("PROGRESS STATISTICS")
            report.append("-" * 40)
            report.append(f"Total Processed: {daily_stats['total_processed']}")
            report.append(f"Total Downloaded: {daily_stats['total_downloaded']}")
            report.append(f"Total Failed: {daily_stats['total_failed']}")
            report.append(f"Success Rate: {status['success_rate']:.1f}%")
            report.append(f"Daily Downloads: {daily_stats['daily_downloads']}")
            report.append(f"Last Run: {daily_stats['last_run_date']}")
            report.append("")
        
        # Analysis Results
        if 'error' not in analysis:
            report.append("DOWNLOAD ANALYSIS")
            report.append("-" * 40)
            report.append(f"Overall Success Rate: {analysis['overall_success_rate']:.1f}%")
            report.append(f"Books Processed: {analysis['total_books_processed']}")
            report.append("")
            
            # Top successful authors
            if analysis['top_successful_authors']:
                report.append("TOP SUCCESSFUL AUTHORS")
                report.append("-" * 40)
                for author, rate in list(analysis['top_successful_authors'].items())[:5]:
                    report.append(f"  {author}: {rate:.1f}%")
                report.append("")
            
            # Common errors
            if analysis['common_errors']:
                report.append("COMMON ERRORS")
                report.append("-" * 40)
                for error, count in list(analysis['common_errors'].items())[:5]:
                    report.append(f"  {error}: {count} times")
                report.append("")
            
            # Recent trend
            if 'recent_trend' in analysis and analysis['recent_trend'].get('trend_direction') != 'insufficient_data':
                trend = analysis['recent_trend']
                report.append("RECENT TREND")
                report.append("-" * 40)
                report.append(f"Recent Success Rate: {trend.get('recent_success_rate', 0):.1f}%")
                report.append(f"Trend: {trend.get('trend_direction', 'unknown')}")
                if trend.get('trend_magnitude', 0) > 0:
                    report.append(f"Change: {trend['trend_magnitude']:.1f} percentage points")
                report.append("")
        
        # Recent Downloads
        if status['recent_downloads']:
            report.append("RECENT DOWNLOADS")
            report.append("-" * 40)
            for download in status['recent_downloads'][-5:]:
                status_icon = "✓" if download['status'] == 'downloaded' else "✗"
                report.append(f"  {status_icon} {download['title']} by {download['author_name']}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        if status['success_rate'] < 50:
            report.append("  ⚠️  Low success rate - consider investigating search terms or MCP server")
        if status['epub_files'] == 0:
            report.append("  ⚠️  No EPUB files found - check download functionality")
        if not status['progress_file_exists']:
            report.append("  ⚠️  No progress file - system may not be running properly")
        
        if status['success_rate'] > 70:
            report.append("  ✓ Good success rate - system is working well")
        if status['epub_files'] > 0:
            report.append("  ✓ EPUB files found - downloads are working")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = None):
        """Save the report to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"download_report_{timestamp}.txt"
        
        report_path = self.results_dir / filename
        self.results_dir.mkdir(exist_ok=True)
        
        report = self.generate_report()
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to: {report_path}")
        return report_path

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor book download progress')
    parser.add_argument('--download-dir',
                       default='organized_outputs/anna_archive_download',
                       help='Directory containing downloaded books')
    parser.add_argument('--save-report', action='store_true',
                       help='Save report to file')
    parser.add_argument('--show-analysis', action='store_true',
                       help='Show detailed analysis')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = DownloadMonitor(args.download_dir)
    
    # Generate and display report
    report = monitor.generate_report()
    print(report)
    
    # Save report if requested
    if args.save_report:
        monitor.save_report()
    
    # Show detailed analysis if requested
    if args.show_analysis:
        analysis = monitor.analyze_download_patterns()
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS")
        print("=" * 80)
        print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()
