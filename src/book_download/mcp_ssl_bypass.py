#!/usr/bin/env python3
"""
MCP SSL Bypass and Alternative Solutions
Investigate and implement solutions for Anna's Archive access issues
"""

import os
import sys
import logging
import subprocess
import urllib.request
import ssl
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnnaArchiveAccess:
    """Handle Anna's Archive access with SSL bypass and alternatives"""
    
    def __init__(self):
        """Initialize Anna's Archive access handler"""
        self.original_domains = [
            "https://annas-archive.org",
            "https://annas-archive.se"
        ]
        self.alternative_domains = [
            "https://annas-archive.net",
            "https://annas-archive.com",
            "https://annas-archive.info"
        ]
        
    def test_ssl_bypass(self, url: str) -> Dict:
        """Test SSL bypass for a given URL"""
        logger.info(f"Testing SSL bypass for: {url}")
        
        try:
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Make request with SSL bypass
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
            
            with urllib.request.urlopen(request, context=ssl_context, timeout=10) as response:
                content = response.read().decode('utf-8')
                
                return {
                    'success': True,
                    'status_code': response.status,
                    'content_length': len(content),
                    'redirected': response.url != url,
                    'final_url': response.url,
                    'content_preview': content[:500] if content else ''
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def test_mcp_with_ssl_bypass(self) -> Dict:
        """Test MCP server with SSL bypass environment"""
        logger.info("Testing MCP server with SSL bypass environment")
        
        # Set environment variables for SSL bypass
        env = os.environ.copy()
        env['ANNAS_SECRET_KEY'] = 'BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP'
        env['ANNAS_DOWNLOAD_PATH'] = '/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download'
        
        # Add SSL bypass environment variables
        env['PYTHONHTTPSVERIFY'] = '0'
        env['CURL_INSECURE'] = '1'
        
        test_terms = ["romance", "Jane Austen", "fiction"]
        results = {}
        
        for term in test_terms:
            try:
                result = subprocess.run([
                    '/home/polina/.local/bin/annas-mcp', 'search', term
                ], capture_output=True, text=True, timeout=30, env=env)
                
                results[term] = {
                    'returncode': result.returncode,
                    'stdout': result.stdout.strip(),
                    'stderr': result.stderr.strip(),
                    'success': result.returncode == 0 and result.stdout.strip() != "No books found."
                }
                
            except Exception as e:
                results[term] = {'error': str(e)}
        
        return results
    
    def test_alternative_domains(self) -> Dict:
        """Test alternative Anna's Archive domains"""
        logger.info("Testing alternative Anna's Archive domains")
        
        results = {}
        
        # Test original domains
        for domain in self.original_domains:
            results[domain] = self.test_ssl_bypass(domain)
        
        # Test alternative domains
        for domain in self.alternative_domains:
            results[domain] = self.test_ssl_bypass(domain)
        
        return results
    
    def check_anna_dl_alternative(self) -> Dict:
        """Check if anna-dl alternative tool is available"""
        logger.info("Checking for anna-dl alternative tool")
        
        try:
            # Check if anna-dl is installed
            result = subprocess.run(['anna-dl', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return {
                    'available': True,
                    'version': 'unknown',
                    'help_output': result.stdout[:500]
                }
            else:
                return {
                    'available': False,
                    'error': result.stderr
                }
                
        except FileNotFoundError:
            return {
                'available': False,
                'error': 'anna-dl not found in PATH'
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def generate_solution_report(self) -> str:
        """Generate a comprehensive solution report"""
        logger.info("Generating solution report")
        
        # Test all approaches
        ssl_bypass_results = self.test_alternative_domains()
        mcp_ssl_results = self.test_mcp_with_ssl_bypass()
        anna_dl_check = self.check_anna_dl_alternative()
        
        report = []
        report.append("=" * 80)
        report.append("ANNA'S ARCHIVE ACCESS SOLUTION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # SSL Bypass Results
        report.append("SSL BYPASS TEST RESULTS")
        report.append("-" * 40)
        for domain, result in ssl_bypass_results.items():
            if result['success']:
                report.append(f"✓ {domain}: Accessible (Status: {result['status_code']})")
                if result['redirected']:
                    report.append(f"  → Redirected to: {result['final_url']}")
                if 'content_preview' in result and result['content_preview']:
                    preview = result['content_preview'][:100].replace('\n', ' ')
                    report.append(f"  → Content preview: {preview}...")
            else:
                report.append(f"✗ {domain}: Failed ({result['error_type']}: {result['error']})")
        report.append("")
        
        # MCP SSL Bypass Results
        report.append("MCP SERVER WITH SSL BYPASS")
        report.append("-" * 40)
        successful_searches = sum(1 for r in mcp_ssl_results.values() if r.get('success', False))
        total_searches = len(mcp_ssl_results)
        report.append(f"Successful searches: {successful_searches}/{total_searches}")
        
        for term, result in mcp_ssl_results.items():
            if result.get('success'):
                report.append(f"✓ {term}: Found results")
            else:
                report.append(f"✗ {term}: {result.get('stdout', result.get('error', 'Unknown error'))}")
        report.append("")
        
        # Anna-dl Alternative
        report.append("ANNA-DL ALTERNATIVE TOOL")
        report.append("-" * 40)
        if anna_dl_check['available']:
            report.append("✓ anna-dl is available")
            report.append(f"Help output: {anna_dl_check['help_output'][:200]}...")
        else:
            report.append("✗ anna-dl is not available")
            report.append(f"Error: {anna_dl_check['error']}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        working_domains = [d for d, r in ssl_bypass_results.items() if r['success']]
        if working_domains:
            report.append("1. Use working domains with SSL bypass:")
            for domain in working_domains:
                report.append(f"   - {domain}")
        else:
            report.append("1. No working domains found - Anna's Archive may be blocked")
        
        if successful_searches > 0:
            report.append("2. MCP server works with SSL bypass - configure environment")
        else:
            report.append("2. MCP server still not working - consider alternatives")
        
        if anna_dl_check['available']:
            report.append("3. Use anna-dl as alternative to MCP server")
        else:
            report.append("3. Install anna-dl as alternative: pip install anna-dl")
        
        report.append("4. Consider using existing downloaded books for testing")
        report.append("5. Implement fallback system for when Anna's Archive is unavailable")
        report.append("")
        
        # Next Steps
        report.append("NEXT STEPS")
        report.append("-" * 40)
        if working_domains:
            report.append("1. Configure MCP server to use working domains")
            report.append("2. Set up SSL bypass environment variables")
            report.append("3. Test with small batch of books")
        else:
            report.append("1. Use existing downloaded books for system testing")
            report.append("2. Implement alternative book sources")
            report.append("3. Set up monitoring for when Anna's Archive becomes available")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main function"""
    logger.info("Starting Anna's Archive access investigation...")
    
    access_handler = AnnaArchiveAccess()
    report = access_handler.generate_solution_report()
    
    print(report)
    
    # Save report
    report_file = "anna_archive_solution_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Solution report saved to: {report_file}")

if __name__ == "__main__":
    main()
