# Anna's Archive Repository Exploration Guide

## Executive Summary

This document provides a systematic approach to explore Anna's Archive's GitLab repository and identify working API endpoints for your book download project. Based on research and testing, here are the key findings and next steps.

## Current Status

### ✅ What's Working
- **MCP Integration**: Your `annas-mcp` binary is properly installed and executable
- **API Key**: Your API key (`BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP`) is valid and accepted
- **Environment Setup**: MCP server configuration is correct

### ⚠️ Current Issues
- **Search Results**: MCP search is returning "No books found" for common titles
- **API Endpoints**: Need to investigate alternative endpoints and search methods

## Key Findings from Research

### 1. Official API Structure
Based on research, Anna's Archive provides:
- **Fast Download API**: `/dyn/api/fast_download.json` (for members)
- **Search Functionality**: Available through various endpoints
- **Authentication**: Uses API keys for member access

### 2. Repository Structure
The Anna's Archive GitLab repository contains:
- **`/api/` Directory**: Core API functionalities
  - `search.py`: Search functionality
  - `auth.py`: Authentication mechanisms
  - `routes.py`: API endpoint definitions
- **`/docs/` Directory**: Documentation and setup instructions
- **`/frontend/` Directory**: Frontend code showing API usage patterns

### 3. Alternative Access Methods
- **Unofficial API Wrappers**: Community-developed libraries
- **SearXNG Integration**: Search engine integration
- **Direct Database Access**: ElasticSearch and MariaDB dumps

## Systematic Exploration Strategy

### Phase 1: Repository Access and Overview
1. **Visit Repository**: https://software.annas-archive.li/AnnaArchivist/annas-archive/-/tree/main
2. **Read README.md**: Understand project structure and setup
3. **Check Directory Structure**: Identify `/api/`, `/docs/`, `/frontend/` directories

### Phase 2: API Deep Dive
1. **Explore `/api/` Directory**:
   - Look for `routes.py` or `endpoints.py`
   - Examine `search.py` for search implementation
   - Check `auth.py` for authentication methods
   - Review recent commits for API changes

2. **Analyze Frontend Code**:
   - Check how the website performs searches
   - Identify API calls and URL patterns
   - Understand request/response formats

### Phase 3: Recent Changes Investigation
1. **Check Commit History**:
   - Look for commits mentioning "API", "search", "disabled", "removed"
   - Review security-related changes
   - Check authentication updates

2. **Identify Alternative Endpoints**:
   - Look for internal or admin APIs
   - Check for member-only endpoints
   - Find configuration examples

### Phase 4: Testing and Validation
1. **Test Discovered Endpoints**:
   - Use your API key to test new endpoints
   - Document working search parameters
   - Validate download functionality

2. **Alternative Methods**:
   - Test unofficial API wrappers
   - Explore SearXNG integration
   - Consider direct database access

## Immediate Next Steps

### 1. Repository Exploration
```bash
# Clone or browse the repository
git clone https://software.annas-archive.li/AnnaArchivist/annas-archive.git
cd annas-archive

# Explore key directories
ls -la api/
ls -la docs/
ls -la frontend/
```

### 2. API Endpoint Discovery
```bash
# Search for API-related files
find . -name "*.py" -exec grep -l "search\|api\|endpoint" {} \;

# Look for route definitions
grep -r "route\|endpoint" api/
grep -r "search" api/
```

### 3. Recent Changes Analysis
```bash
# Check recent commits
git log --oneline --since="3 months ago" | grep -i "api\|search\|auth"

# Look for specific changes
git log --grep="search" --oneline
git log --grep="api" --oneline
```

### 4. Alternative Testing
```bash
# Test different search parameters
ANNAS_SECRET_KEY="BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP" /home/polina/.local/bin/annas-mcp search "romance"
ANNAS_SECRET_KEY="BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP" /home/polina/.local/bin/annas-mcp search "fiction"
```

## What to Look For

### Working Endpoints
- Alternative search endpoints beyond the current MCP implementation
- Direct API endpoints that might bypass MCP limitations
- Member-only or internal APIs with different access patterns

### Authentication Methods
- Different API key formats or requirements
- Session-based authentication
- Alternative authentication headers or parameters

### Search Parameters
- Different search term formats
- Additional search filters or parameters
- Alternative search algorithms or endpoints

### Recent Changes
- API endpoint modifications or removals
- Authentication system updates
- Search algorithm changes

## Troubleshooting Current Issues

### MCP Search Not Returning Results
**Possible Causes**:
1. **Search Algorithm Changes**: Recent updates to search functionality
2. **API Endpoint Changes**: MCP might be using outdated endpoints
3. **Search Parameter Format**: Different search term formatting required
4. **Database Issues**: Temporary database or indexing problems

**Investigation Steps**:
1. Check MCP server logs for detailed error messages
2. Test with different search term formats
3. Verify API key permissions and scope
4. Compare with working examples from repository

### Alternative Approaches
1. **Direct API Calls**: Bypass MCP and call Anna's Archive API directly
2. **Web Scraping**: Use the website's search interface programmatically
3. **Database Access**: Download and query their databases directly
4. **Unofficial Wrappers**: Use community-developed API clients

## Expected Outcomes

After completing this exploration, you should have:
1. **Working API Endpoints**: Functional search and download endpoints
2. **Proper Authentication**: Correct API key usage and authentication methods
3. **Search Parameters**: Optimal search term formatting and parameters
4. **Alternative Methods**: Backup approaches if primary methods fail
5. **Documentation**: Complete understanding of the API structure

## Next Actions

1. **Start Repository Exploration**: Begin with Phase 1 of the systematic approach
2. **Document Findings**: Keep detailed notes of all discoveries
3. **Test Alternatives**: Try different approaches as you discover them
4. **Update Integration**: Modify your MCP integration based on findings
5. **Share Results**: Document successful approaches for future reference

This exploration should resolve the current "No books found" issue and provide you with robust access to Anna's Archive's book database for your romance novel research project.
