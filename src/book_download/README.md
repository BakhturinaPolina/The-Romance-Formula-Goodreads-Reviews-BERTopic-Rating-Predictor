# Anna's Archive Book Download

## Quick Start
```bash
export ANNAS_SECRET_KEY="BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP"
python3 anna_api_client.py "MD5_HASH" --download
```

## Files
- `anna_api_client.py` - Main API client (use this)
- `mcp_integration.py` - MCP server integration
- `download_manager.py` - Batch processing
- `example_usage.py` - Usage examples

## Key Points
- **Tor required** - Anna's Archive only accessible through Tor
- **MD5 needed** - Downloads require file MD5 hash, not title/author
- **Daily limit** - 25 downloads per day
- **Working** - Tested and functional with your API key

## Usage
```python
from anna_api_client import AnnaAPIClient
client = AnnaAPIClient()
result = client.download_book("MD5_HASH", "filename.epub")
```
