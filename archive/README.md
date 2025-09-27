# Archive Directory

This directory contains modules that were moved from the active codebase.

## Archived Files

### annas_torrent_client.py
- **Reason**: Contains only mock/placeholder implementations
- **Original location**: `src/corpus_creation/annas_torrent_client.py`
- **Status**: Mock implementation with no real torrent functionality
- **Dependencies**: Would require actual torrent client integration

### annas_mcp_client.py  
- **Reason**: Contains only mock/placeholder implementations
- **Original location**: `src/corpus_creation/annas_mcp_client.py`
- **Status**: Mock implementation with no real MCP tool integration
- **Dependencies**: Would require actual annas-mcp binary installation

## Notes

These modules were archived because they contained only placeholder code that would never produce real downloads. The free pipeline now gracefully handles their absence by disabling torrent and MCP features when these modules are not available.

If you need to implement real torrent or MCP functionality in the future, these files can serve as a starting point for the actual implementation.
