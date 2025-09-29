#!/bin/bash
# Create Virtual Environment Shim for ebook-convert
# This script creates a shim in the virtual environment so ebook-convert
# is automatically available when the venv is activated

set -e

echo "=== Creating Virtual Environment Shim for ebook-convert ==="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if we're in the project directory
PROJECT_ROOT="/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research"
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Error: Project directory not found: $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

# Check if virtual environment exists
VENV_PATH="./venv"
if [ ! -d "$VENV_PATH" ]; then
    log_warning "Virtual environment not found at $VENV_PATH"
    log_info "Creating virtual environment..."
    python3 -m venv venv
    log_success "Virtual environment created"
fi

# Create the shim
log_info "Creating ebook-convert shim in virtual environment..."

cat > "$VENV_PATH/bin/ebook-convert" <<'EOF'
#!/usr/bin/env bash
# Calibre ebook-convert shim for virtual environment
# Prefers user Calibre; falls back to flatpak wrapper

if command -v ebook-convert.real >/dev/null 2>&1; then
    exec ebook-convert.real "$@"
elif command -v ~/calibre-bin/calibre/ebook-convert >/dev/null 2>&1; then
    exec ~/calibre-bin/calibre/ebook-convert "$@"
elif command -v flatpak >/dev/null 2>&1; then
    exec flatpak run --command=ebook-convert com.calibre_ebook.calibre "$@"
else
    echo "ebook-convert not found. Install Calibre (isolated or flatpak)." >&2
    echo "Run: ./setup_calibre_user.sh" >&2
    exit 1
fi
EOF

chmod +x "$VENV_PATH/bin/ebook-convert"
log_success "Virtual environment shim created"

# Test the shim
log_info "Testing the shim..."
if [ -x "$VENV_PATH/bin/ebook-convert" ]; then
    log_success "Shim is executable"
    
    # Try to run it (it might fail if Calibre isn't installed yet)
    if "$VENV_PATH/bin/ebook-convert" --version >/dev/null 2>&1; then
        log_success "Shim is working - ebook-convert found and functional"
    else
        log_warning "Shim created but ebook-convert not found yet"
        log_info "Install Calibre first: ./setup_calibre_user.sh"
    fi
else
    log_warning "Shim creation failed"
    exit 1
fi

echo ""
echo "=== Shim Creation Complete ==="
echo "The virtual environment now includes an ebook-convert shim."
echo ""
echo "To use it:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Install Calibre: ./setup_calibre_user.sh"
echo "  3. Test: ebook-convert --version"
echo ""
echo "The shim will automatically find Calibre whether it's installed via:"
echo "  - Isolated install (~/calibre-bin)"
echo "  - Flatpak (com.calibre_ebook.calibre)"
echo "  - System package manager"
