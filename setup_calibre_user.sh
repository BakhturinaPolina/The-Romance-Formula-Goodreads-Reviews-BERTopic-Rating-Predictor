#!/bin/bash
# Calibre User Installation Script (No Sudo Required)
# Implements isolated install, Flatpak fallback, and libxcb-cursor0 vendoring
# Based on Calibre's official Linux installation documentation

set -e

echo "=== Calibre User Installation (No Sudo) ==="
echo "This script will install Calibre without requiring root privileges"
echo ""

# Colors for output
RED='\033[0;31m'
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

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the project directory
PROJECT_ROOT="/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research"
if [ ! -d "$PROJECT_ROOT" ]; then
    log_error "Project directory not found: $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if ebook-convert is working
test_ebook_convert() {
    local binary="$1"
    if command_exists "$binary"; then
        if "$binary" --version >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to install libxcb-cursor0 locally
install_libxcb_cursor_locally() {
    log_info "Installing libxcb-cursor0 locally (no sudo required)..."
    
    # Create local lib directory
    mkdir -p ~/.local/libxcb-cursor0
    
    # Download the package
    if command_exists apt; then
        log_info "Downloading libxcb-cursor0 package..."
        apt download libxcb-cursor0 2>/dev/null || {
            log_warning "Could not download libxcb-cursor0 via apt"
            return 1
        }
        
        # Extract to local directory
        log_info "Extracting libxcb-cursor0..."
        dpkg-deb -x libxcb-cursor0_*.deb ~/.local/libxcb-cursor0/ 2>/dev/null || {
            log_warning "Could not extract libxcb-cursor0 package"
            return 1
        }
        
        # Clean up downloaded package
        rm -f libxcb-cursor0_*.deb
        
        # Set up LD_LIBRARY_PATH
        local lib_path="$HOME/.local/libxcb-cursor0/usr/lib/$(uname -m)-linux-gnu"
        if [ -d "$lib_path" ]; then
            log_success "libxcb-cursor0 installed locally"
            echo "export LD_LIBRARY_PATH=\"$lib_path:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
            export LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH"
            return 0
        else
            log_warning "libxcb-cursor0 library path not found"
            return 1
        fi
    else
        log_warning "apt not available - cannot install libxcb-cursor0 locally"
        return 1
    fi
}

# Function to install Calibre isolated
install_calibre_isolated() {
    log_info "Installing Calibre in isolated mode to ~/calibre-bin..."
    
    # Download and install Calibre
    wget -nv -O- https://download.calibre-ebook.com/linux-installer.sh | \
        sh /dev/stdin install_dir=~/calibre-bin isolated=y
    
    # Add to PATH (use the correct calibre subdirectory)
    echo 'export PATH="$HOME/calibre-bin/calibre:$PATH"' >> ~/.bashrc
    export PATH="$HOME/calibre-bin/calibre:$PATH"
    
    # Test installation
    if test_ebook_convert "ebook-convert"; then
        log_success "Calibre isolated installation successful"
        return 0
    else
        log_warning "Calibre installed but ebook-convert not working"
        
        # Try to fix with libxcb-cursor0
        if install_libxcb_cursor_locally; then
            # Also set headless mode
            echo 'export QT_QPA_PLATFORM=offscreen' >> ~/.bashrc
            export QT_QPA_PLATFORM=offscreen
            
            if test_ebook_convert "ebook-convert"; then
                log_success "Calibre working after libxcb-cursor0 fix"
                return 0
            fi
        fi
        
        log_error "Calibre installation failed"
        return 1
    fi
}

# Function to install Calibre via Flatpak
install_calibre_flatpak() {
    log_info "Installing Calibre via Flatpak..."
    
    if ! command_exists flatpak; then
        log_error "Flatpak not available"
        return 1
    fi
    
    # Install Calibre
    flatpak install -y flathub com.calibre_ebook.calibre
    
    # Create wrapper script
    mkdir -p ~/.local/bin
    cat > ~/.local/bin/ebook-convert <<'EOF'
#!/usr/bin/env bash
exec flatpak run --command=ebook-convert com.calibre_ebook.calibre "$@"
EOF
    chmod +x ~/.local/bin/ebook-convert
    
    # Add to PATH
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
    
    # Test installation
    if test_ebook_convert "ebook-convert"; then
        log_success "Calibre Flatpak installation successful"
        return 0
    else
        log_error "Calibre Flatpak installation failed"
        return 1
    fi
}

# Function to create venv shim
create_venv_shim() {
    local venv_path="./venv"
    
    if [ ! -d "$venv_path" ]; then
        log_warning "Virtual environment not found at $venv_path"
        return 1
    fi
    
    log_info "Creating ebook-convert shim in virtual environment..."
    
    cat > "$venv_path/bin/ebook-convert" <<'EOF'
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
    exit 1
fi
EOF
    chmod +x "$venv_path/bin/ebook-convert"
    
    log_success "Virtual environment shim created"
}

# Function to set up environment variables
setup_environment() {
    log_info "Setting up environment variables..."
    
    # Set EBOOK_CONVERT_BIN
    echo 'export EBOOK_CONVERT_BIN="ebook-convert"' >> ~/.bashrc
    export EBOOK_CONVERT_BIN="ebook-convert"
    
    # Set headless mode for Qt (helps with X11 issues)
    echo 'export QT_QPA_PLATFORM=offscreen' >> ~/.bashrc
    export QT_QPA_PLATFORM=offscreen
    
    log_success "Environment variables configured"
}

# Function to test the complete setup
test_setup() {
    log_info "Testing Calibre installation..."
    
    # Test ebook-convert
    if test_ebook_convert "ebook-convert"; then
        local version=$(ebook-convert --version 2>/dev/null | head -n1)
        log_success "ebook-convert is working: $version"
        
        # Test in project context
        log_info "Testing in project context..."
        cd "$PROJECT_ROOT"
        
        if [ -f "src/book_download/test_epub_guard_integration.py" ]; then
            log_info "Running EPUB guard integration test..."
            if python src/book_download/test_epub_guard_integration.py; then
                log_success "EPUB guard integration test passed"
                return 0
            else
                log_warning "EPUB guard integration test had issues"
                return 1
            fi
        else
            log_warning "EPUB guard test not found - manual testing required"
            return 0
        fi
    else
        log_error "ebook-convert is not working"
        return 1
    fi
}

# Main installation logic
main() {
    log_info "Starting Calibre installation process..."
    
    # Check if already installed
    if test_ebook_convert "ebook-convert"; then
        log_success "Calibre is already installed and working"
        setup_environment
        create_venv_shim
        test_setup
        return 0
    fi
    
    # Try Option A: Isolated install
    log_info "Attempting Option A: Isolated install to ~/calibre-bin"
    if install_calibre_isolated; then
        setup_environment
        create_venv_shim
        test_setup
        return 0
    fi
    
    # Try Option B: Flatpak
    log_info "Attempting Option B: Flatpak installation"
    if install_calibre_flatpak; then
        setup_environment
        create_venv_shim
        test_setup
        return 0
    fi
    
    # If all else fails
    log_error "All installation methods failed"
    log_info "Manual installation required:"
    log_info "1. Visit https://calibre-ebook.com/download_linux"
    log_info "2. Download and install Calibre manually"
    log_info "3. Ensure ebook-convert is in your PATH"
    return 1
}

# Run main function
main "$@"

echo ""
echo "=== Installation Complete ==="
echo "To use Calibre in new terminal sessions:"
echo "  source ~/.bashrc"
echo ""
echo "To test the installation:"
echo "  ebook-convert --version"
echo ""
echo "To test EPUB guard integration:"
echo "  python src/book_download/test_epub_guard_integration.py"
echo ""
