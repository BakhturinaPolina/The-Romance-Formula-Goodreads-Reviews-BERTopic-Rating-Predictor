#!/bin/bash
# Setup script for Calibre installation (No Sudo Required)
# Required for MOBI to EPUB conversion in the book download system
# Uses isolated install, Flatpak fallback, and local library vendoring

echo "=== Calibre Installation Setup (No Sudo) ==="
echo "This script will install Calibre without requiring root privileges"
echo ""

# Check if the new user installation script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_SETUP_SCRIPT="$SCRIPT_DIR/setup_calibre_user.sh"

if [ -f "$USER_SETUP_SCRIPT" ]; then
    echo "Found comprehensive user installation script: $USER_SETUP_SCRIPT"
    echo "Running user installation script..."
    echo ""
    
    # Make it executable and run it
    chmod +x "$USER_SETUP_SCRIPT"
    exec "$USER_SETUP_SCRIPT"
else
    echo "User installation script not found. Falling back to basic detection..."
    echo ""
    
    # Basic detection and guidance
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Detected Linux system"
        echo ""
        echo "For no-sudo installation, please use one of these methods:"
        echo ""
        echo "Option 1 - Isolated Install (Recommended):"
        echo "  wget -nv -O- https://download.calibre-ebook.com/linux-installer.sh | sh /dev/stdin install_dir=~/calibre-bin isolated=y"
        echo "  echo 'export PATH=\"\$HOME/calibre-bin:\$PATH\"' >> ~/.bashrc"
        echo "  source ~/.bashrc"
        echo ""
        echo "Option 2 - Flatpak (if available):"
        echo "  flatpak install -y flathub com.calibre_ebook.calibre"
        echo "  mkdir -p ~/.local/bin"
        echo "  echo '#!/usr/bin/env bash' > ~/.local/bin/ebook-convert"
        echo "  echo 'exec flatpak run --command=ebook-convert com.calibre_ebook.calibre \"\$@\"' >> ~/.local/bin/ebook-convert"
        echo "  chmod +x ~/.local/bin/ebook-convert"
        echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
        echo ""
        echo "Option 3 - System Package Manager (requires sudo):"
        echo "  sudo apt-get install calibre  # Ubuntu/Debian"
        echo "  sudo yum install calibre      # CentOS/RHEL"
        echo "  sudo dnf install calibre      # Fedora"
        echo ""
        echo "For detailed instructions and automatic setup, run:"
        echo "  ./setup_calibre_user.sh"
        echo ""
        echo "Visit: https://calibre-ebook.com/download_linux for more information"
    
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Detected macOS system"
        echo ""
        echo "For macOS installation:"
        echo "Option 1 - Homebrew (Recommended):"
        echo "  brew install --cask calibre"
        echo ""
        echo "Option 2 - Manual Download:"
        echo "  Visit: https://calibre-ebook.com/download_osx"
        echo ""
        
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Windows (Git Bash or Cygwin)
        echo "Detected Windows system"
        echo ""
        echo "For Windows installation:"
        echo "  Visit: https://calibre-ebook.com/download_windows"
        echo "  Make sure 'ebook-convert' is available in your PATH"
        echo ""
        
    else
        echo "Unknown operating system: $OSTYPE"
        echo "Please install Calibre manually:"
        echo "Visit: https://calibre-ebook.com/download"
    fi
fi

    # Verify installation
    echo ""
    echo "Verifying Calibre installation..."
    if command -v ebook-convert &> /dev/null; then
        echo "✓ Calibre installed successfully!"
        ebook-convert --version
    else
        echo "✗ Calibre not found in PATH"
        echo "Please ensure 'ebook-convert' is available in your PATH"
        echo "You may need to restart your terminal or add Calibre to your PATH"
        echo ""
        echo "To test the installation after setup:"
        echo "  source ~/.bashrc"
        echo "  ebook-convert --version"
    fi
    
    echo ""
    echo "=== Setup Complete ==="
    echo "The book download system can now convert MOBI files to EPUB format."
    echo ""
    echo "Environment variables to set:"
    echo "  export EBOOK_CONVERT_BIN=\"ebook-convert\""
    echo "  export QT_QPA_PLATFORM=offscreen  # For headless operation"
