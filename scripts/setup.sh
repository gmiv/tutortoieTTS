#!/bin/bash

# Setup script for Tortoise-TTS environment
set -e  # Exit on error

echo "===================================="
echo "   Tortoise-TTS Setup Script"
echo "===================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to check command availability
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed${NC}"
        return 1
    else
        echo -e "${GREEN}✅ $1 found: $(which $1)${NC}"
        return 0
    fi
}

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
echo "--------------------------------"

if ! check_command python3; then
    echo -e "${RED}Error: Python 3 is not installed!${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "   Python version: $PYTHON_VERSION"

if ! check_command pip3; then
    echo -e "${RED}Error: pip3 is not installed!${NC}"
    echo "Install it with: sudo apt-get install python3-pip"
    exit 1
fi

# Check for CUDA (optional but recommended)
echo ""
echo "Checking for CUDA support..."
if [ -d "/usr/lib/wsl/lib" ]; then
    echo -e "${GREEN}✅ WSL CUDA libraries found${NC}"
    CUDA_AVAILABLE=true
elif command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ CUDA installation detected${NC}"
    CUDA_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  No CUDA detected - will use CPU mode (slower)${NC}"
    CUDA_AVAILABLE=false
fi

# Step 2: Create virtual environment
echo ""
echo "Step 2: Setting up Python virtual environment..."
echo "------------------------------------------------"

VENV_DIR="$SCRIPT_DIR/tortoise-venv"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Step 3: Activate virtual environment
echo ""
echo "Step 3: Activating virtual environment..."
echo "-----------------------------------------"

source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Step 4: Install dependencies
echo ""
echo "Step 4: Installing dependencies..."
echo "-----------------------------------"

if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo "No requirements.txt found, installing Tortoise-TTS directly..."

    # Install PyTorch with CUDA support if available
    if [ "$CUDA_AVAILABLE" = true ]; then
        echo "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Installing PyTorch (CPU only)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install Tortoise-TTS
    echo "Installing Tortoise-TTS..."
    pip install tortoise-tts

    # Install additional useful packages
    echo "Installing additional packages..."
    pip install numpy scipy librosa soundfile

    # Save the installed packages
    echo "Saving package list to requirements.txt..."
    pip freeze > requirements.txt
    echo -e "${GREEN}✅ Packages installed and saved${NC}"
fi

# Step 5: Download models (optional)
echo ""
echo "Step 5: Model preparation..."
echo "-----------------------------"

echo "Testing Tortoise-TTS installation..."
python -c "import tortoise; print('✅ Tortoise-TTS successfully imported!')" 2>/dev/null || {
    echo -e "${RED}❌ Failed to import Tortoise-TTS${NC}"
    echo "Please check the installation logs above"
    exit 1
}

# Step 6: Create .gitignore if not exists
echo ""
echo "Step 6: Updating .gitignore..."
echo "-------------------------------"

if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << EOF
# Virtual environment
tortoise-venv/
venv/
env/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Output files
*.wav
*.mp3
*.flac
output/

# Model cache
.cache/
models/

# Logs
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
    echo -e "${GREEN}✅ .gitignore created${NC}"
else
    # Check if tortoise-venv is already in .gitignore
    if ! grep -q "tortoise-venv" .gitignore; then
        echo "Adding tortoise-venv to .gitignore..."
        echo -e "\n# Virtual environment\ntortoise-venv/" >> .gitignore
        echo -e "${GREEN}✅ Updated .gitignore${NC}"
    else
        echo -e "${GREEN}✅ .gitignore already configured${NC}"
    fi
fi

# Step 7: Setup CUDA paths for WSL
if [ -d "/usr/lib/wsl/lib" ]; then
    echo ""
    echo "Step 7: Configuring WSL CUDA paths..."
    echo "--------------------------------------"

    echo "export LD_LIBRARY_PATH=/usr/lib/wsl/lib:\$LD_LIBRARY_PATH" >> "$VENV_DIR/bin/activate"
    echo -e "${GREEN}✅ CUDA paths configured for WSL${NC}"
fi

# Final instructions
echo ""
echo "===================================="
echo -e "${GREEN}   Setup Complete!${NC}"
echo "===================================="
echo ""
echo "To use Tortoise-TTS, run:"
echo "  source ./activate.sh"
echo ""
echo "Or activate manually with:"
echo "  source tortoise-venv/bin/activate"
echo ""
echo "Then you can run:"
echo "  python generate_speech.py 'Your text here'"
echo "  python test_tts.py"
echo "  python clone_voice.py --help"
echo "  python batch_tts.py --help"
echo ""

# Test that everything works
echo "Running a quick test..."
source "$VENV_DIR/bin/activate"
python -c "
import torch
import tortoise
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('All systems ready!')
" || echo -e "${YELLOW}⚠️  Test partially failed - check logs above${NC}"