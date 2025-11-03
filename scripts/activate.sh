#!/bin/bash
# Activation script for Tortoise-TTS environment

echo "🚀 Activating Tortoise-TTS environment..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check if we're being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "⚠️  This script must be sourced, not executed!"
    echo "   Please run: source ./scripts/activate.sh"
    echo "   Or:         . ./scripts/activate.sh"
    exit 1
fi

# Activate the virtual environment with full path
source "$PROJECT_ROOT/tortoise-venv/bin/activate"

# Set the CUDA library path
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Verify Python is from venv
PYTHON_PATH=$(which python)
if [[ "$PYTHON_PATH" == *"tortoise-venv"* ]]; then
    echo "✅ Virtual environment activated!"
    echo "📍 Python location: $PYTHON_PATH"
    echo "🐍 Python version: $(python --version 2>&1)"
else
    echo "❌ Warning: Python not from virtual environment"
    echo "   Current python: $PYTHON_PATH"
fi

# Test if tortoise module is available
python -c "import tortoise; print('✅ Tortoise module found!')" 2>/dev/null || echo "❌ Warning: Tortoise module not accessible"

echo ""
echo "✅ Environment ready! You can now run:"
echo "   python src/tutortoietts/cli/generate.py 'Your text here'"
echo "   python tests/test_tts.py"
echo "   python src/tutortoietts/cli/clone.py --help"
echo "   python src/tutortoietts/cli/batch.py --help"