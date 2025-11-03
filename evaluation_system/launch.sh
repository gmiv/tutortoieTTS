#!/bin/bash

# TTS Evaluation System Launcher

echo "========================================="
echo "    TTS Output Evaluation System        "
echo "========================================="
echo ""

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv ../venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -q -r requirements.txt

# Check if outputs directory exists
OUTPUT_DIR="../outputs"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "⚠️  Output directory not found at: $OUTPUT_DIR"
    echo "Creating output directory..."
    mkdir -p "$OUTPUT_DIR"
fi

# Count WAV files
WAV_COUNT=$(find "$OUTPUT_DIR" -name "*.wav" 2>/dev/null | wc -l)
echo ""
echo "📊 Found $WAV_COUNT WAV files in outputs directory"

# Check for test results
TEST_RESULTS=$(find "$OUTPUT_DIR" -name "test_results_*.json" 2>/dev/null | wc -l)
echo "📝 Found $TEST_RESULTS test result files"

# Launch the server
echo ""
echo "========================================="
echo "Starting evaluation server..."
echo "========================================="
echo ""
echo "🌐 Open your browser to: http://localhost:8000"
echo ""
echo "Keyboard shortcuts:"
echo "  Space     - Play/Pause audio"
echo "  ↑         - Rate as Good (Thumbs up)"
echo "  ↓         - Rate as Bad (Thumbs down)"
echo "  S         - Skip this file"
echo "  ←/→       - Navigate Previous/Next"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="
echo ""

# Start the FastAPI server
python app.py