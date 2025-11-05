#!/bin/bash
# Voice Cloning Wrapper for TutortoieTTS
# Usage: ./clone_voice.sh <voice_name> <sample_path> <text> [preset]

# Check arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <voice_name> <sample_path> <text> [preset]"
    echo "  voice_name: Name for your cloned voice"
    echo "  sample_path: Path to audio samples (can use wildcards like *.wav)"
    echo "  text: Text to generate"
    echo "  preset: ultra_fast, fast, standard, or high_quality (default: fast)"
    echo ""
    echo "Example: $0 my_voice 'samples/*.wav' 'Hello world' fast"
    exit 1
fi

VOICE_NAME="$1"
SAMPLE_PATH="$2"
TEXT="$3"
PRESET="${4:-fast}"

# Set environment variables
export PYTHONPATH="/mnt/c/Users/gmora/Documents/REPO/tutortoieTTS/tortoise-venv/lib/python3.10/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"

# Create output directory if it doesn't exist
mkdir -p outputs/cloned_voices

echo "Starting voice cloning..."
echo "Voice Name: $VOICE_NAME"
echo "Samples: $SAMPLE_PATH"
echo "Preset: $PRESET"
echo "---"

# Run the cloning script
python3 src/tutortoietts/cli/clone.py \
    --name "$VOICE_NAME" \
    --samples $SAMPLE_PATH \
    --text "$TEXT" \
    --preset "$PRESET" \
    --output outputs/cloned_voices

# Check if successful
if [ $? -eq 0 ]; then
    echo "---"
    echo "Voice cloning completed successfully!"
    echo "Output saved to: outputs/cloned_voices/${VOICE_NAME}_${PRESET}.wav"
else
    echo "Error: Voice cloning failed"
    exit 1
fi