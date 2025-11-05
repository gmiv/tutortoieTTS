#!/bin/bash
# Text-to-Speech Generation Wrapper for TutortoieTTS
# Usage: ./generate.sh <text> [options]

# Function to show help
show_help() {
    echo "TutortoieTTS - Text-to-Speech Generation Script"
    echo ""
    echo "Usage: $0 <text> [options]"
    echo ""
    echo "Arguments:"
    echo "  text            Text to convert to speech (required)"
    echo ""
    echo "Options:"
    echo "  -o, --output    Output WAV file path (default: outputs/generated.wav)
                  Relative paths are relative to project root"
    echo "  -v, --voice     Voice to use (default: random)"
    echo "                  Options: random, emma, gertrude, daniel, deniro, freeman, etc."
    echo "  -p, --preset    Quality preset (default: fast)"
    echo "                  Options: ultra_fast, fast, standard, high_quality"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 \"Hello world\""
    echo "  $0 \"Hello world\" -v emma -p fast"
    echo "  $0 \"Hello world\" --output my_speech.wav --voice daniel --preset high_quality"
    echo ""
    exit 0
}

# Default values
OUTPUT="outputs/generated.wav"
VOICE="random"
PRESET="fast"
TEXT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -v|--voice)
            VOICE="$2"
            shift 2
            ;;
        -p|--preset)
            PRESET="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
        *)
            # First non-option argument is the text
            if [ -z "$TEXT" ]; then
                TEXT="$1"
            else
                echo "Error: Multiple text arguments provided. Please quote your text."
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if text was provided
if [ -z "$TEXT" ]; then
    echo "Error: No text provided"
    echo "Use -h or --help for usage information"
    exit 1
fi

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT/tortoise-venv/lib/python3.10/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"

# Handle output path - if relative, make it relative to project root
case "$OUTPUT" in
    /*)
        # Absolute path, use as-is
        ;;
    *)
        # Relative path, prepend project root
        OUTPUT="$PROJECT_ROOT/$OUTPUT"
        ;;
esac

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "TutortoieTTS - Text-to-Speech Generation"
echo "========================================="
echo "Text: \"$TEXT\""
echo "Voice: $VOICE"
echo "Preset: $PRESET"
echo "Output: $OUTPUT"
echo "-----------------------------------------"

# Run the generation script
python3 "$PROJECT_ROOT/src/tutortoietts/cli/generate.py" "$TEXT" \
    --output "$OUTPUT" \
    --voice "$VOICE" \
    --preset "$PRESET"

# Check if successful
if [ $? -eq 0 ]; then
    echo "-----------------------------------------"
    echo "✓ Speech generation completed successfully!"
    echo "Output saved to: $OUTPUT"

    # Get file size
    if [ -f "$OUTPUT" ]; then
        SIZE=$(du -h "$OUTPUT" | cut -f1)
        echo "File size: $SIZE"
    fi
else
    echo "✗ Error: Speech generation failed"
    exit 1
fi