#!/usr/bin/env python3
"""
Quick test to verify the evaluation system loads correctly
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_system():
    """Test that the evaluation system can load files."""

    print("Testing TTS Evaluation System...")
    print("-" * 40)

    # Test paths
    BASE_DIR = Path(__file__).parent
    PROJECT_ROOT = BASE_DIR.parent
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"

    # Check directories exist
    assert OUTPUTS_DIR.exists(), f"Outputs directory not found: {OUTPUTS_DIR}"
    print(f"✓ Outputs directory found: {OUTPUTS_DIR}")

    # Check for WAV files
    wav_files = list(OUTPUTS_DIR.glob("*.wav"))
    print(f"✓ Found {len(wav_files)} WAV files")

    if wav_files:
        print(f"  First file: {wav_files[0].name}")
        print(f"  File size: {wav_files[0].stat().st_size / 1024:.1f} KB")

    # Check for test results
    test_results = list(OUTPUTS_DIR.glob("test_results_*.json"))
    print(f"✓ Found {len(test_results)} test result files")

    if test_results:
        latest = max(test_results, key=lambda x: x.stat().st_mtime)
        print(f"  Latest: {latest.name}")

        # Load and check structure
        with open(latest, 'r') as f:
            data = json.load(f)

        if 'individual_results' in data:
            results = data['individual_results']
            successful = [r for r in results if r.get('success')]
            print(f"  Contains {len(successful)} successful tests")

    # Test FastAPI import
    try:
        from fastapi import FastAPI
        print("✓ FastAPI imported successfully")
    except ImportError:
        print("✗ FastAPI not installed. Run: pip install fastapi uvicorn jinja2")
        return False

    # Test app import
    try:
        from app import app, state
        print("✓ App imported successfully")
        print(f"  Loaded {len(state.wav_files)} files for evaluation")

        if state.wav_files:
            print(f"  First file to evaluate: {state.wav_files[0]['filename']}")
            print(f"  Test name: {state.wav_files[0]['test_name']}")
            print(f"  Preset: {state.wav_files[0]['preset']}")
    except Exception as e:
        print(f"✗ Failed to import app: {e}")
        return False

    print("-" * 40)
    print("✅ All tests passed!")
    print("\nTo start the evaluation system, run:")
    print("  ./launch.sh")
    print("or")
    print("  python app.py")
    print("\nThen open: http://localhost:8000")

    return True

if __name__ == "__main__":
    success = test_system()