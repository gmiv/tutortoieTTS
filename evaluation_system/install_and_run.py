#!/usr/bin/env python3
"""
Install requirements and run the evaluation system
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    requirements = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "jinja2==3.1.2",
        "python-multipart==0.0.6"
    ]

    for req in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", req])
    print("✓ Requirements installed")

def main():
    print("=" * 50)
    print("    TTS Output Evaluation System Setup")
    print("=" * 50)
    print()

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)

    # Install requirements
    try:
        import fastapi
        import uvicorn
        import jinja2
        print("✓ Requirements already installed")
    except ImportError:
        install_requirements()

    # Check for WAV files
    outputs_dir = Path(__file__).parent.parent / "outputs"
    wav_count = len(list(outputs_dir.glob("*.wav")))

    print(f"\n📊 Found {wav_count} WAV files to evaluate")

    if wav_count == 0:
        print("\n⚠️  No WAV files found!")
        print("Generate test files first:")
        print("  python tests/scaffolds/test_emma_scaffold.py")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Starting evaluation server...")
    print("=" * 50)
    print("\n🌐 Open your browser to: http://localhost:8000\n")
    print("Keyboard shortcuts:")
    print("  Space  - Play/Pause")
    print("  ↑      - Thumbs Up")
    print("  ↓      - Thumbs Down")
    print("  S      - Skip")
    print("  ←/→    - Navigate")
    print("\nPress Ctrl+C to stop\n")
    print("=" * 50 + "\n")

    # Run the app
    os.chdir(Path(__file__).parent)
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    main()