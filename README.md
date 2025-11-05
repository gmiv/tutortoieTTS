# TutortoieTTS

High-quality text-to-speech (TTS) system with voice cloning capabilities, built on Tortoise-TTS v3.0.0.

## Features

- **Text-to-Speech Generation**: Convert text to natural-sounding speech with multiple built-in voices
- **Voice Cloning**: Clone any voice with just 3-5 audio samples
- **Multiple Quality Presets**: Choose between speed and quality (ultra_fast, fast, standard, high_quality)
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs
- **Simple CLI**: Easy-to-use wrapper scripts for common tasks

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- WSL2 (for Windows users)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tutortoieTTS.git
cd tutortoieTTS
```

2. Run the setup script:
```bash
./scripts/setup.sh
```

3. Activate the environment:
```bash
source scripts/activate.sh
```

## Usage

### Text-to-Speech Generation

Generate speech from text using the convenient wrapper script:

```bash
# Basic usage - from the scripts directory
cd scripts
./generate.sh "Hello, this is a test of the TTS system"

# Specify voice and quality preset
./generate.sh "Your text here" -v emma -p high_quality

# Save to specific location
./generate.sh "Your text" --output ../outputs/my_audio.wav --voice daniel

# Show help
./generate.sh --help
```

**Available Voices**: emma, gertrude, daniel, deniro, freeman, halle, jlaw, lj, mol, pat, pat2, snakes, tom, weaver, william, and more.

### Voice Cloning

Clone a voice using your own audio samples:

```bash
# From the scripts directory
cd scripts

# Clone a voice with WAV samples
./clone_voice.sh my_voice "../data/samples/*.wav" "Text to generate" fast

# Example with specific voice
./clone_voice.sh john_doe "../data/john/*.wav" "Hello, this is my cloned voice" high_quality
```

#### Voice Sample Requirements

- **Format**: WAV files (22050 Hz recommended, will auto-resample if needed)
- **Duration**: 5-15 seconds per sample
- **Quantity**: 3-5 samples for best results (more samples = better quality)
- **Quality**: Clear speech with minimal background noise
- **Content**: Natural speech, different sentences per sample

## Project Structure

```
tutortoieTTS/
├── scripts/              # Convenience wrapper scripts (run from here)
│   ├── setup.sh         # Installation and environment setup
│   ├── activate.sh      # Activate the Python environment
│   ├── generate.sh      # TTS generation wrapper
│   └── clone_voice.sh   # Voice cloning wrapper
├── src/tutortoietts/    # Main source code
│   ├── cli/            # Command-line interfaces
│   │   ├── generate.py # TTS generation
│   │   └── clone.py    # Voice cloning
│   └── core/           # Core functionality
├── examples/            # Example scripts and workflows
│   └── sample_cloning_workflow.py
├── docs/               # Documentation
│   ├── VOICE_CLONING_GUIDE.md  # Detailed cloning guide
│   ├── USAGE_GUIDE.md          # Complete usage documentation
│   ├── PERFORMANCE.md          # Performance optimization
│   ├── MODEL.md               # Technical model details
│   └── PROJECT_STRUCTURE.md   # Architecture overview
├── data/               # Data directory
│   └── voices/         # Voice samples for cloning
└── outputs/            # Generated audio files
```

## Quality Presets

| Preset | Speed | Quality | Use Case | Est. Time* |
|--------|-------|---------|----------|-----------|
| **ultra_fast** | ⚡⚡⚡ | ★★★☆☆ | Quick previews, testing | 0.5-1s |
| **fast** | ⚡⚡ | ★★★★☆ | General use, batch processing | 2-5s |
| **standard** | ⚡ | ★★★★★ | Production quality | 10-20s |
| **high_quality** | 🐢 | ★★★★★ | Premium quality | 30-60s |

*Per sentence on GPU (RTX 3090/A6000)

## Advanced Usage

### Python API

For programmatic usage, see the example workflow:

```python
from pathlib import Path
import sys
sys.path.insert(0, '/path/to/tortoise-venv/lib/python3.10/site-packages')

from tortoise.api import TextToSpeech

# Initialize TTS
tts = TextToSpeech(half=True, kv_cache=True)

# Generate speech with a preset
wav = tts.tts_with_preset(
    "Your text here",
    voice_samples=None,  # Use built-in voice
    preset='fast'
)

# Save output
import torchaudio
torchaudio.save('output.wav', wav.squeeze(0).cpu(), 24000)
```

See `examples/sample_cloning_workflow.py` for a complete voice cloning workflow example.

### Direct CLI Usage

You can also use the Python CLI scripts directly:

```bash
# Text-to-speech generation
python3 src/tutortoietts/cli/generate.py "Hello world" \
    --output output.wav \
    --voice emma \
    --preset fast

# Voice cloning
python3 src/tutortoietts/cli/clone.py \
    --name my_voice \
    --samples "data/samples/*.wav" \
    --text "Text to generate" \
    --preset fast \
    --output outputs/cloned_voices
```

## Performance Optimization

- **GPU**: RTX 3090/4090 or A6000 recommended for best performance
- **Memory**: 16GB+ system RAM, 8GB+ VRAM
- **FP16 Mode**: Enabled by default for memory efficiency
- **KV Cache**: Enabled for faster processing of longer texts
- **Batch Processing**: Use the example workflow for processing multiple texts efficiently

### Preset Parameters

Each preset uses different parameters for quality vs speed trade-offs:

| Preset | Autoregressive Samples | Diffusion Iterations | Classifier-Free |
|--------|:----------------------:|:-------------------:|:---------------:|
| ultra_fast | 16 | 30 | No |
| fast | 96 | 80 | Yes |
| standard | 256 | 200 | Yes |
| high_quality | 256 | 400 | Yes |

For detailed parameter tuning, see [docs/MODEL.md](docs/MODEL.md).

## Documentation

- **[VOICE_CLONING_GUIDE.md](docs/VOICE_CLONING_GUIDE.md)** - Step-by-step voice cloning guide
- **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Complete usage documentation
- **[PERFORMANCE.md](docs/PERFORMANCE.md)** - Performance optimization tips
- **[MODEL.md](docs/MODEL.md)** - Technical details and parameter tuning
- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Architecture overview

## Troubleshooting

### Common Issues

**CUDA out of memory**
- Use a faster preset (ultra_fast or fast)
- Reduce text length
- Close other GPU applications

**Module not found errors**
- Ensure environment is activated: `source scripts/activate.sh`
- Check PYTHONPATH is set correctly
- Re-run setup: `./scripts/setup.sh`

**Poor voice cloning quality**
- Use 5+ high-quality samples instead of 3
- Ensure samples are clear with no background noise
- Use longer samples (10-15 seconds each)
- Try different text that matches the voice's style
- Use higher quality presets (standard or high_quality)

**Slow generation**
- Enable GPU acceleration (check with `nvidia-smi`)
- Use faster presets (ultra_fast or fast)
- Reduce text length
- Ensure CUDA drivers are up to date

**Scripts not found**
- Make sure you're in the correct directory
- For wrapper scripts, `cd scripts` first
- Make scripts executable: `chmod +x scripts/*.sh`

## Examples

### Example 1: Quick TTS Test
```bash
cd scripts
./generate.sh "Hello world, this is a test" -v emma -p fast
```

### Example 2: Voice Cloning Workflow
```bash
# 1. Prepare your samples in data/voices/my_voice/
mkdir -p ../data/voices/my_voice
# ... add your WAV files ...

# 2. Clone the voice
cd scripts
./clone_voice.sh my_voice "../data/voices/my_voice/*.wav" \
    "The quick brown fox jumps over the lazy dog" standard
```

### Example 3: Batch Processing
```python
# See examples/sample_cloning_workflow.py for complete example
python3 examples/sample_cloning_workflow.py
```

## License

This project uses Tortoise-TTS. Please refer to the original [Tortoise-TTS repository](https://github.com/neonbjb/tortoise-tts) for licensing information.

## Acknowledgments

Built on top of [Tortoise-TTS](https://github.com/neonbjb/tortoise-tts) by James Betker.

## Support

For issues or questions:
1. Check the documentation in `docs/`
2. Review examples in `examples/`
3. Open an issue on GitHub

---

**Quick Reference Commands:**

```bash
# Setup
./scripts/setup.sh
source scripts/activate.sh

# Generate speech
cd scripts
./generate.sh "Your text" -v emma -p fast

# Clone voice
cd scripts
./clone_voice.sh voice_name "samples/*.wav" "Text" fast

# Get help
./generate.sh --help
```
