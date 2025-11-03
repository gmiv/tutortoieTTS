#!/usr/bin/env python3
"""
Test script for Tortoise-TTS with female voice (emma)
"""
import torch
from tortoise.api import TextToSpeech
import torchaudio
import os
from pathlib import Path

# Get the project root directory (parent of tests directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Set environment variable for CUDA
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Check CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Initialize TTS
print("\nInitializing Tortoise-TTS...")
tts = TextToSpeech()

# ============================================================
# DEFAULT VOICE SETTING (Female)
# ============================================================
# To change the voice, modify DEFAULT_VOICE below to one of:
#   - 'emma'    : Female voice (current default)
#   - 'angie'   : Female voice
#   - 'halle'   : Female voice
#   - 'jlaw'    : Female voice (Jennifer Lawrence)
#   - 'lj'      : Female voice
#   - 'rainbow' : Female voice
# Or use male voices: 'daniel', 'deniro', 'freeman', 'geralt',
#                     'tom', 'william', 'pat', etc.
# ============================================================
DEFAULT_VOICE = 'jlaw'  # Current: jlaw (Jennifer Lawrence)
voice_samples_dir = Path(tts.models_dir).parent / 'site-packages' / 'tortoise' / 'voices' / DEFAULT_VOICE

# If the above path doesn't work, try alternate path
if not voice_samples_dir.exists():
    # Try finding in the installed package
    import tortoise
    tortoise_path = Path(tortoise.__file__).parent
    voice_samples_dir = tortoise_path / 'voices' / DEFAULT_VOICE

print(f"\nLoading voice: {DEFAULT_VOICE}")
print(f"Voice samples directory: {voice_samples_dir}")

# Load the voice sample files
voice_samples = []
for voice_file in sorted(voice_samples_dir.glob('*.wav')):
    audio, sr = torchaudio.load(str(voice_file))
    # Resample to 22050 Hz if needed
    if sr != 22050:
        resampler = torchaudio.transforms.Resample(sr, 22050)
        audio = resampler(audio)
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    voice_samples.append(audio)
    print(f"  Loaded: {voice_file.name}")

# Generate speech
text = "Hello! This is a test of Tortoise TTS. The system is working perfectly and generating high quality speech."
print(f"\nGenerating speech for: '{text}'")
print(f"Using voice: {DEFAULT_VOICE} (female)")

# Use fast preset with female voice samples
wav = tts.tts_with_preset(text, voice_samples=voice_samples, preset='ultra_fast')

# Save the output
output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "test_output_uf.wav"
torchaudio.save(str(output_path), wav.squeeze(0).cpu(), 24000)

print(f"\nSuccess! Audio saved to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")