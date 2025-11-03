#!/usr/bin/env python3
"""
Quiet speech generation - no progress bars
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Suppress progress bars
os.environ['TQDM_DISABLE'] = '1'

from tortoise.api import TextToSpeech
import torchaudio
import torch

def generate(text, output='output.wav', preset='fast'):
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

    print(f"🎤 Generating: '{text[:50]}...'")
    print(f"⚡ Preset: {preset}")

    tts = TextToSpeech()
    wav = tts.tts_with_preset(text, preset=preset)
    torchaudio.save(output, wav.squeeze(0).cpu(), 24000)

    size = os.path.getsize(output) / 1024
    duration = wav.shape[-1] / 24000

    print(f"✅ Saved to: {output} ({size:.1f} KB, {duration:.1f}s)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_speech_quiet.py 'Your text here' [output.wav] [preset]")
        sys.exit(1)

    text = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else 'output.wav'
    preset = sys.argv[3] if len(sys.argv) > 3 else 'fast'

    generate(text, output, preset)