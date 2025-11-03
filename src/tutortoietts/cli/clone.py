#!/usr/bin/env python3
"""
Voice Cloning Script for Tortoise-TTS
Usage: python clone_voice.py --name "voice_name" --samples "path/to/audio/*.wav" --text "Your text here"
"""
import os
import argparse
import torch
from tortoise.api import TextToSpeech
import torchaudio
from pathlib import Path

def clone_voice(voice_name, sample_paths, text, preset='fast', output_dir='outputs'):
    """Clone a voice using provided audio samples"""

    # Set environment variable for CUDA
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

    print(f"🎤 Voice Cloning: {voice_name}")
    print(f"📁 Samples: {len(sample_paths)} files")
    print(f"⚡ Preset: {preset}")
    print(f"🔤 Text: {text[:50]}...")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU (will be slow)")

    # Initialize TTS
    print("\n🚀 Initializing Tortoise-TTS...")
    tts = TextToSpeech()

    # Load voice samples
    print("\n📊 Loading voice samples...")
    reference_clips = []
    for path in sample_paths:
        audio, sr = torchaudio.load(path)
        # Resample to 22050 Hz if needed
        if sr != 22050:
            resampler = torchaudio.transforms.Resample(sr, 22050)
            audio = resampler(audio)
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        reference_clips.append(audio)
        print(f"  ✓ Loaded: {Path(path).name}")

    # Generate speech with cloned voice
    print(f"\n🎯 Generating speech with {voice_name} voice...")
    wav = tts.tts_with_preset(
        text,
        voice_samples=reference_clips,
        conditioning_latents=None,
        preset=preset
    )

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{voice_name}_{preset}.wav"
    torchaudio.save(output_path, wav.squeeze(0).cpu(), 24000)

    print(f"\n✅ Success! Audio saved to: {output_path}")
    print(f"📏 File size: {os.path.getsize(output_path) / 1024:.2f} KB")

    return output_path

def main():
    parser = argparse.ArgumentParser(description='Clone a voice using Tortoise-TTS')
    parser.add_argument('--name', required=True, help='Name for the cloned voice')
    parser.add_argument('--samples', required=True, nargs='+', help='Path(s) to voice sample files')
    parser.add_argument('--text', required=True, help='Text to generate')
    parser.add_argument('--preset', default='fast', choices=['ultra_fast', 'fast', 'standard', 'high_quality'],
                       help='Quality preset (default: fast)')
    parser.add_argument('--output', default='outputs', help='Output directory')

    args = parser.parse_args()

    # Verify sample files exist
    sample_paths = []
    for pattern in args.samples:
        files = list(Path('.').glob(pattern))
        if not files:
            print(f"❌ No files found matching: {pattern}")
            return
        sample_paths.extend([str(f) for f in files])

    if not sample_paths:
        print("❌ No valid audio samples found")
        return

    # Clone the voice
    clone_voice(args.name, sample_paths, args.text, args.preset, args.output)

if __name__ == "__main__":
    main()