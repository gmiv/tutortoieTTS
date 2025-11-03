#!/usr/bin/env python3
"""
Batch Text-to-Speech Processing Script for Tortoise-TTS
Processes long texts by splitting into sentences and generating audio for each
"""
import os
import re
import torch
from tortoise.api import TextToSpeech
import torchaudio
import argparse
from pathlib import Path
import time

def split_into_sentences(text):
    """Split text into sentences"""
    # Simple sentence splitting (can be improved with NLTK)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def batch_process_text(input_file, voice='random', preset='fast', output_dir='batch_output',
                      chunk_size=None, combine=True):
    """Process a text file in batches"""

    # Set environment
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

    print(f"📚 Batch TTS Processing")
    print(f"📄 Input: {input_file}")
    print(f"🎤 Voice: {voice}")
    print(f"⚡ Preset: {preset}")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("⚠️  Running on CPU (will be slow)")
        device = 'cpu'

    # Read input text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into sentences or chunks
    if chunk_size:
        # Split by character count
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    else:
        # Split by sentences
        chunks = split_into_sentences(text)

    print(f"📊 Split into {len(chunks)} chunks")

    # Initialize TTS
    print("\n🚀 Initializing Tortoise-TTS...")
    tts = TextToSpeech()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each chunk
    audio_files = []
    total_time = 0

    for i, chunk in enumerate(chunks, 1):
        print(f"\n🔄 Processing chunk {i}/{len(chunks)}...")
        print(f"   Text: {chunk[:50]}...")

        start_time = time.time()

        # Generate audio
        if voice == 'random':
            wav = tts.tts_with_preset(chunk, preset=preset)
        else:
            # For custom voices, load samples from voices/{voice}/ directory
            voice_dir = Path(f"voices/{voice}")
            if voice_dir.exists():
                samples = list(voice_dir.glob("*.wav"))
                if samples:
                    reference_clips = []
                    for sample in samples[:5]:  # Use up to 5 samples
                        audio, sr = torchaudio.load(str(sample))
                        if sr != 22050:
                            resampler = torchaudio.transforms.Resample(sr, 22050)
                            audio = resampler(audio)
                        if audio.shape[0] > 1:
                            audio = torch.mean(audio, dim=0, keepdim=True)
                        reference_clips.append(audio)
                    wav = tts.tts_with_preset(chunk, voice_samples=reference_clips, preset=preset)
                else:
                    print(f"⚠️  No samples found for voice {voice}, using random")
                    wav = tts.tts_with_preset(chunk, preset=preset)
            else:
                print(f"⚠️  Voice directory not found: {voice_dir}, using random")
                wav = tts.tts_with_preset(chunk, preset=preset)

        # Save chunk audio
        chunk_file = f"{output_dir}/chunk_{i:04d}.wav"
        torchaudio.save(chunk_file, wav.squeeze(0).cpu(), 24000)
        audio_files.append(chunk_file)

        elapsed = time.time() - start_time
        total_time += elapsed
        print(f"   ✓ Generated in {elapsed:.1f}s - Saved to {chunk_file}")

    # Combine all chunks if requested
    if combine and len(audio_files) > 1:
        print(f"\n🔗 Combining {len(audio_files)} audio chunks...")
        combined_audio = []

        for file in audio_files:
            audio, sr = torchaudio.load(file)
            combined_audio.append(audio)

        # Add small silence between chunks
        silence = torch.zeros(1, int(24000 * 0.5))  # 0.5 second silence
        final_audio = []
        for i, audio in enumerate(combined_audio):
            final_audio.append(audio)
            if i < len(combined_audio) - 1:
                final_audio.append(silence)

        combined = torch.cat(final_audio, dim=1)
        combined_file = f"{output_dir}/combined.wav"
        torchaudio.save(combined_file, combined, 24000)

        print(f"✅ Combined audio saved to: {combined_file}")
        print(f"📏 Total size: {os.path.getsize(combined_file) / 1024 / 1024:.2f} MB")

    print(f"\n✨ Batch processing complete!")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"📁 Output directory: {output_dir}")

    return audio_files

def main():
    parser = argparse.ArgumentParser(description='Batch TTS processing with Tortoise')
    parser.add_argument('input', help='Input text file')
    parser.add_argument('--voice', default='random', help='Voice to use (default: random)')
    parser.add_argument('--preset', default='fast',
                       choices=['ultra_fast', 'fast', 'standard', 'high_quality'],
                       help='Quality preset (default: fast)')
    parser.add_argument('--output', default='batch_output', help='Output directory')
    parser.add_argument('--chunk-size', type=int, help='Split by character count instead of sentences')
    parser.add_argument('--no-combine', action='store_true', help="Don't combine chunks into single file")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    batch_process_text(
        args.input,
        voice=args.voice,
        preset=args.preset,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        combine=not args.no_combine
    )

if __name__ == "__main__":
    main()