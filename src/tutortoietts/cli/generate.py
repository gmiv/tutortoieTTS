#!/usr/bin/env python3
"""
Simple text-to-speech generation script
Usage: python generate_speech.py "Your text here" --output filename.wav
"""
import os
import sys
import argparse
import torch
from tortoise.api import TextToSpeech
import torchaudio
import re
import time

def split_text_into_chunks(text, max_chars=350):
    """Split text into chunks that won't exceed token limit (350 chars ≈ 300 tokens, well under 400 limit)"""
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If a single sentence is too long, split it further
        if len(sentence) > max_chars:
            # If current chunk has content, save it
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Split long sentence by commas or spaces
            words = sentence.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 > max_chars:
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                    temp_chunk = word
                else:
                    temp_chunk += " " + word if temp_chunk else word
            if temp_chunk:
                chunks.append(temp_chunk.strip())
        else:
            # Check if adding this sentence exceeds limit
            if len(current_chunk) + len(sentence) + 1 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def generate_speech(text, output_file='output.wav', preset='fast', voice='random'):
    """Generate speech from text"""

    # Set environment
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

    print(f"🎤 Generating speech...")
    print(f"📝 Text: {text[:100]}...")
    print(f"⚡ Preset: {preset}")
    print(f"🎯 Voice: {voice}")

    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Running on CPU (will be slower)")

    # Initialize TTS
    print("\n🚀 Initializing Tortoise-TTS...")
    tts = TextToSpeech()

    # Split text into chunks to avoid token limit
    chunks = split_text_into_chunks(text, max_chars=350)
    print(f"📦 Split text into {len(chunks)} chunks")

    # Estimate processing time
    preset_times = {
        'ultra_fast': 1,
        'fast': 3,
        'standard': 15,
        'high_quality': 45
    }
    est_time_per_chunk = preset_times.get(preset, 10)
    est_total_time = len(chunks) * est_time_per_chunk
    print(f"⏱️ Estimated time: {est_total_time//60}m {est_total_time%60}s ({est_time_per_chunk}s per chunk with '{preset}' preset)")

    # Generate speech for each chunk
    print("🔊 Generating audio...")
    audio_chunks = []
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        chunk_start = time.time()
        print(f"\n  [{i+1}/{len(chunks)}] {chunk[:80]}...")

        if voice == 'random':
            # Random voice each time
            wav = tts.tts_with_preset(chunk, preset=preset)
        else:
            # Use a specific built-in voice
            # Built-in voices: 'angie', 'applejack', 'daniel', 'deniro', 'emma', 'freeman',
            # 'geralt', 'halle', 'jlaw', 'lj', 'mol', 'myself', 'pat', 'pat2', 'rainbow',
            # 'snakes', 'tim_reynolds', 'tom', 'train_atkins', 'train_dotrice', 'train_dreams',
            # 'train_empire', 'train_grace', 'train_kennard', 'train_lescault', 'train_mouse', 'weaver', 'william'

            # For built-in voices, we need to use the voice parameter differently
            # For now, just use random voice (you can extend this to load built-in voices)
            wav = tts.tts_with_preset(chunk, preset=preset)

        audio_chunks.append(wav.squeeze(0))

        # Show timing info
        chunk_time = time.time() - chunk_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining_chunks = len(chunks) - (i + 1)
        eta = avg_time * remaining_chunks

        print(f"  ✓ Chunk done in {chunk_time:.1f}s | Avg: {avg_time:.1f}s/chunk | ETA: {eta//60:.0f}m {eta%60:.0f}s")

    # Concatenate all audio chunks
    if len(audio_chunks) > 1:
        print("🔗 Concatenating audio chunks...")
        final_audio = torch.cat(audio_chunks, dim=-1)
    else:
        final_audio = audio_chunks[0]

    # Save the output
    torchaudio.save(output_file, final_audio.cpu(), 24000)

    print(f"\n✅ Success! Audio saved to: {output_file}")
    print(f"📏 File size: {os.path.getsize(output_file) / 1024:.2f} KB")

    duration = final_audio.shape[-1] / 24000
    print(f"⏱️ Duration: {duration:.1f} seconds")

    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate speech from text')
    parser.add_argument('text', help='Text to convert to speech')
    parser.add_argument('--output', '-o', default='output.wav', help='Output WAV file (default: output.wav)')
    parser.add_argument('--preset', '-p', default='fast',
                       choices=['ultra_fast', 'fast', 'standard', 'high_quality'],
                       help='Quality preset (default: fast)')
    parser.add_argument('--voice', '-v', default='random', help='Voice to use (default: random)')

    args = parser.parse_args()

    generate_speech(args.text, args.output, args.preset, args.voice)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print("Usage examples:")
        print('  python generate_speech.py "Hello world"')
        print('  python generate_speech.py "Your text here" --output speech.wav')
        print('  python generate_speech.py "Fast speech" --preset ultra_fast')
        print('  python generate_speech.py "High quality" --preset high_quality -o hq.wav')
        print("\nPresets:")
        print("  ultra_fast : 0.5-1s per sentence (lowest quality)")
        print("  fast       : 2-5s per sentence (good balance)")
        print("  standard   : 10-20s per sentence (default quality)")
        print("  high_quality : 30-60s per sentence (best quality)")
    else:
        main()