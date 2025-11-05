#!/usr/bin/env python3
"""
Complete Voice Cloning Workflow Example
Demonstrates the full process from recording to generation
"""

import os
import sys
from pathlib import Path

# Add tortoise to path
sys.path.insert(0, '/mnt/c/Users/gmora/Documents/REPO/tutortoieTTS/tortoise-venv/lib/python3.10/site-packages')

from tortoise.api import TextToSpeech
import torchaudio
import torch

class VoiceCloningWorkflow:
    def __init__(self, voice_name):
        self.voice_name = voice_name
        self.samples_dir = Path(f"data/voices/custom/{voice_name}")
        self.output_dir = Path(f"outputs/cloned_voices/{voice_name}")
        self.tts = None
        self.voice_samples = []
        self.conditioning_latents = None

    def setup(self):
        """Initialize directories and TTS engine"""
        # Set environment variables
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

        # Create directories
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Setting up voice cloning for: {self.voice_name}")
        print(f"Samples directory: {self.samples_dir}")
        print(f"Output directory: {self.output_dir}")

        # Initialize TTS with optimizations
        print("\nInitializing Tortoise-TTS...")
        self.tts = TextToSpeech(
            half=True,  # Use FP16 for memory efficiency
            kv_cache=True  # Enable KV cache for longer texts
        )

        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Warning: Running on CPU (will be slow)")

    def load_samples(self):
        """Load and preprocess voice samples"""
        print(f"\nLoading samples from {self.samples_dir}")

        wav_files = list(self.samples_dir.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"No WAV files found in {self.samples_dir}")

        # Load up to 5 samples
        for wav_file in wav_files[:5]:
            print(f"  Loading: {wav_file.name}")
            audio, sr = torchaudio.load(wav_file)

            # Resample to 22050 Hz if needed
            if sr != 22050:
                resampler = torchaudio.transforms.Resample(sr, 22050)
                audio = resampler(audio)

            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            self.voice_samples.append(audio)

        print(f"Loaded {len(self.voice_samples)} samples successfully")

    def compute_latents(self, save=True):
        """Compute and optionally save conditioning latents"""
        print("\nComputing conditioning latents...")
        self.conditioning_latents = self.tts.get_conditioning_latents(self.voice_samples)

        if save:
            import pickle
            latent_path = self.output_dir / f"{self.voice_name}_latents.pkl"
            with open(latent_path, 'wb') as f:
                pickle.dump(self.conditioning_latents, f)
            print(f"Saved latents to: {latent_path}")

    def generate_speech(self, text, preset='fast', use_latents=True):
        """Generate speech with the cloned voice"""
        print(f"\nGenerating speech with {self.voice_name} voice")
        print(f"Preset: {preset}")
        print(f"Text: {text[:100]}...")

        if use_latents and self.conditioning_latents:
            # Use pre-computed latents (faster)
            wav = self.tts.tts_with_preset(
                text,
                conditioning_latents=self.conditioning_latents,
                preset=preset
            )
        else:
            # Use raw samples
            wav = self.tts.tts_with_preset(
                text,
                voice_samples=self.voice_samples,
                preset=preset
            )

        # Save output
        output_path = self.output_dir / f"{self.voice_name}_{preset}.wav"
        torchaudio.save(str(output_path), wav.squeeze(0).cpu(), 24000)

        file_size_kb = output_path.stat().st_size / 1024
        print(f"\nSaved to: {output_path}")
        print(f"File size: {file_size_kb:.2f} KB")

        return output_path

    def batch_generate(self, texts, preset='fast'):
        """Generate multiple texts efficiently"""
        print(f"\nBatch generating {len(texts)} texts")

        outputs = []
        for i, text in enumerate(texts, 1):
            print(f"\nProcessing {i}/{len(texts)}: {text[:50]}...")
            output_path = self.output_dir / f"{self.voice_name}_batch_{i:03d}.wav"

            wav = self.tts.tts_with_preset(
                text,
                conditioning_latents=self.conditioning_latents,
                preset=preset
            )

            torchaudio.save(str(output_path), wav.squeeze(0).cpu(), 24000)
            outputs.append(output_path)
            print(f"  Saved: {output_path.name}")

        return outputs

    def test_quality_presets(self, test_text="This is a test of voice cloning quality."):
        """Test all quality presets"""
        print("\nTesting all quality presets...")

        presets = ['ultra_fast', 'fast', 'standard', 'high_quality']
        results = {}

        for preset in presets:
            print(f"\nTesting preset: {preset}")
            import time
            start = time.time()

            output = self.generate_speech(test_text, preset=preset)
            elapsed = time.time() - start

            results[preset] = {
                'file': output,
                'time': elapsed,
                'size_kb': output.stat().st_size / 1024
            }

            print(f"  Time: {elapsed:.2f}s")

        # Print summary
        print("\n" + "="*50)
        print("QUALITY PRESET COMPARISON")
        print("="*50)
        for preset, data in results.items():
            print(f"{preset:12} | Time: {data['time']:6.2f}s | Size: {data['size_kb']:7.2f} KB")

        return results

def main():
    """Example workflow"""

    # Example 1: Basic voice cloning
    print("="*60)
    print("VOICE CLONING WORKFLOW EXAMPLE")
    print("="*60)

    # Initialize workflow
    workflow = VoiceCloningWorkflow("example_voice")
    workflow.setup()

    print("\n" + "-"*60)
    print("STEP 1: Place your WAV files in:")
    print(f"  {workflow.samples_dir}")
    print("\nSample requirements:")
    print("  - 3-5 WAV files")
    print("  - 5-15 seconds each")
    print("  - Clear speech, minimal background noise")
    print("-"*60)

    # Check if samples exist
    if not list(workflow.samples_dir.glob("*.wav")):
        print("\nNo samples found. Creating example with built-in voice...")

        # Use built-in emma voice as example
        emma_samples = Path("/mnt/c/Users/gmora/Documents/REPO/tutortoieTTS/tortoise-venv/lib/python3.10/site-packages/tortoise/voices/emma")
        if emma_samples.exists():
            import shutil
            for wav in emma_samples.glob("*.wav"):
                shutil.copy(wav, workflow.samples_dir)
            print(f"Copied Emma voice samples to {workflow.samples_dir}")

    # Load samples
    try:
        workflow.load_samples()
    except FileNotFoundError:
        print("\nPlease add WAV files to the samples directory and run again.")
        return

    # Compute latents
    workflow.compute_latents()

    # Example 2: Single generation
    print("\n" + "-"*60)
    print("EXAMPLE: Single Text Generation")
    print("-"*60)

    sample_text = """
    Welcome to the voice cloning demonstration.
    This text is being spoken in a cloned voice,
    created from just a few audio samples.
    """

    workflow.generate_speech(sample_text.strip(), preset='fast')

    # Example 3: Batch processing
    print("\n" + "-"*60)
    print("EXAMPLE: Batch Processing")
    print("-"*60)

    batch_texts = [
        "First paragraph of your content.",
        "Second paragraph goes here.",
        "Third paragraph with different text."
    ]

    workflow.batch_generate(batch_texts, preset='ultra_fast')

    # Example 4: Quality comparison
    print("\n" + "-"*60)
    print("EXAMPLE: Quality Preset Comparison")
    print("-"*60)

    comparison_text = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."
    workflow.test_quality_presets(comparison_text)

    print("\n" + "="*60)
    print("WORKFLOW COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {workflow.output_dir}")
    print("\nNext steps:")
    print("1. Listen to the generated files to evaluate quality")
    print("2. Adjust the preset based on your speed/quality needs")
    print("3. Use the saved latents for faster future generation")

    # Example 5: Using saved latents
    print("\n" + "-"*60)
    print("TIP: Loading and using saved latents")
    print("-"*60)

    import pickle
    latent_file = workflow.output_dir / f"{workflow.voice_name}_latents.pkl"
    if latent_file.exists():
        print(f"Loading latents from: {latent_file}")
        with open(latent_file, 'rb') as f:
            saved_latents = pickle.load(f)
        print("Latents loaded successfully!")
        print("You can now use these latents for instant voice cloning without reprocessing samples.")

if __name__ == "__main__":
    main()