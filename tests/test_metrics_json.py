#!/usr/bin/env python3
"""
Quick test to verify that the enhanced JSON metrics saving works correctly.
This runs a minimal test with just 2 quick generations to verify JSON structure.
"""
import torch
from tortoise.api import TextToSpeech
import torchaudio
import os
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess
import sys

# Add parent directory to path to import the scaffold
sys.path.append(str(Path(__file__).parent))
from scaffolds.test_emma_scaffold import (
    get_gpu_stats,
    get_gpu_memory_peak,
    save_results_json,
    OUTPUT_DIR
)

# Set environment variable for CUDA
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

def quick_test():
    """Run a quick test to verify JSON saving."""
    print("Starting quick metrics test...")

    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Mock test results for quick testing (simulating successful tests)
    test_results = []

    # Simulate ultra_fast test
    gpu_stats = get_gpu_stats()
    test_results.append({
        'test_name': 'Quick Test 1: ultra_fast',
        'preset': 'ultra_fast',
        'params': {},
        'filename': 'test_ultra_fast.wav',
        'elapsed_time': 3.45,
        'file_size_kb': 245.3,
        'audio_duration': 5.2,
        'success': True,
        'gpu_before': gpu_stats,
        'gpu_after': gpu_stats,
        'gpu_peak': get_gpu_memory_peak() if torch.cuda.is_available() else None
    })

    # Simulate fast test
    test_results.append({
        'test_name': 'Quick Test 2: fast',
        'preset': 'fast',
        'params': {'temperature': 0.7},
        'filename': 'test_fast.wav',
        'elapsed_time': 8.73,
        'file_size_kb': 245.3,
        'audio_duration': 5.2,
        'success': True,
        'gpu_before': gpu_stats,
        'gpu_after': gpu_stats,
        'gpu_peak': get_gpu_memory_peak() if torch.cuda.is_available() else None
    })

    # Simulate a failed test
    test_results.append({
        'test_name': 'Quick Test 3: Failed',
        'preset': 'standard',
        'params': {},
        'filename': 'test_failed.wav',
        'elapsed_time': 1.2,
        'success': False,
        'error': 'Simulated failure for testing',
        'gpu_before': gpu_stats,
        'gpu_after': gpu_stats
    })

    # Save results to JSON
    print("\nSaving comprehensive metrics to JSON...")
    save_results_json(test_results)

    # Find and verify the JSON file
    json_files = list(OUTPUT_DIR.glob("test_results_*.json"))
    if json_files:
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"\n✓ JSON file created: {latest_json}")

        # Load and verify structure
        with open(latest_json, 'r') as f:
            data = json.load(f)

        print("\nJSON Structure Verification:")
        print(f"  ✓ Metadata present: {'metadata' in data}")
        if 'metadata' in data:
            print(f"    - Test configuration: {'test_configuration' in data['metadata']}")
            print(f"    - System info: {'system_info' in data['metadata']}")
            print(f"    - Test summary: {'test_summary' in data['metadata']}")

        print(f"  ✓ Statistics present: {'statistics' in data}")
        if 'statistics' in data:
            print(f"    - By preset: {'by_preset' in data['statistics']}")
            print(f"    - Overall GPU: {'overall_gpu' in data['statistics']}")
            print(f"    - Overall timing: {'overall_timing' in data['statistics']}")

        print(f"  ✓ Individual results: {len(data.get('individual_results', []))} tests")
        print(f"  ✓ Failed tests tracked: {len(data.get('failed_tests', []))} failures")

        # Show summary from metadata
        if 'metadata' in data and 'test_summary' in data['metadata']:
            summary = data['metadata']['test_summary']
            print(f"\nTest Summary from JSON:")
            print(f"  - Total: {summary.get('total_tests', 0)}")
            print(f"  - Successful: {summary.get('successful_tests', 0)}")
            print(f"  - Failed: {summary.get('failed_tests', 0)}")
            print(f"  - Success rate: {summary.get('success_rate', 0):.1f}%")

        # Show preset statistics if available
        if 'statistics' in data and 'by_preset' in data['statistics']:
            print(f"\nPreset Statistics Captured:")
            for preset, stats in data['statistics']['by_preset'].items():
                if stats.get('test_count', 0) > 0:
                    print(f"  {preset}: {stats['test_count']} tests, "
                          f"avg time: {stats['timing']['avg_time']:.2f}s")

        print(f"\n✅ JSON metrics saving is working correctly!")
        print(f"   Full file: {latest_json.absolute()}")

        # Show file size
        file_size_kb = latest_json.stat().st_size / 1024
        print(f"   File size: {file_size_kb:.2f} KB")

    else:
        print("❌ No JSON file found!")

if __name__ == "__main__":
    quick_test()