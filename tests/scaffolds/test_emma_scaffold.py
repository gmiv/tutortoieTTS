#!/usr/bin/env python3
"""
Comprehensive testing scaffold for Emma voice across all quality presets and parameters.
Each test outputs timing data and generates a unique WAV file.
"""
import torch
from tortoise.api import TextToSpeech
import torchaudio
import os
import time
from pathlib import Path
from datetime import datetime
import json
import subprocess

# Set environment variable for CUDA
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Test configuration
TEST_VOICE = 'emma'

# ============================================================
# TEST TEXT OPTIONS - Uncomment the one you want to use
# ============================================================

# ACTIVE: Phonetically Difficult - Tests consonant clusters, th/th combinations
TEST_TEXT = "Throughout the thousands of thorough thoughts, she thoroughly questioned whether the weathered leather would withstand further pressure."

# Alternative test texts (uncomment to activate):

# Simple & Casual - Basic test
# TEST_TEXT = "Hi! I'm emma. What are you up to?"

# Phonetic Variety & Tongue Twisters - Tests rapid articulation
# TEST_TEXT = "The sixth sick sheik's sixth sheep's sick - Peter Piper picked a peck of pickled peppers!"

# Numbers, Abbreviations & Complex Punctuation - Tests normalization
# TEST_TEXT = "Dr. Smith's appointment is at 3:45 PM on March 23rd, 2024. Call 555-0123 or email info@example.com - it's urgent!"

# Emotional Range & Natural Speech - Tests prosody and intonation
# TEST_TEXT = "Wait, what?! I can't believe it's already 2024... Time flies so fast, doesn't it? Well, I suppose we'd better get going then."

# Balanced Challenge - Mix of pronunciation, numbers, and natural speech
# TEST_TEXT = "Hi! I'm Emma. Did you know that approximately 73% of people struggle with pronunciation? It's fascinating, isn't it? Let's test this thoroughly!"

# ============================================================

# Get the project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Results storage
test_results = []

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi."""
    try:
        # Query GPU stats in CSV format for easier parsing
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        # Parse the output (handle multiple GPUs, take first one)
        lines = result.stdout.strip().split('\n')
        if not lines or not lines[0]:
            return None
        
        # Parse CSV line
        parts = [p.strip() for p in lines[0].split(',')]
        if len(parts) < 8:
            return None
        
        return {
            'gpu_index': parts[0],
            'gpu_name': parts[1],
            'gpu_utilization': float(parts[2]) if parts[2] else 0.0,
            'memory_utilization': float(parts[3]) if parts[3] else 0.0,
            'memory_used_mb': float(parts[4]) if parts[4] else 0.0,
            'memory_total_mb': float(parts[5]) if parts[5] else 0.0,
            'temperature': float(parts[6]) if parts[6] else 0.0,
            'power_draw': float(parts[7]) if parts[7] else 0.0,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError) as e:
        return None

def get_gpu_memory_peak():
    """Get peak GPU memory usage during the test."""
    try:
        # Try to get current memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)  # MB
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**2)  # MB
            max_memory_allocated = torch.cuda.max_memory_allocated(0) / (1024**2)  # MB
            max_memory_reserved = torch.cuda.max_memory_reserved(0) / (1024**2)  # MB
            return {
                'pytorch_memory_allocated_mb': memory_allocated,
                'pytorch_memory_reserved_mb': memory_reserved,
                'pytorch_max_memory_allocated_mb': max_memory_allocated,
                'pytorch_max_memory_reserved_mb': max_memory_reserved,
            }
    except Exception:
        pass
    return None

def load_voice_samples(tts, voice_name):
    """Load voice samples for the specified voice."""
    voice_samples_dir = Path(tts.models_dir).parent / 'site-packages' / 'tortoise' / 'voices' / voice_name
    
    # If the above path doesn't work, try alternate path
    if not voice_samples_dir.exists():
        import tortoise
        tortoise_path = Path(tortoise.__file__).parent
        voice_samples_dir = tortoise_path / 'voices' / voice_name
    
    if not voice_samples_dir.exists():
        raise FileNotFoundError(f"Voice samples directory not found for '{voice_name}'")
    
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
    
    return voice_samples

def generate_filename(preset, params=None, test_num=None):
    """Generate a unique filename for each test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if test_num is not None:
        base = f"emma_{preset}_{test_num:03d}"
    else:
        base = f"emma_{preset}"
    
    # Add parameter identifiers if custom params
    if params:
        param_strs = []
        for key, value in sorted(params.items()):
            if key not in ['num_autoregressive_samples', 'diffusion_iterations']:  # Skip preset params
                # Create short identifier
                short_key = key[:3] if len(key) > 3 else key
                short_val = str(value).replace('.', '').replace('-', '')
                param_strs.append(f"{short_key}{short_val}")
        if param_strs:
            base += "_" + "_".join(param_strs)
    
    return f"{base}_{timestamp}.wav"

def run_test(tts, voice_samples, text, preset, params=None, test_name=""):
    """Run a single TTS test and return timing/data."""
    filename = generate_filename(preset, params)
    output_path = OUTPUT_DIR / filename
    
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"Preset: {preset}")
    if params:
        print(f"Custom params: {params}")
    print(f"Output: {filename}")
    print(f"{'='*80}")
    
    # Get baseline GPU stats
    gpu_before = get_gpu_stats()
    if gpu_before:
        print(f"GPU Before: {gpu_before['gpu_utilization']:.1f}% util, "
              f"{gpu_before['memory_used_mb']:.0f}/{gpu_before['memory_total_mb']:.0f} MB memory, "
              f"{gpu_before['temperature']:.0f}°C")
    
    # Start timing
    start_time = time.time()
    
    # Reset PyTorch memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    
    try:
        # Generate speech
        if params:
            # Use custom parameters
            wav = tts.tts_with_preset(
                text, 
                voice_samples=voice_samples, 
                preset=preset,
                **params
            )
        else:
            # Use preset defaults
            wav = tts.tts_with_preset(
                text, 
                voice_samples=voice_samples, 
                preset=preset
            )
        
        # Save the output
        torchaudio.save(str(output_path), wav.squeeze(0).cpu(), 24000)
        
        # Calculate timing
        elapsed_time = time.time() - start_time
        
        # Get file info
        file_size = os.path.getsize(output_path) / 1024  # KB
        audio_duration = wav.shape[-1] / 24000  # seconds
        
        # Get GPU stats after test
        gpu_after = get_gpu_stats()
        gpu_peak = get_gpu_memory_peak()
        
        # Store results
        result = {
            'test_name': test_name,
            'preset': preset,
            'params': params or {},
            'filename': filename,
            'elapsed_time': elapsed_time,
            'file_size_kb': file_size,
            'audio_duration': audio_duration,
            'success': True,
            'gpu_before': gpu_before,
            'gpu_after': gpu_after,
            'gpu_peak': gpu_peak
        }
        
        print(f"✓ Success!")
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  File size: {file_size:.2f} KB")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Speed ratio: {audio_duration/elapsed_time:.2f}x")
        
        if gpu_after:
            print(f"  GPU After: {gpu_after['gpu_utilization']:.1f}% util, "
                  f"{gpu_after['memory_used_mb']:.0f}/{gpu_after['memory_total_mb']:.0f} MB memory, "
                  f"{gpu_after['temperature']:.0f}°C, {gpu_after['power_draw']:.1f}W")
        
        if gpu_peak:
            print(f"  PyTorch Peak Memory: {gpu_peak['pytorch_max_memory_allocated_mb']:.0f} MB allocated, "
                  f"{gpu_peak['pytorch_max_memory_reserved_mb']:.0f} MB reserved")
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        gpu_after = get_gpu_stats()
        
        result = {
            'test_name': test_name,
            'preset': preset,
            'params': params or {},
            'filename': filename,
            'elapsed_time': elapsed_time,
            'success': False,
            'error': str(e),
            'gpu_before': gpu_before,
            'gpu_after': gpu_after
        }
        print(f"✗ Failed: {e}")
        if gpu_after:
            print(f"  GPU After: {gpu_after['gpu_utilization']:.1f}% util, "
                  f"{gpu_after['memory_used_mb']:.0f}/{gpu_after['memory_total_mb']:.0f} MB memory")
        return result

def print_summary(results):
    """Print a nice summary of all test results."""
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\nTotal tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n{'Test Name':<30} {'Preset':<15} {'Time (s)':<12} {'GPU Util %':<12} {'GPU Mem (MB)':<15} {'PyTorch Peak (MB)':<18}")
        print("-" * 110)
        for r in successful:
            gpu_util = f"{r['gpu_after']['gpu_utilization']:.1f}%" if r.get('gpu_after') else "N/A"
            gpu_mem = f"{r['gpu_after']['memory_used_mb']:.0f}" if r.get('gpu_after') else "N/A"
            pytorch_peak = f"{r['gpu_peak']['pytorch_max_memory_allocated_mb']:.0f}" if r.get('gpu_peak') else "N/A"
            print(f"{r['test_name']:<30} {r['preset']:<15} {r['elapsed_time']:<12.2f} {gpu_util:<12} {gpu_mem:<15} {pytorch_peak:<18}")
        
        print(f"\n{'Test Name':<30} {'Preset':<15} {'Time (s)':<12} {'Size (KB)':<12} {'Duration (s)':<15} {'Speed':<10}")
        print("-" * 100)
        for r in successful:
            speed = f"{r['audio_duration']/r['elapsed_time']:.2f}x" if r['elapsed_time'] > 0 else "N/A"
            print(f"{r['test_name']:<30} {r['preset']:<15} {r['elapsed_time']:<12.2f} {r['file_size_kb']:<12.2f} {r['audio_duration']:<15.2f} {speed:<10}")
    
    if failed:
        print(f"\nFailed Tests:")
        for r in failed:
            print(f"  - {r['test_name']} ({r['preset']}): {r.get('error', 'Unknown error')}")
    
    # Timing statistics by preset
    print(f"\n{'='*80}")
    print("TIMING STATISTICS BY PRESET")
    print(f"{'='*80}")
    
    presets = ['ultra_fast', 'fast', 'standard', 'high_quality']
    for preset in presets:
        preset_results = [r for r in successful if r['preset'] == preset]
        if preset_results:
            times = [r['elapsed_time'] for r in preset_results]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # GPU stats
            gpu_utils = [r['gpu_after']['gpu_utilization'] for r in preset_results if r.get('gpu_after')]
            gpu_mems = [r['gpu_after']['memory_used_mb'] for r in preset_results if r.get('gpu_after')]
            pytorch_peaks = [r['gpu_peak']['pytorch_max_memory_allocated_mb'] for r in preset_results if r.get('gpu_peak')]
            
            print(f"\n{preset.upper()}:")
            print(f"  Tests: {len(preset_results)}")
            print(f"  Avg time: {avg_time:.2f}s")
            print(f"  Min time: {min_time:.2f}s")
            print(f"  Max time: {max_time:.2f}s")
            
            if gpu_utils:
                print(f"  Avg GPU util: {sum(gpu_utils)/len(gpu_utils):.1f}%")
            if gpu_mems:
                print(f"  Avg GPU memory: {sum(gpu_mems)/len(gpu_mems):.0f} MB")
            if pytorch_peaks:
                print(f"  Avg PyTorch peak: {sum(pytorch_peaks)/len(pytorch_peaks):.0f} MB")
    
    # GPU Statistics Summary
    print(f"\n{'='*80}")
    print("GPU STATISTICS SUMMARY")
    print(f"{'='*80}")
    
    all_gpu_utils = [r['gpu_after']['gpu_utilization'] for r in successful if r.get('gpu_after')]
    all_gpu_mems = [r['gpu_after']['memory_used_mb'] for r in successful if r.get('gpu_after')]
    all_pytorch_peaks = [r['gpu_peak']['pytorch_max_memory_allocated_mb'] for r in successful if r.get('gpu_peak')]
    
    if all_gpu_utils:
        print(f"\nGPU Utilization:")
        print(f"  Average: {sum(all_gpu_utils)/len(all_gpu_utils):.1f}%")
        print(f"  Max: {max(all_gpu_utils):.1f}%")
        print(f"  Min: {min(all_gpu_utils):.1f}%")
    
    if all_gpu_mems:
        print(f"\nGPU Memory Usage:")
        print(f"  Average: {sum(all_gpu_mems)/len(all_gpu_mems):.0f} MB")
        print(f"  Max: {max(all_gpu_mems):.0f} MB")
        print(f"  Min: {min(all_gpu_mems):.0f} MB")
    
    if all_pytorch_peaks:
        print(f"\nPyTorch Peak Memory:")
        print(f"  Average: {sum(all_pytorch_peaks)/len(all_pytorch_peaks):.0f} MB")
        print(f"  Max: {max(all_pytorch_peaks):.0f} MB")
        print(f"  Min: {min(all_pytorch_peaks):.0f} MB")

def save_results_json(results):
    """Save comprehensive results and statistics to JSON file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = OUTPUT_DIR / f"test_results_{timestamp}.json"

    # Calculate summary statistics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    # Statistics by preset
    preset_stats = {}
    presets = ['ultra_fast', 'fast', 'standard', 'high_quality']
    for preset in presets:
        preset_results = [r for r in successful if r['preset'] == preset]
        if preset_results:
            times = [r['elapsed_time'] for r in preset_results]
            gpu_utils = [r['gpu_after']['gpu_utilization'] for r in preset_results if r.get('gpu_after')]
            gpu_mems = [r['gpu_after']['memory_used_mb'] for r in preset_results if r.get('gpu_after')]
            pytorch_peaks = [r['gpu_peak']['pytorch_max_memory_allocated_mb'] for r in preset_results if r.get('gpu_peak')]
            audio_durations = [r['audio_duration'] for r in preset_results]
            file_sizes = [r['file_size_kb'] for r in preset_results]

            preset_stats[preset] = {
                'test_count': len(preset_results),
                'timing': {
                    'avg_time': sum(times) / len(times) if times else 0,
                    'min_time': min(times) if times else 0,
                    'max_time': max(times) if times else 0,
                    'total_time': sum(times) if times else 0
                },
                'gpu_stats': {
                    'avg_utilization': sum(gpu_utils)/len(gpu_utils) if gpu_utils else None,
                    'max_utilization': max(gpu_utils) if gpu_utils else None,
                    'min_utilization': min(gpu_utils) if gpu_utils else None,
                    'avg_memory_mb': sum(gpu_mems)/len(gpu_mems) if gpu_mems else None,
                    'max_memory_mb': max(gpu_mems) if gpu_mems else None,
                    'min_memory_mb': min(gpu_mems) if gpu_mems else None,
                    'avg_pytorch_peak_mb': sum(pytorch_peaks)/len(pytorch_peaks) if pytorch_peaks else None,
                    'max_pytorch_peak_mb': max(pytorch_peaks) if pytorch_peaks else None,
                    'min_pytorch_peak_mb': min(pytorch_peaks) if pytorch_peaks else None
                },
                'output_stats': {
                    'avg_audio_duration': sum(audio_durations)/len(audio_durations) if audio_durations else 0,
                    'total_audio_duration': sum(audio_durations) if audio_durations else 0,
                    'avg_file_size_kb': sum(file_sizes)/len(file_sizes) if file_sizes else 0,
                    'total_file_size_kb': sum(file_sizes) if file_sizes else 0,
                    'avg_speed_ratio': sum(audio_durations)/sum(times) if times and sum(times) > 0 else 0
                }
            }

    # Overall GPU statistics
    all_gpu_utils = [r['gpu_after']['gpu_utilization'] for r in successful if r.get('gpu_after')]
    all_gpu_mems = [r['gpu_after']['memory_used_mb'] for r in successful if r.get('gpu_after')]
    all_gpu_temps = [r['gpu_after']['temperature'] for r in successful if r.get('gpu_after')]
    all_gpu_power = [r['gpu_after']['power_draw'] for r in successful if r.get('gpu_after')]
    all_pytorch_peaks = [r['gpu_peak']['pytorch_max_memory_allocated_mb'] for r in successful if r.get('gpu_peak')]
    all_pytorch_reserved = [r['gpu_peak']['pytorch_max_memory_reserved_mb'] for r in successful if r.get('gpu_peak')]

    overall_gpu_stats = {
        'utilization': {
            'average': sum(all_gpu_utils)/len(all_gpu_utils) if all_gpu_utils else None,
            'max': max(all_gpu_utils) if all_gpu_utils else None,
            'min': min(all_gpu_utils) if all_gpu_utils else None
        },
        'memory_mb': {
            'average': sum(all_gpu_mems)/len(all_gpu_mems) if all_gpu_mems else None,
            'max': max(all_gpu_mems) if all_gpu_mems else None,
            'min': min(all_gpu_mems) if all_gpu_mems else None
        },
        'temperature_c': {
            'average': sum(all_gpu_temps)/len(all_gpu_temps) if all_gpu_temps else None,
            'max': max(all_gpu_temps) if all_gpu_temps else None,
            'min': min(all_gpu_temps) if all_gpu_temps else None
        },
        'power_draw_w': {
            'average': sum(all_gpu_power)/len(all_gpu_power) if all_gpu_power else None,
            'max': max(all_gpu_power) if all_gpu_power else None,
            'min': min(all_gpu_power) if all_gpu_power else None
        },
        'pytorch_peak_mb': {
            'average': sum(all_pytorch_peaks)/len(all_pytorch_peaks) if all_pytorch_peaks else None,
            'max': max(all_pytorch_peaks) if all_pytorch_peaks else None,
            'min': min(all_pytorch_peaks) if all_pytorch_peaks else None
        },
        'pytorch_reserved_mb': {
            'average': sum(all_pytorch_reserved)/len(all_pytorch_reserved) if all_pytorch_reserved else None,
            'max': max(all_pytorch_reserved) if all_pytorch_reserved else None,
            'min': min(all_pytorch_reserved) if all_pytorch_reserved else None
        }
    }

    # Overall timing statistics
    all_times = [r['elapsed_time'] for r in successful]
    all_audio_durations = [r['audio_duration'] for r in successful]
    all_file_sizes = [r['file_size_kb'] for r in successful]

    overall_timing_stats = {
        'total_processing_time': sum(all_times) if all_times else 0,
        'avg_processing_time': sum(all_times)/len(all_times) if all_times else 0,
        'total_audio_duration': sum(all_audio_durations) if all_audio_durations else 0,
        'avg_audio_duration': sum(all_audio_durations)/len(all_audio_durations) if all_audio_durations else 0,
        'overall_speed_ratio': sum(all_audio_durations)/sum(all_times) if all_times and sum(all_times) > 0 else 0,
        'total_file_size_kb': sum(all_file_sizes) if all_file_sizes else 0,
        'avg_file_size_kb': sum(all_file_sizes)/len(all_file_sizes) if all_file_sizes else 0
    }

    # Get system info
    system_info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'timestamp': timestamp,
        'test_date': datetime.now().isoformat()
    }

    # Get initial GPU info if available
    gpu_info = get_gpu_stats()
    if gpu_info:
        system_info['gpu_info'] = {
            'name': gpu_info['gpu_name'],
            'total_memory_mb': gpu_info['memory_total_mb']
        }

    # Comprehensive results object
    comprehensive_results = {
        'metadata': {
            'test_configuration': {
                'voice': TEST_VOICE,
                'test_text': TEST_TEXT,
                'test_text_length': len(TEST_TEXT),
                'output_directory': str(OUTPUT_DIR.absolute())
            },
            'system_info': system_info,
            'test_summary': {
                'total_tests': len(results),
                'successful_tests': len(successful),
                'failed_tests': len(failed),
                'success_rate': len(successful) / len(results) * 100 if results else 0
            }
        },
        'statistics': {
            'by_preset': preset_stats,
            'overall_gpu': overall_gpu_stats,
            'overall_timing': overall_timing_stats
        },
        'individual_results': results,
        'failed_tests': [
            {
                'test_name': r['test_name'],
                'preset': r['preset'],
                'error': r.get('error', 'Unknown error')
            } for r in failed
        ]
    }

    # Save to JSON with proper formatting
    with open(json_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)

    print(f"\n✓ Comprehensive results saved to: {json_path}")
    print(f"  - {len(results)} total tests")
    print(f"  - {len(successful)} successful, {len(failed)} failed")
    print(f"  - Includes {len(preset_stats)} preset statistics")
    print(f"  - Full GPU and performance metrics captured")

def main():
    # Check CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Check nvidia-smi availability
    gpu_stats = get_gpu_stats()
    if gpu_stats:
        print(f"\nGPU Info:")
        print(f"  Name: {gpu_stats['gpu_name']}")
        print(f"  Total Memory: {gpu_stats['memory_total_mb']:.0f} MB")
        print(f"  Initial State: {gpu_stats['gpu_utilization']:.1f}% util, "
              f"{gpu_stats['memory_used_mb']:.0f} MB used, {gpu_stats['temperature']:.0f}°C")
    else:
        print("\n⚠️  Warning: nvidia-smi not available or GPU not detected. GPU stats will be limited.")
    
    # Initialize TTS
    print("\nInitializing Tortoise-TTS...")
    tts = TextToSpeech()
    
    # Load Emma voice samples
    print(f"\nLoading voice samples for: {TEST_VOICE}")
    voice_samples = load_voice_samples(tts, TEST_VOICE)
    print(f"Loaded {len(voice_samples)} voice samples")
    
    print(f"\nTest text: '{TEST_TEXT}'")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # ====================================================================
    # TEST SUITE 1: All Quality Presets
    # ====================================================================
    print(f"\n{'#'*80}")
    print("# TEST SUITE 1: Quality Presets")
    print(f"{'#'*80}")
    
    presets = ['ultra_fast', 'fast', 'standard', 'high_quality']
    for i, preset in enumerate(presets, 1):
        result = run_test(
            tts, voice_samples, TEST_TEXT, preset,
            test_name=f"Preset {i}: {preset}"
        )
        test_results.append(result)
    
    # ====================================================================
    # TEST SUITE 2: Parameter Variations
    # ====================================================================
    print(f"\n{'#'*80}")
    print("# TEST SUITE 2: Parameter Variations")
    print(f"{'#'*80}")
    
    # Test different temperature values
    temperatures = [0.6, 0.8, 1.0]
    for i, temp in enumerate(temperatures, 1):
        result = run_test(
            tts, voice_samples, TEST_TEXT, 'fast',
            params={'temperature': temp},
            test_name=f"Temperature {i}: {temp}"
        )
        test_results.append(result)
    
    # Test different repetition penalties
    rep_penalties = [1.5, 2.0, 2.5]
    for i, rep_pen in enumerate(rep_penalties, 1):
        result = run_test(
            tts, voice_samples, TEST_TEXT, 'fast',
            params={'repetition_penalty': rep_pen},
            test_name=f"Repetition Penalty {i}: {rep_pen}"
        )
        test_results.append(result)
    
    # Test different top_p values
    top_p_values = [0.7, 0.8, 0.9]
    for i, top_p in enumerate(top_p_values, 1):
        result = run_test(
            tts, voice_samples, TEST_TEXT, 'fast',
            params={'top_p': top_p},
            test_name=f"Top-p {i}: {top_p}"
        )
        test_results.append(result)
    
    # Test different diffusion temperatures
    diff_temps = [0.8, 1.0, 1.2]
    for i, diff_temp in enumerate(diff_temps, 1):
        result = run_test(
            tts, voice_samples, TEST_TEXT, 'standard',
            params={'diffusion_temperature': diff_temp},
            test_name=f"Diffusion Temp {i}: {diff_temp}"
        )
        test_results.append(result)
    
    # Test cond_free_k variations
    cond_free_k_values = [1.0, 2.0, 3.0]
    for i, cfk in enumerate(cond_free_k_values, 1):
        result = run_test(
            tts, voice_samples, TEST_TEXT, 'standard',
            params={'cond_free_k': cfk},
            test_name=f"Cond Free K {i}: {cfk}"
        )
        test_results.append(result)
    
    # ====================================================================
    # TEST SUITE 3: Combined Parameter Tests
    # ====================================================================
    print(f"\n{'#'*80}")
    print("# TEST SUITE 3: Combined Parameter Tests")
    print(f"{'#'*80}")
    
    # High quality combination
    result = run_test(
        tts, voice_samples, TEST_TEXT, 'standard',
        params={
            'temperature': 0.7,
            'repetition_penalty': 2.5,
            'top_p': 0.9,
            'diffusion_temperature': 1.0
        },
        test_name="High Quality Combo"
    )
    test_results.append(result)
    
    # Fast but consistent combination
    result = run_test(
        tts, voice_samples, TEST_TEXT, 'fast',
        params={
            'temperature': 0.6,
            'repetition_penalty': 2.0,
            'top_p': 0.7
        },
        test_name="Fast Consistent Combo"
    )
    test_results.append(result)
    
    # ====================================================================
    # Print Summary and Save Results
    # ====================================================================
    print_summary(test_results)
    save_results_json(test_results)
    
    print(f"\n{'='*80}")
    print("All tests completed!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

