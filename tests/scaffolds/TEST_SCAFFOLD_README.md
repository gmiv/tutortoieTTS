# Emma Voice Testing Scaffold

A comprehensive testing framework for evaluating Tortoise-TTS performance with the Emma voice across all quality presets and parameter combinations. Each test captures detailed timing data, GPU utilization metrics, and generates uniquely named output files.

## Overview

This testing scaffold systematically tests the Emma voice through:
- **4 Quality Presets**: `ultra_fast`, `fast`, `standard`, `high_quality`
- **Multiple Parameter Variations**: Temperature, repetition penalty, top-p, diffusion temperature, and conditioning-free K values
- **Combined Parameter Tests**: High-quality and fast-consistent combinations

Each test generates:
- Unique WAV output files with descriptive naming
- Detailed timing metrics (elapsed time, audio duration, speed ratio)
- GPU utilization and memory statistics via `nvidia-smi`
- PyTorch memory peak tracking
- Comprehensive JSON results export

## Requirements

- Python 3.10+
- Tortoise-TTS installed and configured
- CUDA-capable GPU (recommended)
- NVIDIA drivers with `nvidia-smi` (for GPU monitoring)
- PyTorch with CUDA support

## Usage

### Basic Usage

```bash
# Activate your virtual environment
source tortoise-venv/bin/activate  # Linux/Mac
# or
tortoise-venv\Scripts\activate     # Windows

# Run the test scaffold
python test_emma_scaffold.py
```

### Configuration

You can modify the test configuration at the top of the script:

```python
TEST_VOICE = 'emma'  # Voice to test
TEST_TEXT = "Your test text here"  # Text to generate
OUTPUT_DIR = Path("test_outputs")  # Output directory
```

## Test Suites

### Suite 1: Quality Presets
Tests all 4 quality presets with default parameters:
- `ultra_fast`: Fastest generation (~0.5-1s per sentence)
- `fast`: Balanced quality and speed (~2-5s per sentence)
- `standard`: High quality (~10-20s per sentence)
- `high_quality`: Best quality (~30-60s per sentence)

### Suite 2: Parameter Variations
Tests individual parameters on `fast` and `standard` presets:

**Temperature** (0.6, 0.8, 1.0)
- Controls randomness in autoregressive generation
- Lower = more consistent, Higher = more varied

**Repetition Penalty** (1.5, 2.0, 2.5)
- Prevents repetitive outputs
- Higher = less repetition

**Top-p** (0.7, 0.8, 0.9)
- Nucleus sampling parameter
- Lower = more focused outputs

**Diffusion Temperature** (0.8, 1.0, 1.2)
- Controls variance in diffusion model
- Higher = more variation

**Conditioning-Free K** (1.0, 2.0, 3.0)
- Balances conditioning-free vs conditioned signals
- Higher = more conditioning-free influence

### Suite 3: Combined Parameter Tests
Tests optimized parameter combinations:
- **High Quality Combo**: Optimized for best quality
- **Fast Consistent Combo**: Optimized for speed with consistency

## Output Files

### WAV Files
Each test generates a uniquely named WAV file in the `test_outputs/` directory:

**Naming Convention:**
```
emma_{preset}_{params}_{timestamp}.wav
```

**Examples:**
- `emma_ultra_fast_20240101_120000.wav` - Ultra fast preset
- `emma_fast_tem08_20240101_120015.wav` - Fast preset with temperature=0.8
- `emma_standard_tem07_rep25_top09_20240101_120030.wav` - Combined parameters

### JSON Results
A comprehensive JSON file with all test results:
```
test_outputs/test_results_YYYYMMDD_HHMMSS.json
```

**JSON Structure:**
```json
{
  "test_name": "Preset 1: ultra_fast",
  "preset": "ultra_fast",
  "params": {},
  "filename": "emma_ultra_fast_...wav",
  "elapsed_time": 1.23,
  "file_size_kb": 45.67,
  "audio_duration": 3.45,
  "success": true,
  "gpu_before": {
    "gpu_utilization": 5.0,
    "memory_used_mb": 1024,
    "temperature": 45
  },
  "gpu_after": {
    "gpu_utilization": 85.0,
    "memory_used_mb": 8192,
    "temperature": 65
  },
  "gpu_peak": {
    "pytorch_max_memory_allocated_mb": 7500,
    "pytorch_max_memory_reserved_mb": 8500
  }
}
```

## GPU Monitoring

The script captures comprehensive GPU statistics using `nvidia-smi`:

### Metrics Captured
- **GPU Utilization**: Percentage of GPU compute being used
- **Memory Utilization**: Percentage of GPU memory being used
- **Memory Used/Total**: Actual memory usage in MB
- **Temperature**: GPU temperature in Celsius
- **Power Draw**: Current power consumption in Watts
- **PyTorch Peak Memory**: Maximum memory allocated/reserved by PyTorch

### When GPU Monitoring Fails
If `nvidia-smi` is not available, the script will:
- Display a warning message
- Continue running tests
- Still capture PyTorch memory stats (if CUDA is available)
- Mark GPU stats as "N/A" in output

## Console Output

The script provides detailed console output:

### Per-Test Output
```
================================================================================
Test: Preset 1: ultra_fast
Preset: ultra_fast
Output: emma_ultra_fast_20240101_120000.wav
================================================================================
GPU Before: 5.0% util, 1024/8192 MB memory, 45°C
✓ Success!
  Time: 1.23s
  File size: 45.67 KB
  Audio duration: 3.45s
  Speed ratio: 2.80x
  GPU After: 85.0% util, 8192/8192 MB memory, 65°C, 250.5W
  PyTorch Peak Memory: 7500 MB allocated, 8500 MB reserved
```

### Summary Output
```
================================================================================
TEST SUMMARY
================================================================================

Total tests: 20
Successful: 20
Failed: 0

Test Name                       Preset          Time (s)     GPU Util %   GPU Mem (MB)    PyTorch Peak (MB)
--------------------------------------------------------------------------------------------------------------
Preset 1: ultra_fast           ultra_fast      1.23         85.0%        8192            7500
...

TIMING STATISTICS BY PRESET
================================================================================

ULTRA_FAST:
  Tests: 1
  Avg time: 1.23s
  Min time: 1.23s
  Max time: 1.23s
  Avg GPU util: 85.0%
  Avg GPU memory: 8192 MB
  Avg PyTorch peak: 7500 MB

GPU STATISTICS SUMMARY
================================================================================

GPU Utilization:
  Average: 75.5%
  Max: 95.0%
  Min: 45.0%

GPU Memory Usage:
  Average: 6144 MB
  Max: 8192 MB
  Min: 2048 MB

PyTorch Peak Memory:
  Average: 5500 MB
  Max: 8500 MB
  Min: 3000 MB
```

## Performance Expectations

Approximate timings per test (varies by GPU):

| Preset | Time per Test | GPU Memory |
|--------|---------------|------------|
| `ultra_fast` | 0.5-2s | ~2-4 GB |
| `fast` | 2-8s | ~4-8 GB |
| `standard` | 10-30s | ~6-12 GB |
| `high_quality` | 30-90s | ~8-16 GB |

**Total runtime**: ~5-15 minutes for all tests (depending on GPU)

## Troubleshooting

### nvidia-smi Not Found
- Ensure NVIDIA drivers are installed
- Verify `nvidia-smi` is in your PATH
- On Windows, ensure CUDA toolkit is installed
- Tests will continue without GPU stats

### Out of Memory Errors
- Try testing fewer presets at a time
- Use `ultra_fast` or `fast` presets only
- Close other GPU-intensive applications
- Reduce batch size if processing multiple texts

### Voice Samples Not Found
- Ensure Emma voice samples exist in the Tortoise voices directory
- Check that the voice name matches exactly (case-sensitive)
- Verify voice samples are in WAV format

## Customization

### Adding More Tests
Edit the `main()` function to add custom test cases:

```python
# Custom test example
result = run_test(
    tts, voice_samples, TEST_TEXT, 'fast',
    params={'temperature': 0.75, 'repetition_penalty': 2.2},
    test_name="Custom Test"
)
test_results.append(result)
```

### Changing Test Text
Modify `TEST_TEXT` at the top of the script or pass different text per test.

### Output Directory
Change `OUTPUT_DIR` to save files to a different location.

## Integration with CI/CD

The script can be integrated into automated testing pipelines:

```bash
# Run tests and capture output
python test_emma_scaffold.py > test_results.log 2>&1

# Check exit code (0 = success)
echo $?

# Parse JSON results for automated analysis
python -c "import json; data=json.load(open('test_outputs/test_results_*.json')); print(sum(1 for t in data if t['success']))"
```

## License

This testing scaffold is part of the tutortoieTTS project.

## See Also

- [USAGE_GUIDE.md](USAGE_GUIDE.md) - General Tortoise-TTS usage guide
- [PLAN.md](PLAN.md) - Project planning document
- [Delivery.md](Delivery.md) - Delivery documentation

