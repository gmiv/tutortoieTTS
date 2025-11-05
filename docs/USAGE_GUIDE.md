# Tortoise-TTS Usage Guide

## 🚀 Quick Start

### Activate Environment
```bash
source tortoise-venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### Basic Text-to-Speech
```bash
# Simple test with random voice
python test_tts.py

# Using the Python API
python -c "from tortoise.api import TextToSpeech; tts = TextToSpeech(); wav = tts.tts_with_preset('Hello world', preset='fast'); import torchaudio; torchaudio.save('output.wav', wav.squeeze(0).cpu(), 24000)"
```

## 🎤 Voice Cloning

### Prepare Voice Samples
1. Record 3-5 audio clips (5-15 seconds each)
2. Save as WAV files (22050 Hz mono preferred)
3. Place in `voices/[voice_name]/` directory

### Clone a Voice
```bash
# Using the clone_voice.py script
python clone_voice.py \
  --name "my_voice" \
  --samples "voices/my_voice/*.wav" \
  --text "Your text here" \
  --preset fast

# Preset Quality & Speed Guide:
# - ultra_fast: ~0.5-1s latency, lowest quality, fastest (5-10s per sentence)
# - fast: 2-5s latency, recommended for most, good balance (30-45s per paragraph)
# - standard: 10-20s latency, default quality (1-2 min for longer text)
# - high_quality: 30-60s latency, best quality, slowest (3-5 min for long passages)
```

## 📚 Batch Processing

### Process Long Text Files
```bash
# Basic batch processing
python batch_tts.py input.txt --voice random --preset fast

# With custom voice
python batch_tts.py novel.txt --voice my_voice --preset standard

# Split by character count instead of sentences
python batch_tts.py input.txt --chunk-size 500

# Don't combine chunks
python batch_tts.py input.txt --no-combine
```

## ⚡ Performance Optimization

### Speed Optimizations
```python
from tortoise.api import TextToSpeech

# Use FP16 for half memory usage
tts = TextToSpeech(half=True)

# Enable KV cache for long sequences
tts = TextToSpeech(kv_cache=True)

# Use DeepSpeed (if installed)
tts = TextToSpeech(use_deepspeed=True)

# Combine all optimizations
tts = TextToSpeech(
    half=True,
    kv_cache=True,
    use_deepspeed=True
)
```

### Memory Management
- **GPU Memory**: ~10-20GB VRAM depending on preset
- **System RAM**: 8-16GB for typical usage
- **Reduce memory usage**:
  - Use `ultra_fast` or `fast` presets
  - Process shorter text segments
  - Enable FP16 mode

## 🔧 Advanced Usage

### Custom Voice Directory Structure
```
voices/
├── my_voice/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── sample3.wav
├── narrator/
│   └── narrator_sample.wav
└── character/
    └── character_voice.wav
```

### Python API Examples

#### Generate with Custom Settings
```python
from tortoise.api import TextToSpeech
import torchaudio

tts = TextToSpeech()

# Custom generation parameters
gen_kwargs = {
    "num_autoregressive_samples": 16,  # More samples = better quality
    "diffusion_iterations": 80,         # More iterations = better quality
    "temperature": 0.8,                 # Lower = more consistent
    "length_penalty": 1.0,              # Higher = prefer longer utterances
    "repetition_penalty": 2.0,          # Higher = less repetition
    "top_p": 0.8,                      # Lower = more focused
    "diffusion_temperature": 1.0,       # Higher = more variation
}

wav = tts.tts("Your text here", **gen_kwargs)
torchaudio.save("custom_output.wav", wav.squeeze(0).cpu(), 24000)
```

#### Voice Conditioning
```python
# Load and condition on multiple voice samples
import torch

voice_samples = []
for file in ["voice1.wav", "voice2.wav", "voice3.wav"]:
    audio, sr = torchaudio.load(file)
    # Resample to 22050 Hz if needed
    if sr != 22050:
        resampler = torchaudio.transforms.Resample(sr, 22050)
        audio = resampler(audio)
    voice_samples.append(audio)

# Generate with voice conditioning
wav = tts.tts_with_preset(
    "Your text here",
    voice_samples=voice_samples,
    preset='standard'
)
```

## 🎯 Real-Time/Streaming (Experimental)

### Start Streaming Server
```python
# In one terminal
python -m tortoise.socket_server

# The server listens on port 5000
# You can send text via websocket and receive audio chunks
```

### Client Example
```python
import asyncio
import websockets
import json

async def generate_speech():
    async with websockets.connect('ws://localhost:5000') as websocket:
        request = {
            'text': 'Hello, this is streaming TTS',
            'preset': 'ultra_fast'
        }
        await websocket.send(json.dumps(request))

        # Receive audio chunks
        while True:
            chunk = await websocket.recv()
            if chunk == b'END':
                break
            # Process audio chunk
            process_audio_chunk(chunk)
```

## 🔍 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Monitor GPU usage
nvidia-smi

# Solutions:
# 1. Use faster preset
# 2. Enable FP16 mode
# 3. Process shorter texts
# 4. Restart to clear memory
```

#### Slow Generation
- First run downloads models (~5GB)
- Use `ultra_fast` or `fast` presets
- Enable GPU acceleration (verify with `torch.cuda.is_available()`)
- Consider the Tortoise-TTS-Fastest fork for 5x speed

#### Poor Audio Quality
- Use more/better voice samples for cloning
- Try `standard` or `high_quality` presets
- Generate multiple samples and pick the best
- Ensure voice samples are clean (no background noise)

### File Access from Windows
```powershell
# WSL files accessible from Windows at:
\\wsl$\Ubuntu-22.04\home\[username]\[project_path]

# Copy files to Windows
cp output.wav /mnt/c/Users/[username]/Desktop/
```

## 📊 Performance Expectations

### Generation Speed (RTX A6000)
| Preset | Latency | Quality | Use Case |
|--------|---------|---------|----------|
| ultra_fast | 0.5-1s | Low | Real-time demos |
| fast | 2-5s | Good | Most uses |
| standard | 10-20s | High | Production |
| high_quality | 30-60s | Best | Final output |



### Resource Usage
- **First run**: 5-10 min (model downloads)
- **VRAM**: 10-20GB depending on preset
- **Disk**: 5-10GB for model cache

## 🛠️ Maintenance

### Update Packages
```bash
source tortoise-venv/bin/activate
pip install --upgrade tortoise-tts
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Clear Model Cache
```bash
# Models stored in:
rm -rf ~/.cache/huggingface/hub/models--*
```

### Backup Models
```bash
# Backup downloaded models to avoid re-downloading
tar -czf tortoise_models_backup.tar.gz ~/.cache/huggingface/
```

## 💡 Tips & Best Practices

1. **Voice Cloning Quality**
   - Use 3-5 diverse samples (different tones/speeds)
   - Keep samples 5-15 seconds each
   - Ensure clean audio (no background noise)
   - Same speaker throughout all samples

2. **Text Preparation**
   - Add punctuation for natural pauses
   - Break long texts into paragraphs
   - Use phonetic spelling for unusual words
   - Avoid special characters

3. **Performance**
   - Start with `fast` preset, adjust as needed
   - Process in batches for long texts
   - Use FP16 mode for memory savings
   - Monitor GPU temperature during long sessions

4. **Quality vs Speed Trade-off**
   - Development: Use `ultra_fast`/`fast`
   - Testing: Use `standard`
   - Production: Use `high_quality`

## 📚 Resources

- [Official Tortoise-TTS GitHub](https://github.com/neonbjb/tortoise-tts)
- [Tortoise-TTS-Fast Fork](https://github.com/manmaynakhashi/tortoise-tts-fastest) (5x speed)
- [Hugging Face Models](https://huggingface.co/models?search=tortoise)

---

**Remember**: "Work to your potential, not your quota. Make the most of yourself."

This setup provides production-ready TTS with voice cloning on your RTX A6000!

---

## Appendix: Complete Preset Parameters Reference

For a comprehensive deep dive into all preset parameters, their effects, tuning guides, and examples, see the main [README.md](../README.md#-complete-preset-parameters-reference).

### Quick Reference

**Preset-Specific Parameters:**
| Preset | `num_autoregressive_samples` | `diffusion_iterations` | `cond_free` |
|--------|----------------------------:|----------------------:|:-----------:|
| ultra_fast | 16 | 30 | `False` |
| fast | 96 | 80 | `True` |
| standard | 256 | 200 | `True` |
| high_quality | 256 | 400 | `True` |

**Default Parameters (applied to all presets):**
- `temperature`: 0.8
- `length_penalty`: 1.0
- `repetition_penalty`: 2.0
- `top_p`: 0.8
- `cond_free_k`: 2.0
- `diffusion_temperature`: 1.0

See [README.md](../README.md) for detailed explanations of each parameter, tuning recommendations, and examples.