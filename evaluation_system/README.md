# TTS Output Evaluation System

A web-based evaluation system for rating Text-to-Speech outputs from Tortoise-TTS tests.

## Features

- **Web Interface**: Clean, modern interface for listening and rating WAV files
- **Keyboard Shortcuts**: Fast evaluation with keyboard controls
- **Progress Tracking**: See how many files you've rated
- **Rating Storage**: All ratings saved to JSON for analysis
- **Test Integration**: Automatically loads files from test results
- **Notes Support**: Add optional notes to each rating
- **Statistics**: View rating statistics and export results

## Quick Start

### Option 1: Using Launch Script (Recommended)

```bash
cd evaluation_system
chmod +x launch.sh
./launch.sh
```

### Option 2: Manual Setup

```bash
cd evaluation_system

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

Then open your browser to: http://localhost:8000

## Usage

### Rating Files

1. The system will automatically load all WAV files from `outputs/`
2. Listen to each file using the audio player
3. Rate using:
   - **👍 Good** - Audio quality is acceptable
   - **👎 Bad** - Audio has issues
   - **⏭️ Skip** - Uncertain or want to revisit later
4. Add optional notes about specific issues or observations
5. System auto-advances to next file after rating

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Space** | Play/Pause audio |
| **↑** | Rate as Good |
| **↓** | Rate as Bad |
| **S** | Skip file |
| **←** | Previous file |
| **→** | Next file |

### Viewing Statistics

Visit http://localhost:8000/stats to see:
- Total files evaluated
- Rating distribution
- Statistics by preset
- Individual file ratings

### Exporting Results

Visit http://localhost:8000/export to download:
- All ratings with timestamps
- Test configuration metadata
- System information

## File Structure

```
evaluation_system/
├── app.py              # FastAPI server
├── requirements.txt    # Python dependencies
├── launch.sh          # Launch script
├── README.md          # This file
├── ratings.json       # Saved ratings (created automatically)
├── templates/         # HTML templates
│   ├── index.html     # Main evaluation interface
│   └── no_files.html  # No files found page
└── static/           # Static files (if needed)
```

## Ratings File Format

Ratings are saved to `ratings.json`:

```json
{
  "ratings": {
    "filename.wav": {
      "rating": "up",
      "notes": "Clear pronunciation, good emotion",
      "timestamp": "2024-11-03T12:00:00"
    }
  },
  "metadata": {
    "created_at": "2024-11-03T11:00:00",
    "total_files": 20,
    "rated_count": 15
  }
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main evaluation interface |
| `/audio/{filename}` | GET | Serve WAV file |
| `/rate` | POST | Submit rating |
| `/navigate` | POST | Navigate between files |
| `/stats` | GET | View statistics |
| `/export` | GET | Export ratings as JSON |

## Troubleshooting

### No files found
- Make sure you've run tests that generate WAV files
- Check that files are in `outputs/` directory
- Run: `python tests/scaffolds/test_emma_scaffold.py`

### Audio won't play
- Check browser supports WAV format
- Verify file permissions
- Try different browser (Chrome/Firefox/Edge)

### Server won't start
- Check port 8000 is not in use
- Verify Python 3.8+ is installed
- Ensure all dependencies are installed

## Development

### Adding Features

The system is built with:
- **FastAPI** - Modern Python web framework
- **Jinja2** - HTML templating
- **Vanilla JS** - No framework dependencies

### Customizing Interface

Edit `templates/index.html` to:
- Change styling (inline CSS)
- Add new rating categories
- Modify keyboard shortcuts

### Extending Ratings

Modify `app.py` to:
- Add more rating options
- Include automatic quality metrics
- Integrate with other analysis tools

## Tips for Evaluation

1. **Listen to entire clip** before rating
2. **Focus on**:
   - Pronunciation clarity
   - Natural flow
   - Emotional appropriateness
   - Audio artifacts or glitches
3. **Use notes** for specific issues:
   - "Robotic at 0:15"
   - "Mispronounced 'thoroughly'"
   - "Good emotion but clipping"
4. **Be consistent** in your criteria
5. **Take breaks** to avoid ear fatigue

## Integration with Tests

The system automatically reads from test result JSON files:
1. Finds latest `test_results_*.json` in `outputs/`
2. Extracts successful test files
3. Displays test metadata (preset, parameters, timing)
4. Links ratings back to specific test conditions

This helps identify which presets and parameters produce the best quality.