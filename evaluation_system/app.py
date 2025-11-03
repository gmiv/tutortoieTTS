#!/usr/bin/env python3
"""
TTS Output Evaluation System
FastAPI server for listening to and rating TTS outputs
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import uvicorn

# Setup paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RATINGS_FILE = BASE_DIR / "ratings.json"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Create FastAPI app
app = FastAPI(title="TTS Evaluation System")

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global state
class EvaluationState:
    def __init__(self):
        self.test_results = {}
        self.wav_files = []
        self.current_index = 0
        self.ratings = {}
        self.load_test_results()
        self.load_ratings()

    def load_test_results(self):
        """Load test results from the most recent test results JSON."""
        test_files = list(OUTPUTS_DIR.glob("test_results_*.json"))
        if not test_files:
            print("No test results found!")
            return

        # Get the most recent file
        latest_test_file = max(test_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading test results from: {latest_test_file}")

        with open(latest_test_file, 'r') as f:
            self.test_results = json.load(f)

        # Extract WAV files from individual results
        if 'individual_results' in self.test_results:
            for result in self.test_results['individual_results']:
                if result.get('success') and 'filename' in result:
                    wav_path = OUTPUTS_DIR / result['filename']
                    if wav_path.exists():
                        self.wav_files.append({
                            'filename': result['filename'],
                            'path': str(wav_path),
                            'test_name': result.get('test_name', 'Unknown'),
                            'preset': result.get('preset', 'unknown'),
                            'params': result.get('params', {}),
                            'elapsed_time': result.get('elapsed_time', 0),
                            'audio_duration': result.get('audio_duration', 0),
                            'file_size_kb': result.get('file_size_kb', 0)
                        })

        print(f"Found {len(self.wav_files)} WAV files to evaluate")

    def load_ratings(self):
        """Load existing ratings from file."""
        if RATINGS_FILE.exists():
            with open(RATINGS_FILE, 'r') as f:
                self.ratings = json.load(f)
        else:
            self.ratings = {
                'ratings': {},
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_files': len(self.wav_files),
                    'rated_count': 0
                }
            }

    def save_ratings(self):
        """Save ratings to file."""
        self.ratings['metadata']['updated_at'] = datetime.now().isoformat()
        self.ratings['metadata']['rated_count'] = len(self.ratings['ratings'])

        with open(RATINGS_FILE, 'w') as f:
            json.dump(self.ratings, f, indent=2)

    def add_rating(self, filename: str, rating: str, notes: str = ""):
        """Add or update a rating."""
        self.ratings['ratings'][filename] = {
            'rating': rating,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        self.save_ratings()

    def get_current_file(self):
        """Get the current WAV file info."""
        if 0 <= self.current_index < len(self.wav_files):
            return self.wav_files[self.current_index]
        return None

    def get_rating(self, filename: str):
        """Get existing rating for a file."""
        return self.ratings['ratings'].get(filename)

# Initialize state
state = EvaluationState()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main evaluation interface."""
    current_file = state.get_current_file()

    if not current_file:
        return templates.TemplateResponse("no_files.html", {"request": request})

    # Get existing rating if any
    existing_rating = state.get_rating(current_file['filename'])

    # Calculate progress
    rated_count = len(state.ratings['ratings'])
    total_count = len(state.wav_files)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_file": current_file,
        "current_index": state.current_index + 1,
        "total_files": total_count,
        "rated_count": rated_count,
        "existing_rating": existing_rating,
        "progress_percentage": (rated_count / total_count * 100) if total_count > 0 else 0
    })

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve WAV file."""
    file_path = OUTPUTS_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"}
    )

@app.post("/rate")
async def rate_audio(request: Request):
    """Submit rating for current audio."""
    data = await request.json()

    filename = data.get('filename')
    rating = data.get('rating')
    notes = data.get('notes', '')

    if not filename or not rating:
        raise HTTPException(status_code=400, detail="Missing filename or rating")

    if rating not in ['up', 'down', 'skip']:
        raise HTTPException(status_code=400, detail="Invalid rating")

    # Save rating
    state.add_rating(filename, rating, notes)

    # Return updated stats
    return JSONResponse({
        'success': True,
        'rated_count': len(state.ratings['ratings']),
        'total_count': len(state.wav_files)
    })

@app.post("/navigate")
async def navigate(request: Request):
    """Navigate to next/previous file."""
    data = await request.json()
    direction = data.get('direction')

    if direction == 'next':
        if state.current_index < len(state.wav_files) - 1:
            state.current_index += 1
    elif direction == 'previous':
        if state.current_index > 0:
            state.current_index -= 1
    elif direction == 'goto':
        index = data.get('index', 0)
        if 0 <= index < len(state.wav_files):
            state.current_index = index

    return JSONResponse({
        'success': True,
        'current_index': state.current_index,
        'redirect': '/'
    })

@app.get("/stats")
async def get_stats():
    """Get evaluation statistics."""
    ratings_summary = {'up': 0, 'down': 0, 'skip': 0}

    for file_rating in state.ratings['ratings'].values():
        rating = file_rating['rating']
        if rating in ratings_summary:
            ratings_summary[rating] += 1

    # Stats by preset
    preset_stats = {}
    for wav_file in state.wav_files:
        preset = wav_file['preset']
        filename = wav_file['filename']

        if preset not in preset_stats:
            preset_stats[preset] = {'up': 0, 'down': 0, 'skip': 0, 'total': 0}

        preset_stats[preset]['total'] += 1

        if filename in state.ratings['ratings']:
            rating = state.ratings['ratings'][filename]['rating']
            preset_stats[preset][rating] += 1

    return JSONResponse({
        'total_files': len(state.wav_files),
        'rated_count': len(state.ratings['ratings']),
        'ratings_summary': ratings_summary,
        'preset_stats': preset_stats,
        'files': [
            {
                'filename': f['filename'],
                'test_name': f['test_name'],
                'preset': f['preset'],
                'rating': state.ratings['ratings'].get(f['filename'], {}).get('rating', 'unrated')
            }
            for f in state.wav_files
        ]
    })

@app.get("/export")
async def export_ratings():
    """Export ratings as downloadable JSON."""
    export_data = {
        'ratings': state.ratings,
        'test_results_summary': {
            'total_tests': len(state.wav_files),
            'test_configuration': state.test_results.get('metadata', {}).get('test_configuration', {}),
            'system_info': state.test_results.get('metadata', {}).get('system_info', {})
        },
        'export_timestamp': datetime.now().isoformat()
    }

    return JSONResponse(
        content=export_data,
        headers={
            'Content-Disposition': f'attachment; filename=ratings_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        }
    )

if __name__ == "__main__":
    print(f"Starting TTS Evaluation Server...")
    print(f"Output directory: {OUTPUTS_DIR}")
    print(f"Ratings file: {RATINGS_FILE}")
    print(f"Found {len(state.wav_files)} files to evaluate")
    print(f"\nOpen http://localhost:8000 in your browser")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)