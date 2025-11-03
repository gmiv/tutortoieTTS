# TutortoieTTS Project Structure

This document describes the organized structure of the TutortoieTTS project.

## Directory Layout

```
tutortoieTTS/
├── src/tutortoietts/              # Main source code package
│   ├── __init__.py                # Package initialization
│   ├── cli/                       # CLI command modules
│   │   ├── __init__.py
│   │   ├── generate.py            # Main TTS generation (formerly generate_speech.py)
│   │   ├── generate_quiet.py      # Quiet version (formerly generate_speech_quiet.py)
│   │   ├── batch.py               # Batch processing (formerly batch_tts.py)
│   │   └── clone.py               # Voice cloning (formerly clone_voice.py)
│   └── utils/                     # Utility modules
│       ├── __init__.py
│       └── text_utils.py          # Text utilities (formerly squeeze_to_oneliner.py)
│
├── tests/                         # All test files
│   ├── __init__.py
│   ├── test_tts.py                # Main TTS tests
│   └── scaffolds/                 # Test scaffolds and utilities
│       ├── __init__.py
│       ├── test_emma_scaffold.py  # Emma voice test scaffold
│       └── TEST_SCAFFOLD_README.md # Scaffold documentation
│
├── docs/                          # All documentation
│   ├── PLAN.md                    # Project planning
│   ├── USAGE_GUIDE.md             # Usage documentation
│   └── Delivery.md                # Delivery notes
│
├── scripts/                       # Setup and utility scripts
│   ├── setup.sh                   # Installation and setup
│   └── activate.sh                # Environment activation
│
├── data/                          # Data files
│   └── voices/                    # Voice sample data
│       ├── custom/                # Custom voice samples
│       └── samples/               # Sample voice files
│
├── outputs/                       # Generated output files
│   └── test_output.wav            # Test outputs
│
├── logs/                          # Log files
│   └── tts_test.log               # TTS test logs
│
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
└── tortoise-venv/                 # Virtual environment (not tracked)
```

## Module Organization

### Source Code (`src/tutortoietts/`)
All source code is organized as a proper Python package:
- **cli/**: Command-line interface modules for different TTS operations
- **utils/**: Utility functions and helper modules

### Tests (`tests/`)
All test files are organized together:
- Unit tests at the root level
- Test scaffolds in dedicated subdirectory

### Documentation (`docs/`)
All project documentation in one place:
- Project planning and design docs
- Usage guides and examples
- Delivery and release notes

### Scripts (`scripts/`)
Installation and utility scripts:
- Environment setup
- Activation scripts

### Data (`data/`)
Application data files:
- Voice samples for cloning
- Training data (if any)

### Outputs (`outputs/`)
Generated files from TTS operations

### Logs (`logs/`)
Application logs and debugging information

## Import Paths

With the new structure, import paths are:

```python
# Import CLI modules
from tutortoietts.cli import generate, batch, clone, generate_quiet

# Import utilities
from tutortoietts.utils import text_utils
```

## Running Commands

The CLI scripts can be run from the project root:

```bash
# Main TTS generation
python src/tutortoietts/cli/generate.py "Hello world"

# Quiet generation
python src/tutortoietts/cli/generate_quiet.py "Hello world"

# Batch processing
python src/tutortoietts/cli/batch.py input.txt

# Voice cloning
python src/tutortoietts/cli/clone.py --name myvoice --samples data/voices/custom/*.wav --text "Hello"
```

## Benefits of This Structure

1. **Clear separation of concerns**: Source code, tests, docs, and data are clearly separated
2. **Proper Python package**: The src layout follows Python best practices
3. **Easy imports**: Clean import paths for all modules
4. **Scalability**: Easy to add new features, tests, and documentation
5. **Professional**: Follows industry-standard project structure
