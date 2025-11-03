# Refactor Notes - Post-Reorganization Fixes

## Overview
After reorganizing the project structure, several path-related issues were identified and fixed to ensure all tests and scripts work correctly from their new locations.

## Changes Made

### 1. Test Files Path Fixes

#### tests/test_tts.py
**Issue**: Output file was being written to relative path, which would create files in the wrong location.

**Fix**:
- Added `PROJECT_ROOT = Path(__file__).parent.parent` to get project root
- Changed output path from `"test_output_uf.wav"` to `PROJECT_ROOT / "outputs" / "test_output_uf.wav"`
- Ensured outputs directory is created before writing

**Impact**: Test outputs now correctly go to `outputs/` directory regardless of where the script is run from.

#### tests/scaffolds/test_emma_scaffold.py
**Issue**: `OUTPUT_DIR = Path("test_outputs")` was using relative path, which would create directory in wrong location.

**Fix**:
- Added `PROJECT_ROOT = Path(__file__).parent.parent.parent` to get project root (3 levels up)
- Changed to `OUTPUT_DIR = PROJECT_ROOT / "outputs"`
- All test outputs now go to centralized `outputs/` directory

**Impact**: All comprehensive test results and WAV files are now saved to the project-level `outputs/` directory.

### 2. Script Updates

#### scripts/activate.sh
**Issues**:
- Path to virtual environment was relative and wouldn't work from scripts/ directory
- Example commands referenced old file locations

**Fixes**:
- Added `PROJECT_ROOT` variable calculation
- Updated venv path to `$PROJECT_ROOT/tortoise-venv/bin/activate`
- Updated all example commands to use new paths:
  - `python src/tutortoietts/cli/generate.py 'Your text here'`
  - `python tests/test_tts.py`
  - `python src/tutortoietts/cli/clone.py --help`
  - `python src/tutortoietts/cli/batch.py --help`

**Impact**: Activation script now works correctly and provides accurate usage examples.

### 3. Configuration Updates

#### .gitignore
**Added**:
- `outputs/` - for generated audio files
- `test_outputs/` - legacy path, in case any old configs reference it
- Updated `logs/` to be more specific

**Impact**: Git now properly ignores generated files in the new structure.

## Testing Recommendations

Before running tests, ensure:

1. **Virtual Environment Setup**:
   ```bash
   source scripts/activate.sh
   ```

2. **Run Basic Test**:
   ```bash
   python tests/test_tts.py
   ```
   Should create `outputs/test_output_uf.wav`

3. **Run Comprehensive Test**:
   ```bash
   python tests/scaffolds/test_emma_scaffold.py
   ```
   Should create multiple files in `outputs/` directory with JSON results

4. **Run CLI Tools**:
   ```bash
   python src/tutortoietts/cli/generate.py "Hello world" --output outputs/hello.wav
   ```

## Path Resolution Summary

All test files now use absolute path resolution:

| File | Root Calculation | Usage |
|------|------------------|-------|
| `tests/test_tts.py` | `Path(__file__).parent.parent` | 1 level up to project root |
| `tests/scaffolds/test_emma_scaffold.py` | `Path(__file__).parent.parent.parent` | 2 levels up to project root |
| `scripts/activate.sh` | `cd "$SCRIPT_DIR/.."` | Shell navigation to parent |

## Benefits

1. **Location Independence**: Tests can be run from any directory
2. **Centralized Outputs**: All generated files go to `outputs/` directory
3. **Clear Organization**: No scattered output files in test directories
4. **Git Clean**: All outputs properly ignored by git

## Potential Issues to Watch For

1. **Virtual Environment**: If venv doesn't exist, scripts will fail - run `scripts/setup.sh` first
2. **Import Paths**: If running from different directories, Python may need the project root in PYTHONPATH
3. **Voice Samples**: Test files assume Tortoise-TTS built-in voices are available

## Next Steps

1. Verify virtual environment setup works: `bash scripts/setup.sh`
2. Test all CLI tools with new paths
3. Consider adding setup.py or pyproject.toml for proper package installation
4. Add pytest configuration for easier test running
