"""
Tutortortie TTS - A Tortoise-TTS wrapper for easy text-to-speech generation
"""

__version__ = "0.1.0"
__author__ = "TutortoieTTS Team"

# Make CLI modules easily importable
from . import cli
from . import utils

__all__ = ["cli", "utils", "__version__"]
