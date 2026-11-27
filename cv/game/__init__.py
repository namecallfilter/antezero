"""Balatro game simulation engine package."""
from pathlib import Path
import sys

# Ensure the game directory is present on sys.path so that internal module
# references (e.g. `import balatro_models` or `from engine import ...`)
# resolve properly in all environments (CLI, unittest discovery, IDEs).
_GAME_DIR = str(Path(__file__).resolve().parent)
if _GAME_DIR not in sys.path:
    sys.path.insert(0, _GAME_DIR)
