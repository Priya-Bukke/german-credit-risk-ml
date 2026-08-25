from pathlib import Path

# This file lives in src/utils/paths.py, so go up 2 levels to reach the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"