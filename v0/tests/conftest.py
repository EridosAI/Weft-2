"""pytest configuration — adds the repo root to sys.path so `from v0.src.*` and `from shared.*` imports work."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
