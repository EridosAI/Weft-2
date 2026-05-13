"""pytest configuration — adds the repo root to sys.path so `import src.*` works."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
