"""Pytest configuration for v1 tests.

Adds repo root to sys.path so namespaced imports (v0.src.*, v1.src.*,
shared.*) resolve when tests are invoked from any cwd.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
