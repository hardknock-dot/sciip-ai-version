"""
ui_prefs.py

Tiny, dependency-free store for dashboard view preferences (e.g. which Item Master
columns to show). Saved as JSON beside the database so a chosen view survives
browser refreshes and restarts. Never raises: a missing or unreadable file simply
means "use the defaults".
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "data" / "ui_prefs.json"


def load(path: Path | str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save(updates: dict, path: Path | str | None = None) -> bool:
    """Merge `updates` into the stored preferences (atomic write). Returns success."""
    p = Path(path) if path else DEFAULT_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        merged = {**load(p), **updates}
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        os.replace(tmp, p)
        return True
    except Exception:
        return False
