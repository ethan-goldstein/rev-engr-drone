"""The info card beside the locked person (the FACE LOCK panel): plain text from a file you write.

It doesn't recognize anyone. Whoever the drone locks onto gets this card, so swap the file
(--profile friend.json) when following someone else. profile.json holds "LABEL": "value" pairs,
shown in order, e.g. {"NAME": "JANE DOE", "OCCUPATION": "STUDENT / SWE"}.
"""
from __future__ import annotations

import json
from pathlib import Path

MAX_ROWS = 6
MAX_LABEL = 12
MAX_VALUE = 22  # the drone view is only 640 px wide


def _clean(text: str, limit: int) -> str:
    # OpenCV's fonts only draw ASCII; anything else would show up as "?".
    return text.encode("ascii", "replace").decode().upper().strip()[:limit]


def load_profile(path: str | Path) -> list[tuple[str, str]]:
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(f"--profile: couldn't read {path}: {e}")
    if not isinstance(data, dict) or not all(isinstance(v, (str, int, float)) for v in data.values()):
        raise SystemExit(f'--profile: {path} should look like {{"NAME": "...", "OCCUPATION": "..."}}')
    rows = [(_clean(str(k), MAX_LABEL), _clean(str(v), MAX_VALUE)) for k, v in data.items()]
    if len(rows) > MAX_ROWS:
        print(f"--profile: showing the first {MAX_ROWS} of {len(rows)} lines from {path}")
    return rows[:MAX_ROWS]
