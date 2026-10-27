"""Append-only, hash-chained audit log (JSON Lines).

Each entry stores the SHA-256 of the previous entry, so editing or deleting a past
entry breaks the chain and `verify` reports where. This makes tampering detectable;
it does not prevent it (anyone with file access could rewrite the whole chain).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _digest(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def log(self, event: str, details: dict[str, Any], actor: str = "pipeline") -> dict:
        if not event:
            raise ValueError("event name required")
        prev = self.entries()
        entry = {
            "seq": len(prev) + 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event, "actor": actor, "details": details,
            "prev_hash": prev[-1]["hash"] if prev else GENESIS,
        }
        entry["hash"] = _digest(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def verify(self) -> dict[str, Any]:
        prev_hash = GENESIS
        for i, e in enumerate(self.entries(), start=1):
            if e.get("seq") != i or e.get("prev_hash") != prev_hash or _digest(e) != e.get("hash"):
                return {"valid": False, "broken_at_seq": i}
            prev_hash = e["hash"]
        return {"valid": True, "entries": len(self.entries())}
