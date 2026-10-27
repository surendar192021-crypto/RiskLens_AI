"""A small file-based model registry (models/registry.json).

Each entry records what a model is, how it was made and how well it did, plus the
SHA-256 of its file, so a loaded model can be checked against what was registered.
Statuses: deployed, challenger, baseline, experiment, archived. At most one model
is "deployed" at a time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.load_data import file_sha256

STATUSES = {"deployed", "challenger", "baseline", "experiment", "archived"}
REQUIRED = {"model_version", "model_file", "status", "created_at"}


class ModelRegistry:
    def __init__(self, path: Path, models_dir: Path):
        self.path = Path(path)
        self.models_dir = Path(models_dir)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))["models"]

    def _save(self, entries: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), "models": entries}
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def register(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Add or replace an entry (keyed by model_version); records the file hash."""
        missing = REQUIRED - set(entry)
        if missing:
            raise ValueError(f"registry entry missing {sorted(missing)}")
        if entry["status"] not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")
        model_path = self.models_dir / entry["model_file"]
        if not model_path.exists():
            raise FileNotFoundError(f"model file not found: {model_path}")
        entry = {**entry, "sha256": file_sha256(model_path)}
        entries = [e for e in self.load() if e["model_version"] != entry["model_version"]]
        if entry["status"] == "deployed":
            for e in entries:
                if e["status"] == "deployed":
                    e["status"] = "archived"
        entries.append(entry)
        self._save(entries)
        return entry

    def deployed(self) -> dict[str, Any] | None:
        d = [e for e in self.load() if e["status"] == "deployed"]
        return d[0] if d else None

    def verify(self, model_version: str) -> bool:
        """True if the file on disk still matches the registered hash."""
        for e in self.load():
            if e["model_version"] == model_version:
                return file_sha256(self.models_dir / e["model_file"]) == e["sha256"]
        raise KeyError(f"unknown model version {model_version!r}")
