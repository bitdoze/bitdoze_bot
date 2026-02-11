from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    path: Path

    def get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self.data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def resolve_path(self, value: str | Path | None, *, default: str | Path | None = None) -> Path:
        candidate = value
        if isinstance(candidate, str) and not candidate.strip():
            candidate = None
        if candidate is None:
            candidate = default
        if candidate is None:
            raise ValueError("Path value is required")

        resolved = Path(str(candidate)).expanduser()
        if not resolved.is_absolute():
            resolved = (self.base_dir / resolved).resolve()
        return resolved

    def resolve_optional_path(
        self,
        value: str | Path | None,
        *,
        default: str | Path | None = None,
    ) -> Path | None:
        candidate = value
        if isinstance(candidate, str) and not candidate.strip():
            candidate = None
        if candidate is None and default is None:
            return None
        return self.resolve_path(candidate, default=default)


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    raw = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")
    return Config(data=data, path=config_path)
