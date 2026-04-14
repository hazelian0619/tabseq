from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def choose(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            data = json.load(f)
        elif ext in {".yaml", ".yml"}:
            if yaml is None:
                raise ImportError("PyYAML is required to load YAML config files")
            data = yaml.safe_load(f)
        else:
            raise ValueError(f"unsupported config format: {path}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def resolve_section(config: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
    if not config:
        return {}
    section = config.get(name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return section


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
