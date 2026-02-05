from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise ImportError("PyYAML is required for YAML config files.") from exc
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML config root must be a mapping")
        return data
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("JSON config root must be an object")
        return data
    raise ValueError(f"unsupported config format: {ext}")


def resolve_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    if not config:
        return {}
    if section in config:
        sec = config.get(section) or {}
        if not isinstance(sec, dict):
            raise ValueError(f"config section '{section}' must be a mapping")
        return sec
    if all(isinstance(v, dict) for v in config.values()):
        return {}
    return config


def choose(cli_value: Any, config_value: Any, default_value: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default_value


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
