"""Configuration utilities for MoA.

This module centralizes YAML loading, inheritance, deep merging, variable
substitution, and config snapshot persistence for experiment runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence
import json
import os
import re

import yaml


class Config:
    """Configuration manager for MPSL experiments."""

    _VARIABLE_PATTERN = re.compile(r"\$\{([^}]+)\}")

    def __init__(self, config_dict: Dict[str, Any], source_paths: Optional[Sequence[Path]] = None):
        self._config = config_dict
        self._source_paths = [Path(path) for path in (source_paths or [])]

    @classmethod
    def load(
        cls,
        base_config_path: str = "configs/base.yaml",
        storage_config_path: str = "configs/storage.yaml",
        model_config_path: Optional[str] = None,
        overlay_config_paths: Optional[Sequence[str]] = None,
    ) -> "Config":
        """Load and merge storage/base/model/overlay configs.

        Merge order:
        1. storage config under key ``storage``
        2. base config
        3. model config (optional)
        4. overlay configs in order (optional)
        """

        source_paths: list[Path] = []
        storage_path = Path(storage_config_path)
        base_path = Path(base_config_path)
        storage_config = cls._load_yaml(storage_path)
        base_config = cls._load_with_inheritance(base_path, source_paths)

        merged_config: Dict[str, Any] = {"storage": storage_config}
        merged_config = cls._deep_merge(merged_config, base_config)

        if model_config_path:
            model_path = Path(model_config_path)
            model_config = cls._load_with_inheritance(model_path, source_paths)
            merged_config = cls._deep_merge(merged_config, model_config)

        for overlay_path in overlay_config_paths or []:
            overlay_config = cls._load_with_inheritance(Path(overlay_path), source_paths)
            merged_config = cls._deep_merge(merged_config, overlay_config)

        merged_config = cls._resolve_variables(merged_config)
        return cls(merged_config, source_paths=[storage_path, base_path, *source_paths])

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    @classmethod
    def _load_with_inheritance(cls, path: Path, source_paths: list[Path]) -> Dict[str, Any]:
        config = cls._load_yaml(path)
        source_paths.append(path)

        inherit_paths = config.pop("inherits", None)
        if not inherit_paths:
            return config

        if isinstance(inherit_paths, str):
            inherit_paths = [inherit_paths]
        if not isinstance(inherit_paths, list):
            raise TypeError(f"'inherits' must be a string or list in {path}")

        merged: Dict[str, Any] = {}
        for inherit_path in inherit_paths:
            parent_path = (path.parent / inherit_path).resolve() if not Path(inherit_path).is_absolute() else Path(inherit_path)
            parent_config = cls._load_with_inheritance(parent_path, source_paths)
            merged = cls._deep_merge(merged, parent_config)

        return cls._deep_merge(merged, config)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def _resolve_variables(cls, config: Dict[str, Any], root: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if root is None:
            root = config

        resolved: Dict[str, Any] = {}
        for key, value in config.items():
            if isinstance(value, dict):
                resolved[key] = cls._resolve_variables(value, root)
            elif isinstance(value, list):
                resolved[key] = [cls._resolve_list_item(item, root) for item in value]
            elif isinstance(value, str) and "${" in value:
                resolved[key] = cls._substitute_variable(value, root)
            else:
                resolved[key] = value
        return resolved

    @classmethod
    def _resolve_list_item(cls, item: Any, root: Dict[str, Any]) -> Any:
        if isinstance(item, dict):
            return cls._resolve_variables(item, root)
        if isinstance(item, list):
            return [cls._resolve_list_item(value, root) for value in item]
        if isinstance(item, str) and "${" in item:
            return cls._substitute_variable(item, root)
        return item

    @classmethod
    def _substitute_variable(cls, value: str, root: Dict[str, Any]) -> Any:
        matches = cls._VARIABLE_PATTERN.findall(value)
        if not matches:
            return value

        if value.startswith("${") and value.endswith("}") and len(matches) == 1:
            return cls._get_nested_value(root, matches[0])

        result = value
        for match in matches:
            replacement = str(cls._get_nested_value(root, match))
            result = result.replace(f"${{{match}}}", replacement)
        return result

    @staticmethod
    def _get_nested_value(config: Dict[str, Any], path: str) -> Any:
        keys = path.split(".")
        value: Any = config
        for key in keys:
            if isinstance(value, dict):
                if key not in value:
                    raise KeyError(f"Config key not found: {path}")
                value = value[key]
            else:
                raise KeyError(f"Cannot traverse non-dict at {key!r} in {path!r}")
        return value

    def get(self, path: str, default: Any = None) -> Any:
        try:
            return self._get_nested_value(self._config, path)
        except KeyError:
            return default

    def require(self, path: str) -> Any:
        return self._get_nested_value(self._config, path)

    def set(self, path: str, value: Any) -> None:
        keys = path.split(".")
        current = self._config
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        self._config = self._resolve_variables(self._config)

    def merge(self, overlay: Dict[str, Any]) -> "Config":
        merged = self._deep_merge(self._config, overlay)
        merged = self._resolve_variables(merged)
        return Config(merged, source_paths=self._source_paths)

    def snapshot(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(self._config, file, sort_keys=False)
        return path

    def snapshot_json(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self._config, file, indent=2)
        return path

    def to_dict(self) -> Dict[str, Any]:
        return self._config

    @property
    def source_paths(self) -> list[Path]:
        return list(self._source_paths)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def __repr__(self) -> str:
        return f"Config(keys={list(self._config.keys())!r})"


def load_config(
    base_config: str = "configs/base.yaml",
    storage_config: str = "configs/storage.yaml",
    model_config: Optional[str] = None,
    overlay_configs: Optional[Iterable[str]] = None,
) -> Config:
    """Convenience wrapper for ``Config.load``."""

    return Config.load(
        base_config_path=base_config,
        storage_config_path=storage_config,
        model_config_path=model_config,
        overlay_config_paths=list(overlay_configs or []),
    )
