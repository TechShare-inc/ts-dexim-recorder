"""Shared config helpers for the dexim-recorder CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEXIM_CONFIG_DIR_ENV = "DEXIM_CONFIG_DIR"


def get_config_dir(config_dir: str | None = None) -> Path:
    """Resolve the CLI config root directory.

    Args:
        config_dir: Explicit override; falls back to ``DEXIM_CONFIG_DIR``
            env var or ``./config``.

    Returns:
        Absolute resolved Path.
    """
    raw_path = config_dir or os.environ.get(DEXIM_CONFIG_DIR_ENV) or "./config"
    return Path(raw_path).expanduser().resolve()


def get_recorder_config_dir(config_dir: str | None = None) -> Path:
    """Return the directory that stores named recorder configs.

    Args:
        config_dir: Config root directory override.

    Returns:
        ``{config_dir}/dexim/recorder/`` as an absolute Path.
    """
    return get_config_dir(config_dir) / "dexim" / "recorder"


def resolve_config_path(name_or_path: str, config_dir: str | None = None) -> Path:
    """Resolve a config name or file path to an actual YAML path.

    Resolution order:

    1. Existing explicit file path.
    2. ``{config_dir}/dexim/recorder/{name}.yaml``
    3. ``{config_dir}/dexim/{name}.yaml``

    Args:
        name_or_path: Logical config name or explicit YAML path.
        config_dir: Config root directory override.

    Returns:
        Resolved absolute Path to the YAML file.

    Raises:
        FileNotFoundError: If the config cannot be found.
    """
    candidate = Path(name_or_path).expanduser()
    if candidate.suffix in {".yaml", ".yml"} or candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Config file not found: {resolved}")

    config_root = get_config_dir(config_dir)
    search_paths = [
        config_root / "dexim" / "recorder" / f"{name_or_path}.yaml",
        config_root / "dexim" / f"{name_or_path}.yaml",
    ]
    for path in search_paths:
        if path.exists():
            return path.resolve()

    searched = ", ".join(str(path) for path in search_paths)
    raise FileNotFoundError(f"Config '{name_or_path}' not found. Searched: {searched}")


def get_config_yaml_path(name: str, config_dir: str | None = None) -> Path:
    """Return the expected path for a named recorder config.

    Args:
        name: Logical config name.
        config_dir: Config root directory override.

    Returns:
        ``{config_dir}/dexim/recorder/{name}.yaml`` as an absolute Path.
    """
    return get_recorder_config_dir(config_dir) / f"{name}.yaml"


def list_named_configs(config_dir: str | None = None) -> list[str]:
    """List all named recorder configuration files.

    Args:
        config_dir: Config root directory override.

    Returns:
        Sorted list of config names (without ``.yaml`` extension).
    """
    recorder_dir = get_recorder_config_dir(config_dir)
    if not recorder_dir.is_dir():
        return []
    return sorted(p.stem for p in recorder_dir.glob("*.yaml"))


def create_config_yaml(
    name: str,
    data: dict[str, Any],
    config_dir: str | None = None,
) -> Path:
    """Create a new named recorder config YAML.

    Args:
        name: Logical config name (e.g. ``lab-hdf5``).
        data: Config dict to serialize.
        config_dir: Config root directory override.

    Returns:
        Path to the created YAML file.

    Raises:
        FileExistsError: If a config with this name already exists.
    """
    path = get_config_yaml_path(name, config_dir)
    if path.exists():
        raise FileExistsError(
            f"Config '{name}' already exists: {path}\n"
            f"Use 'config edit {name}' to modify it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    return path


def edit_config_yaml(
    name: str,
    updates: dict[str, Any],
    config_dir: str | None = None,
) -> Path:
    """Patch fields in an existing named recorder config YAML.

    Performs a shallow merge: top-level keys in *updates* overwrite
    corresponding keys in the existing file.  Nested dicts are merged
    one level deep.

    Args:
        name: Logical config name.
        updates: Mapping of fields to set/overwrite.
        config_dir: Config root directory override.

    Returns:
        Path to the updated YAML file.

    Raises:
        FileNotFoundError: If the config does not exist.
    """
    path = get_config_yaml_path(name, config_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Config '{name}' not found: {path}\n"
            f"Use 'config new {name}' to create it first."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value

    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    return path


def remove_config_yaml(name: str, config_dir: str | None = None) -> Path:
    """Delete a named recorder config YAML from disk.

    Args:
        name: Logical config name.
        config_dir: Config root directory override.

    Returns:
        Path that was deleted.

    Raises:
        FileNotFoundError: If the config does not exist.
    """
    path = get_config_yaml_path(name, config_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Config '{name}' not found: {path}\n"
            f"Use 'config list' to see available configs."
        )
    path.unlink()
    return path
