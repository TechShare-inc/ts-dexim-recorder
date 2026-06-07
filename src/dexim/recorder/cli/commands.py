"""Click commands for the dexim-recorder CLI."""

from __future__ import annotations

import datetime
from pathlib import Path

import rich_click as click
from dexim.cli.common import get_console, handle_cli_error, make_table

from .config_utils import (
    create_config_yaml,
    edit_config_yaml,
    get_config_yaml_path,
    list_named_configs,
    remove_config_yaml,
    resolve_config_path,
)

_IMPORT_HINT = "[muted]Install dependencies with: pip install dexim-recorder[/]"


def _path_value(p: Path | None) -> str | None:
    return str(p) if p is not None else None


# -- run -----------------------------------------------------------------------


@click.command()
@click.option(
    "--config",
    "-c",
    default=None,
    help="YAML config file or named config. Overrides other options if provided.",
)
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@click.option(
    "--format",
    "storage_format",
    type=click.Choice(["hdf5", "lerobot"]),
    default="hdf5",
    show_default=True,
    help="Storage backend format.",
)
@click.option(
    "--output-dir",
    default="output/episodes",
    show_default=True,
    help="Directory for episode files.",
)
@click.option(
    "--endpoint",
    "-e",
    "endpoints",
    multiple=True,
    help="ZMQ data endpoint to subscribe to (repeatable).",
)
@click.option(
    "--default-task",
    default="",
    help="Default task name embedded in episode metadata.",
)
@click.option(
    "--fps",
    type=int,
    default=30,
    show_default=True,
    help="Frame rate for LeRobot dataset.",
)
@click.option(
    "--node-id",
    default="recorder",
    show_default=True,
    help="Unique node identifier.",
)
@handle_cli_error
def run(
    config: str | None,
    config_dir: Path | None,
    storage_format: str,
    output_dir: str,
    endpoints: tuple[str, ...],
    default_task: str,
    fps: int,
    node_id: str,
) -> None:
    """Start the data recorder node."""
    console = get_console()

    try:
        from dexim.recorder.config import RecorderNodeConfig, load_config
        from dexim.recorder.node import DataRecorderNode
    except ImportError as exc:
        console.print(f"[error]Missing dependency: {exc}[/]")
        console.print(_IMPORT_HINT)
        raise SystemExit(1) from exc

    dir_value = _path_value(config_dir)

    if config:
        cfg = load_config(str(resolve_config_path(config, dir_value)))
    else:
        if not endpoints:
            raise click.UsageError(
                "Provide --endpoint/-e (at least one) or --config/-c."
            )
        cfg = RecorderNodeConfig(
            node_id=node_id,
            data_endpoints=list(endpoints),
            storage_format=storage_format,
            output_dir=output_dir,
            default_task=default_task,
            fps=fps,
        )

    console.print(
        f"[info]Starting recorder in [key]{cfg.storage_format}[/key] mode...[/]"
    )
    console.print(f"  Node ID:    [key]{cfg.node_id}[/key]")
    console.print(f"  Format:     [key]{cfg.storage_format}[/key]")
    console.print(f"  Output:     [key]{cfg.output_dir}[/key]")
    console.print(f"  Endpoints:  [key]{', '.join(cfg.data_endpoints)}[/key]")
    if cfg.default_task:
        console.print(f"  Default Task:       [key]{cfg.default_task}[/key]")

    try:
        node = DataRecorderNode(config=cfg)
        console.print("[success]* Recorder node is running. Press Ctrl+C to stop.[/]")
        node.run()
        console.print("[success]Recorder stopped.[/]")
    except KeyboardInterrupt:
        console.print(
            "\n[warning]Stopping -- waiting for encoding to complete. "
            "Please do not press Ctrl+C again.[/]"
        )
        console.print("[success]Recorder stopped.[/]")


# -- status --------------------------------------------------------------------


@click.command()
@click.option(
    "--output-dir",
    default="output/episodes",
    show_default=True,
    help="Directory containing episode files.",
)
@click.option(
    "--format",
    "storage_format",
    type=click.Choice(["hdf5", "lerobot"]),
    default="hdf5",
    show_default=True,
    help="Storage format to inspect.",
)
@handle_cli_error
def status(output_dir: str, storage_format: str) -> None:
    """Show recorded episode output status."""
    console = get_console()
    from .display import render_output_status

    out_path = Path(output_dir)
    if not out_path.is_dir():
        console.print(f"[warning]Output directory does not exist:[/] {output_dir}")
        return

    if storage_format == "hdf5":
        episode_files = sorted(out_path.glob("episode_*.h5"))
    else:
        episode_files = sorted(out_path.glob("**/*.parquet"))

    episode_count = len(episode_files)
    total_bytes = sum(f.stat().st_size for f in episode_files)
    total_size_mb = total_bytes / (1024 * 1024)

    if episode_files:
        last_mtime = max(f.stat().st_mtime for f in episode_files)
        last_modified = datetime.datetime.fromtimestamp(last_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    else:
        last_modified = "--"

    render_output_status(
        episode_count=episode_count,
        total_size_mb=total_size_mb,
        last_modified=last_modified,
        output_dir=str(out_path.resolve()),
        storage_format=storage_format,
        console=console,
    )


# -- config group --------------------------------------------------------------


@click.group(name="config")
def config_group() -> None:
    """Manage named recorder configuration files."""


@config_group.command(name="show")
@click.argument("config_name")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@handle_cli_error
def config_show(config_name: str, config_dir: Path | None) -> None:
    """Pretty-print a resolved recorder configuration file."""
    console = get_console()
    try:
        from dexim.recorder.config import load_config
    except ImportError as exc:
        console.print(f"[error]Missing dependency: {exc}[/]")
        console.print(_IMPORT_HINT)
        raise SystemExit(1) from exc

    cfg = load_config(str(resolve_config_path(config_name, _path_value(config_dir))))
    from .display import render_config

    render_config(cfg, console)


@config_group.command(name="validate")
@click.argument("config_name")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@handle_cli_error
def config_validate(config_name: str, config_dir: Path | None) -> None:
    """Validate a recorder configuration file without starting the node."""
    console = get_console()
    try:
        from dexim.recorder.config import load_config
    except ImportError as exc:
        console.print(f"[error]Missing dependency: {exc}[/]")
        console.print(_IMPORT_HINT)
        raise SystemExit(1) from exc

    resolved = resolve_config_path(config_name, _path_value(config_dir))
    load_config(str(resolved))
    console.print(f"[success][OK] Config is valid:[/] {resolved}")


@config_group.command(name="list")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
)
@handle_cli_error
def config_list(config_dir: Path | None) -> None:
    """List all named recorder configuration files."""
    console = get_console()
    dir_value = _path_value(config_dir)
    names = list_named_configs(dir_value)
    if not names:
        console.print("[muted]No configs found.[/]")
        return
    rows = [[name, str(get_config_yaml_path(name, dir_value))] for name in names]
    console.print(
        make_table(
            title="Named Recorder Configs",
            columns=[("Name", "key"), ("Path", "value")],
            rows=rows,
        )
    )


def _config_dir_option(func):
    return click.option(
        "--config-dir",
        type=click.Path(path_type=Path, file_okay=False),
        default=None,
        help="Config root directory (default: ./config or DEXIM_CONFIG_DIR).",
    )(func)


def _config_field_options(func):
    """Shared CLI options for config new / config edit."""
    func = click.option(
        "--format",
        "storage_format",
        type=click.Choice(["hdf5", "lerobot"]),
        default=None,
        help="Storage backend format.",
    )(func)
    func = click.option(
        "--output-dir",
        default=None,
        help="Directory for episode files.",
    )(func)
    func = click.option(
        "--endpoint",
        "-e",
        "endpoints",
        multiple=True,
        help="ZMQ data endpoint to subscribe to (repeatable).",
    )(func)
    func = click.option(
        "--default-task",
        default=None,
        help="Default task name embedded in episode metadata.",
    )(func)
    func = click.option(
        "--fps",
        type=int,
        default=None,
        help="Frame rate for LeRobot dataset.",
    )(func)
    func = click.option(
        "--node-id",
        default=None,
        help="Unique node identifier.",
    )(func)
    return func


@config_group.command(name="new")
@click.argument("config_name")
@_config_field_options
@_config_dir_option
@handle_cli_error
def config_new(
    config_name: str,
    config_dir: Path | None,
    storage_format: str | None,
    output_dir: str | None,
    endpoints: tuple[str, ...],
    default_task: str | None,
    fps: int | None,
    node_id: str | None,
) -> None:
    """Create a new named recorder configuration file."""
    console = get_console()
    if not endpoints:
        raise click.UsageError("Provide at least one --endpoint/-e for the new config.")
    data: dict = {"data_endpoints": list(endpoints)}
    if storage_format is not None:
        data["storage_format"] = storage_format
    if output_dir is not None:
        data["output_dir"] = output_dir
    if default_task is not None:
        data["default_task"] = default_task
    if fps is not None:
        data["fps"] = fps
    if node_id is not None:
        data["node_id"] = node_id
    path = create_config_yaml(config_name, data, _path_value(config_dir))
    console.print(f"[success]\u2713 Created config '{config_name}':[/] {path}")


@config_group.command(name="edit")
@click.argument("config_name")
@_config_field_options
@_config_dir_option
@handle_cli_error
def config_edit(
    config_name: str,
    config_dir: Path | None,
    storage_format: str | None,
    output_dir: str | None,
    endpoints: tuple[str, ...],
    default_task: str | None,
    fps: int | None,
    node_id: str | None,
) -> None:
    """Update fields in an existing named recorder configuration file."""
    console = get_console()
    updates: dict = {}
    if endpoints:
        updates["data_endpoints"] = list(endpoints)
    if storage_format is not None:
        updates["storage_format"] = storage_format
    if output_dir is not None:
        updates["output_dir"] = output_dir
    if default_task is not None:
        updates["default_task"] = default_task
    if fps is not None:
        updates["fps"] = fps
    if node_id is not None:
        updates["node_id"] = node_id
    if not updates:
        raise click.UsageError("Specify at least one field to change.")
    path = edit_config_yaml(config_name, updates, _path_value(config_dir))
    console.print(f"[success]\u2713 Updated config '{config_name}':[/] {path}")


@config_group.command(name="remove")
@click.argument("config_name")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Confirm deletion without prompting.",
)
@_config_dir_option
@handle_cli_error
def config_remove(
    config_name: str,
    yes: bool,
    config_dir: Path | None,
) -> None:
    """Delete a named recorder configuration file."""
    console = get_console()
    if not yes:
        raise click.UsageError(
            f"This will permanently delete config '{config_name}'. "
            f"Pass --yes to confirm."
        )
    path = remove_config_yaml(config_name, _path_value(config_dir))
    console.print(f"[success]\u2713 Removed config '{config_name}':[/] {path}")
