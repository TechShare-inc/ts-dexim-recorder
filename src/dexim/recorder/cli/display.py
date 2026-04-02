"""Rich display helpers for the dexim-recorder CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexim.cli.common import make_table
from rich.console import Console
from rich.rule import Rule

if TYPE_CHECKING:
    from dexim.recorder.config import RecorderNodeConfig


def render_config(cfg: RecorderNodeConfig, console: Console) -> None:
    """Render a RecorderNodeConfig as a Rich table.

    Args:
        cfg: Resolved RecorderNodeConfig instance.
        console: Rich Console to print to.
    """
    console.print(Rule("Recorder Configuration", style="brand.dim"))

    rows = [
        ["node_id", cfg.node_id],
        ["storage_format", cfg.storage_format],
        ["output_dir", cfg.output_dir],
        ["task", cfg.task or "(none)"],
        ["fps", str(cfg.fps)],
        ["data_endpoints", ", ".join(cfg.data_endpoints)],
        ["master_clock_topic", cfg.master_clock_topic or "(auto)"],
        ["continuous_topics", ", ".join(cfg.continuous_topics) or "(none)"],
        ["control_endpoint", cfg.control_endpoint],
        ["status_endpoint", cfg.status_endpoint],
    ]

    if cfg.storage_format == "lerobot":
        rows.extend([
            ["lerobot_dataset_path", cfg.lerobot_dataset_path],
            ["lerobot_repo_id", cfg.lerobot_repo_id or "(none)"],
            ["push_to_hub", str(cfg.push_to_hub)],
        ])

    console.print(
        make_table(
            title="",
            columns=[("Key", "key"), ("Value", "value")],
            rows=rows,
        )
    )


def render_output_status(
    episode_count: int,
    total_size_mb: float,
    last_modified: str,
    output_dir: str,
    storage_format: str,
    console: Console,
) -> None:
    """Render output directory status as a Rich table.

    Args:
        episode_count: Number of episode files found.
        total_size_mb: Total size of episode files in megabytes.
        last_modified: Human-readable timestamp of the last modified episode.
        output_dir: Path to the output directory.
        storage_format: Storage format (hdf5/lerobot).
        console: Rich Console to print to.
    """
    console.print(Rule("Recorder Output Status", style="brand.dim"))

    rows = [
        ["Output directory", output_dir],
        ["Storage format", storage_format],
        ["Episodes", str(episode_count)],
        ["Total size", f"{total_size_mb:.2f} MB"],
        ["Last modified", last_modified],
    ]

    console.print(
        make_table(
            title="",
            columns=[("Property", "key"), ("Value", "value")],
            rows=rows,
        )
    )
