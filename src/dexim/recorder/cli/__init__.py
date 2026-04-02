"""dexim.recorder.cli — Data recorder CLI command group.

Exposes:
    recorder_group: Click group usable standalone or mounted by the umbrella.
    standalone_app: Entry point registered in ``[project.scripts]``.
"""

__version__ = "0.1.0"

import rich_click as click
from dexim.cli.common import print_banner, setup_error_handling

from .commands import config_group, run, status

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.STYLE_COMMANDS_TABLE_COLUMN_WIDTH_RATIO = (1, 3)


@click.group(name="recorder")
def recorder_group() -> None:
    """Data recorder — run, status, config."""


recorder_group.add_command(run)
recorder_group.add_command(status)
recorder_group.add_command(config_group, name="config")


def standalone_app() -> None:
    """Entry point for the standalone ``dexim-recorder`` CLI."""
    setup_error_handling()
    print_banner(app_name="DexImitate · Recorder", version=__version__)
    recorder_group(standalone_mode=True)
