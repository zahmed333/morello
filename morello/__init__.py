"""Morello: flag batches of a table whose rows stop making sense together."""

from .checks import merge_statistics
from .monitor import Monitor, TableConfig
from .report import Alert, Finding, Report
from .state import StateError
from .store import FsspecStore, LocalStore, TableStore

__all__ = [
    "Alert",
    "Finding",
    "FsspecStore",
    "LocalStore",
    "Monitor",
    "Report",
    "StateError",
    "TableConfig",
    "TableStore",
    "merge_statistics",
]
__version__ = "0.7.0"
