"""morello: flag batches of a table whose rows stop making sense together."""
from .monitor import Monitor, Report, merge_stats
from .store import LocalStore, FsspecStore, TableStore

__all__ = ["Monitor", "Report", "merge_stats", "LocalStore", "FsspecStore", "TableStore"]
__version__ = "0.3.0"
