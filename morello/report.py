"""The result of checking one batch."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .history import HistoryEntry
from .term import paint

OK, ALERT, WARMING_UP = "ok", "alert", "warming_up"
UNNAMED_BATCH = "(unnamed)"
MAX_SUSPECTS_SHOWN = 5
MAX_ALERTS_SHOWN = 8


@dataclass
class Alert:
    check: str
    column: str
    value: float
    usual: float
    ratio: float  # the change, as a multiple of the largest change seen on a clean batch


@dataclass
class Report:
    batch: str | None
    status: str  # "ok", "alert" or "warming_up"
    rows_checked: int
    history_batches: int
    segment: str = ""
    alerts: list[Alert] = field(default_factory=list)  # largest ratio first
    suspect_columns: list[str] = field(default_factory=list)
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # column -> rows
    skipped_columns: dict[str, str] = field(default_factory=dict)  # column -> reason
    history_entry: HistoryEntry | None = field(default=None, repr=False)

    @property
    def alert(self) -> bool:
        return self.status == ALERT

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch if self.batch is not None else UNNAMED_BATCH,
            "segment": self.segment,
            "status": self.status,
            "rows_checked": self.rows_checked,
            "alerts": [asdict(alert) for alert in self.alerts],
            "suspect_columns": self.suspect_columns,
            "examples": self.examples,
            "history_batches": self.history_batches,
        }

    def text(self, color: bool = False) -> str:
        """The report as plain text. color=True adds terminal colour codes (see morello.term)."""
        batch = self.batch if self.batch is not None else UNNAMED_BATCH
        segment = f" [{self.segment}]" if self.segment else ""
        status = paint(self.status.upper().replace("_", " "), self.status, color)
        head = (
            f"batch {batch}{segment}: {status} "
            f"({self.rows_checked} rows checked, {self.history_batches} batches of history)"
        )
        if self.status == WARMING_UP:
            return head + "\n  not enough history yet to alert reliably"
        if not self.alerts:
            return head

        shown = self.suspect_columns[:MAX_SUSPECTS_SHOWN]
        lines = [head, "  suspect columns: " + ", ".join(paint(c, "bold", color) for c in shown)]
        for alert in self.alerts[:MAX_ALERTS_SHOWN]:
            ratio = paint(f"{alert.ratio:.1f}x", "bold", color)
            lines.append(
                f"  - {alert.check} | {paint(alert.column, 'bold', color)}: {alert.value:.4g} "
                f"(usual {alert.usual:.4g}), {ratio} the largest change seen before"
            )
        for column, rows in self.examples.items():
            lines.append(f"  example rows for {paint(column, 'bold', color)}:")
            for row in rows:
                values = ", ".join(f"{name}={value}" for name, value in row.items())
                lines.append(paint("    " + values, "dim", color))
        return "\n".join(lines)
