"""The result of checking one batch."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .history import HistoryEntry
from .term import paint

OK, ALERT, WARMING_UP = "ok", "alert", "warming_up"
UNNAMED_BATCH = "(unnamed)"
MAX_FINDINGS_SHOWN = 5
EXPECTED = "(model expects)"  # key of the model's expectation in an example row


@dataclass
class Alert:
    check: str
    column: str
    value: float
    usual: float
    ratio: float  # the change, as a multiple of the largest change seen on a clean batch
    feature: str = ""  # the model feature behind the check; differs from column for derived ones
    detail: str = ""  # category share checks: the value whose share moved


@dataclass
class Finding:
    """One thing that went wrong in a batch, in a sentence. See morello.findings."""

    column: str
    summary: str
    ratio: float  # of the check the sentence is taken from
    checks: list[str] = field(default_factory=list)  # every check that agrees, strongest first
    where: str = ""  # the slice of the batch the affected rows are concentrated in, if any
    affects: list[str] = field(default_factory=list)  # columns that stopped fitting as a result
    check: str = ""
    feature: str = ""


@dataclass
class Report:
    batch: str | None
    status: str  # "ok", "alert" or "warming_up"
    rows_checked: int
    history_batches: int
    segment: str = ""
    findings: list[Finding] = field(default_factory=list)  # one per problem, largest first
    alerts: list[Alert] = field(default_factory=list)  # every check behind them, largest first
    suspect_columns: list[str] = field(default_factory=list)
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # column -> rows
    skipped_columns: dict[str, str] = field(default_factory=dict)  # column -> reason
    new_columns: list[str] = field(default_factory=list)  # in the batch, unknown to the model
    missing_columns: list[str] = field(default_factory=list)  # in the model, not in the batch
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
            "findings": [asdict(finding) for finding in self.findings],
            "alerts": [asdict(alert) for alert in self.alerts],
            "new_columns": self.new_columns,
            "missing_columns": self.missing_columns,
            "suspect_columns": self.suspect_columns,
            "examples": self.examples,
            "history_batches": self.history_batches,
        }

    def text(self, color: bool = False, details: bool = False) -> str:
        """The report as plain text.

        color    add terminal colour codes (see morello.term)
        details  also list every check behind the findings
        """
        batch = self.batch if self.batch is not None else UNNAMED_BATCH
        segment = f" [{self.segment}]" if self.segment else ""
        status = paint(self.status.upper().replace("_", " "), self.status, color)
        lines = [
            f"batch {batch}{segment}: {status} "
            f"({self.rows_checked} rows checked, {self.history_batches} batches of history)"
        ]
        if self.missing_columns:
            lines.append("  columns missing from this batch: " + ", ".join(self.missing_columns))
        if self.new_columns:
            lines.append(
                "  new columns the model has not learned (refit to include them): "
                + ", ".join(self.new_columns)
            )
        if self.status == WARMING_UP:
            lines.append("  not enough history yet to alert reliably")
        for number, finding in enumerate(self.findings[:MAX_FINDINGS_SHOWN], start=1):
            lines.append(f"  {number}. {paint(finding.summary, 'bold', color)}")
            agree = len(finding.checks)
            checks = f"{agree} checks agree" if agree > 1 else f"check: {finding.checks[0]}"
            ratio = f"{finding.ratio:.1f}x the largest change seen before"
            lines.append(f"     {ratio}; {checks}")
            if finding.where:
                lines.append(f"     {finding.where}")
            if finding.affects:
                lines.append(
                    f"     as a result these stopped fitting: {', '.join(finding.affects)}"
                )
            for row in self.examples.get(finding.column, []):
                lines.append(paint("     " + _row_text(row), "dim", color))
        hidden = len(self.findings) - MAX_FINDINGS_SHOWN
        if hidden > 0:
            lines.append(f"  and {hidden} more; see the JSON report")
        if details:
            for alert in self.alerts:
                lines.append(
                    f"  - {alert.check} | {alert.column}: {alert.value:.4g} "
                    f"(usual {alert.usual:.4g}), {alert.ratio:.1f}x"
                )
        return "\n".join(lines)


def _row_text(row: dict[str, Any]) -> str:
    """An example row on one line. The model's expectation follows the value it is about."""
    parts: list[str] = []
    for name, value in row.items():
        if name == EXPECTED and parts:
            about = "about " if isinstance(value, float) else ""
            parts[-1] += f" (model expects {about}{value})"
        else:
            parts.append(f"{name}={value}")
    return ", ".join(parts)
