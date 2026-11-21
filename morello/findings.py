"""Group the alerts of one batch into findings: one sentence per column that went wrong.

A single fault usually trips several checks at once. A join that attaches the wrong record
raises the surprise of a column, its count of extreme values and its count of values that
contradict their row. The report states it once, in the words of the most specific check,
and lists the other checks that agree.
"""

from __future__ import annotations

from collections.abc import Callable

from .checks import WHOLE_ROW
from .report import Alert, Finding

# Families of checks, most specific first. The first family present supplies the sentence.
NULLS, SHIFT, UNEXPECTED, RELATIONSHIP, DIGITS, SURPRISE, DUPLICATES = range(7)
SHIFT_CHECKS = ("mean", "median", "spread", "distribution shift", "zero rate")
MODEL_FAMILIES = (RELATIONSHIP, SURPRISE)
CAUSE_FAMILIES = (NULLS, SHIFT, UNEXPECTED)  # a column that changed on its own


def family(check: str) -> int:
    if check == "null rate":
        return NULLS
    if check in SHIFT_CHECKS or check.startswith("category share"):
        return SHIFT
    if check == "unexpected values":
        return UNEXPECTED
    if check.startswith(("broken relationship", "rows contradicting")):
        return RELATIONSHIP
    if check == "digit pattern":
        return DIGITS
    if check == "duplicate rows":
        return DUPLICATES
    return SURPRISE


def _vagueness(check: str) -> int:
    """Within a family, a check that names a value or a number makes the better sentence."""
    if check.startswith("category share") or check == "zero rate":
        return 0
    return 2 if check == "distribution shift" else 1


def summarize(alerts: list[Alert], related: Callable[[str], list[str]]) -> list[Finding]:
    """One finding per column, largest change first.

    alerts   sorted by ratio, largest first
    related  feature name -> the columns the model predicts it from
    """
    by_column: dict[str, list[Alert]] = {}
    for alert in alerts:
        by_column.setdefault(alert.column, []).append(alert)
    whole_row = by_column.pop(WHOLE_ROW, [])
    findings = [_finding(group, related) for group in by_column.values()]

    duplicates = [alert for alert in whole_row if family(alert.check) == DUPLICATES]
    contradictions = [alert for alert in whole_row if family(alert.check) != DUPLICATES]
    if duplicates:
        findings.append(_finding(duplicates, related))
    if contradictions:
        # Rows that contradict themselves are the same problem as a column that stopped
        # fitting its row, seen from the row's side.
        host = next(
            (f for f in findings if any(family(check) in MODEL_FAMILIES for check in f.checks)),
            None,
        )
        if host is None:
            findings.append(_finding(contradictions, related))
        else:
            host.checks = _distinct(host.checks + [alert.check for alert in contradictions])
    return sorted(_fold_consequences(findings, related), key=lambda finding: -finding.ratio)


def _fold_consequences(
    findings: list[Finding], related: Callable[[str], list[str]]
) -> list[Finding]:
    """Fold a column that only stopped fitting its row into the column that caused it.

    When region turns null, currency stops fitting its row because the model predicts currency
    from region. That is one problem, and region is the one to look at.
    """
    causes = {f.column: f for f in findings if family(f.check) in CAUSE_FAMILIES}
    kept = []
    for finding in findings:
        cause = None
        if family(finding.check) in MODEL_FAMILIES and finding.feature:
            cause = next((causes[c] for c in related(finding.feature) if c in causes), None)
        if cause is None:
            kept.append(finding)
        else:
            cause.affects.append(finding.column)
            cause.ratio = max(cause.ratio, finding.ratio)
    return kept


def _finding(alerts: list[Alert], related: Callable[[str], list[str]]) -> Finding:
    headline = min(alerts, key=lambda a: (family(a.check), _vagueness(a.check), -a.ratio))
    return Finding(
        column=headline.column,
        feature=headline.feature,
        check=headline.check,
        summary=_sentence(headline, related(headline.feature) if headline.feature else []),
        ratio=headline.ratio,
        checks=_distinct([alert.check for alert in alerts]),
    )


def _sentence(alert: Alert, related: list[str]) -> str:
    name = display_name(alert.feature, alert.column)
    kind = family(alert.check)

    def of(what: str) -> str:
        return f"{_share(alert.value)} of {what} (usual {_share(alert.usual)})"

    if alert.column == WHOLE_ROW:
        if kind == DUPLICATES:
            return of("rows are exact duplicates")
        return of("rows have columns that contradict each other")
    if kind == NULLS:
        return f"{name}: {of('values are null')}"
    if alert.feature.endswith("__known_format"):
        return f"{name}: some values have a format that was rare or absent in training"
    if kind == UNEXPECTED:
        return f"{name}: {of('values were never seen in training')}"
    if kind == SHIFT:
        if alert.check.startswith("category share"):
            return f"{name}: {of(f'rows have the value {alert.detail}')}"
        if alert.check == "zero rate":
            return f"{name}: {of('values are zero')}"
        if alert.check == "distribution shift":
            return f"{name}: the distribution of values changed"
        return f"{name}: values shifted, {alert.check} {alert.value:.4g} (usual {alert.usual:.4g})"
    if kind == RELATIONSHIP:
        others = " and ".join(related) if related else "the rest of its row"
        return f"{name} stopped fitting {others} in {of('rows')}"
    if kind == DIGITS:
        return f"{name}: the last digits of values follow a new pattern (rounding or a unit change)"
    if alert.check == "extreme values":
        return f"{name}: {of('values are far from what their row predicts')}"
    return f"{name}: values are harder to predict from the rest of the row than usual"


def display_name(feature: str, column: str) -> str:
    """A feature in the words of its source column."""
    if feature.endswith("__hour"):
        return f"{column} (hour of day)"
    if "__minus__" in feature:
        return f"{column} (time since {feature.split('__minus__', 1)[1]})"
    if feature.endswith("__length"):
        return f"{column} (length)"
    return column


def _share(value: float) -> str:
    """A share of rows as a short percentage: 0%, 0.47%, 3.6%, 42%."""
    if value <= 0:
        return "0%"
    if value >= 0.1:
        return f"{value:.0%}"
    return f"{value:.1%}" if value >= 0.01 else f"{value:.2%}"


def _distinct(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
