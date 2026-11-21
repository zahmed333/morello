"""A status page for one table: a single HTML file built from the results table.

It shows what is open now, a strip with the status of every recent batch, and the findings
behind each alert. The file has no external requests, so it can be opened from disk, attached
to a ticket or served as a static page.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .monitor import Monitor
from .sql import connect, identifier, literal
from .store import Store, open_store
from .table import ACCEPTED, TIMESTAMP_FORMAT, _key

STRIP_BATCHES = 90
LABELS = {
    "ok": "ok",
    "alert": "alert",
    ACCEPTED: "accepted",
    "warming_up": "warming up",
}
GLYPHS = {"alert": "!", ACCEPTED: "✓"}  # status is never shown by colour alone


def write_page(
    source: Any,
    store: str | Store,
    results_table: str,
    out: str,
    key: str | None = None,
    table: str | None = None,
    dialect: str | None = None,
) -> None:
    source, store = connect(source, dialect), open_store(store)
    monitor = Monitor.load(store, _key(key, table))
    table = table or monitor.table_config.table
    results = source.query(
        f"SELECT * FROM {identifier(results_table)} WHERE table_name = {literal(str(table))}"
    )
    results.columns = [name.lower() for name in results.columns]
    with open(out, "w", encoding="utf-8") as file:
        file.write(render_page(str(table), results, monitor))


def latest_per_check(results: pd.DataFrame) -> pd.DataFrame:
    """The newest row for each (batch, segment), oldest batch first."""
    ordered = results.sort_values("checked_at", kind="stable")
    latest = ordered.drop_duplicates(["batch", "segment"], keep="last")
    return latest.sort_values(["batch", "segment"], kind="stable").reset_index(drop=True)


def render_page(table: str, results: pd.DataFrame, monitor: Monitor) -> str:
    latest = latest_per_check(results)
    whole = latest[latest["segment"] == ""].tail(STRIP_BATCHES)
    problems = latest[latest["status"].isin(["alert", ACCEPTED])].iloc[::-1]
    open_batches = list(monitor.unresolved_alerts)
    generated = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
    floor = monitor.smallest_detectable_share

    tiles = [
        ("Open alerts", str(len(open_batches)), ", ".join(open_batches[:4]) or "none"),
        ("Batches checked", str(latest["batch"].nunique()), f"last: {monitor.last_checked}"),
        (
            "Accepted",
            str(latest.loc[latest["status"] == ACCEPTED, "batch"].nunique()),
            "alerts a person reviewed and accepted",
        ),
        ("Detection floor", f"{floor:.2%}", f"of {monitor.sample_rows} sampled rows per batch"),
    ]
    cells = "".join(
        f'<li class="cell {_e(row.status)}" tabindex="0" data-tip="{_e(_tip(row))}">'
        f'<span aria-hidden="true">{GLYPHS.get(row.status, "")}</span>'
        f'<span class="visually-hidden">{_e(_tip(row))}</span></li>'
        for row in whole.itertuples()
    )
    rows = "".join(
        f"<tr><td class='mono'>{_e(row.batch)}</td><td>{_e(row.segment) or 'whole table'}</td>"
        f"<td><span class='status {_e(row.status)}'>{GLYPHS.get(row.status, '')} "
        f"{_e(LABELS.get(row.status, row.status))}</span></td>"
        f"<td class='found'>{_findings_html(row)}</td>"
        f"<td class='mono'>{_e(row.checked_at)}</td></tr>"
        for row in problems.itertuples()
    )
    every = "".join(
        f"<tr><td class='mono'>{_e(row.batch)}</td><td>{_e(row.segment) or 'whole table'}</td>"
        f"<td>{_e(LABELS.get(row.status, row.status))}</td><td class='num'>{row.rows_checked}</td>"
        f"</tr>"
        for row in latest.iloc[::-1].itertuples()
    )
    tile_html = "".join(
        f"<div class='tile'><p class='label'>{_e(label)}</p><p class='value'>{_e(value)}</p>"
        f"<p class='note'>{_e(note)}</p></div>"
        for label, value, note in tiles
    )
    first = _e(whole["batch"].iloc[0]) if len(whole) else ""
    last = _e(whole["batch"].iloc[-1]) if len(whole) else ""
    return _TEMPLATE.format(
        table=_e(table),
        generated=generated,
        tiles=tile_html,
        cells=cells or "<li>no checks recorded yet</li>",
        first=first,
        last=last,
        rows=rows or "<tr><td colspan='5'>No batch has alerted.</td></tr>",
        every=every,
        columns=_e(", ".join(plan.column for plan in monitor.schema.plans)),
        fitted=monitor.checked_since_fit,
    )


def _findings(row: Any) -> list[str]:
    try:
        details = json.loads(row.details) if row.details else {}
    except (TypeError, ValueError):
        details = {}
    findings = [text for text in details.get("findings", []) if text]
    if not findings and row.top_check:
        findings = [f"{row.top_column}: {row.top_check}"]
    return findings


def _findings_html(row: Any) -> str:
    try:
        notes = json.loads(row.details).get("notes", []) if row.details else []
    except (TypeError, ValueError):
        notes = []
    items = []
    for index, text in enumerate(_findings(row)[:3]):
        note = notes[index] if index < len(notes) and notes[index] else ""
        items.append(_e(text) + (f"<br><span class='muted'>{_e(note)}</span>" if note else ""))
    return "<br>".join(items) or "&nbsp;"


def _tip(row: Any) -> str:
    findings = _findings(row) if row.status in ("alert", ACCEPTED) else []
    text = f"{row.batch}: {LABELS.get(row.status, row.status)}"
    return f"{text}. {findings[0]}" if findings else text


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Morello: {table}</title>
<style>
  :root {{
    --ground: #F6F8FC; --surface: #FFFFFF; --tint: #DDE6F7; --ink: #391217; --muted: #4F658B;
    --line: #B4C3DE; --alert: #9E2A33; --ok: #8DA5CA; --accepted: #4F658B; --on-mark: #FFFFFF;
    --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
    --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --ground: #24090D; --surface: #391217; --tint: #4A1C22; --ink: #EEF2FA; --muted: #B7C5E0;
      --line: #6A353C; --alert: #EE9AA2; --ok: #5F7397; --accepted: #B7C5E0; --on-mark: #24090D;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--ground); color: var(--ink); font: 16px/1.55 var(--sans); }}
  main {{ width: min(100% - 32px, 1040px); margin: 0 auto; padding: 32px 0 64px;
         display: grid; gap: 32px; }}
  main > * {{ min-width: 0; }}  /* let wide tables scroll inside their own box */
  h1 {{ font-size: 1.75rem; line-height: 1.15; margin: 0; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.125rem; margin: 0 0 12px; }}
  p {{ margin: 0; }}
  .muted, .note, .label {{ color: var(--muted); font-size: 0.875rem; }}
  .mono, .num {{ font-family: var(--mono); font-size: 0.875rem; white-space: nowrap; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .tiles {{ display: grid; gap: 12px;
           grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
  .tile {{ background: var(--surface); border-radius: 12px; padding: 16px; }}
  .tile .value {{ font-size: 2rem; font-weight: 600; line-height: 1.2;
                 font-variant-numeric: tabular-nums; }}
  .strip {{ list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 2px; }}
  .cell {{ width: 18px; height: 32px; border-radius: 4px; background: var(--ok);
          color: var(--on-mark); font: 700 12px/32px var(--sans); text-align: center; }}
  .cell.alert {{ background: var(--alert); }}
  .cell.accepted {{ background: var(--accepted); }}
  .cell.warming_up {{ background: var(--tint); }}
  .cell:hover, .cell:focus-visible {{ outline: 2px solid var(--ink); outline-offset: 1px; }}
  .axis {{ display: flex; justify-content: space-between; margin-top: 6px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 8px; }}
  .legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px;
                          border-radius: 2px; margin-right: 6px; background: var(--ok); }}
  .legend .alert::before {{ background: var(--alert); }}
  .legend .accepted::before {{ background: var(--accepted); }}
  .legend .warming_up::before {{ background: var(--tint); }}
  .scroll {{ overflow-x: auto; background: var(--surface); border-radius: 12px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 10px 14px; text-align: left; vertical-align: top;
           border-bottom: 1px solid var(--tint); }}
  th {{ font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase;
       color: var(--muted); }}
  tr:last-child td {{ border-bottom: 0; }}
  .found {{ min-width: 260px; }}
  .status {{ font-weight: 600; white-space: nowrap; }}
  .status.alert {{ color: var(--alert); }}
  details summary {{ cursor: pointer; color: var(--muted); padding: 8px 0; }}
  #tip {{ position: fixed; pointer-events: none; background: var(--ink); color: var(--ground);
         padding: 6px 10px; border-radius: 6px; font-size: 0.8125rem; max-width: 320px;
         display: none; z-index: 2; }}
  .visually-hidden {{ position: absolute; width: 1px; height: 1px; overflow: hidden;
                     clip-path: inset(50%); white-space: nowrap; }}
</style>
</head>
<body>
<main>
  <header>
    <p class="muted">Morello status page, generated {generated} UTC</p>
    <h1>{table}</h1>
  </header>
  <section class="tiles" aria-label="Summary">{tiles}</section>
  <section aria-labelledby="strip-title">
    <h2 id="strip-title">Status of each batch, whole table</h2>
    <ul class="strip">{cells}</ul>
    <div class="axis muted"><span>{first}</span><span>{last}</span></div>
    <div class="legend muted">
      <span>ok</span><span class="alert">! alert</span>
      <span class="accepted">✓ accepted by a person</span>
      <span class="warming_up">warming up</span>
    </div>
  </section>
  <section aria-labelledby="alerts-title">
    <h2 id="alerts-title">Alerts, newest first</h2>
    <div class="scroll"><table>
      <thead><tr><th>Batch</th><th>Scope</th><th>Status</th><th>What was found</th>
      <th>Checked at</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
  </section>
  <section>
    <details>
      <summary>Every check as a table</summary>
      <div class="scroll"><table>
        <thead><tr><th>Batch</th><th>Scope</th><th>Status</th><th class="num">Rows checked</th>
        </tr></thead>
        <tbody>{every}</tbody>
      </table></div>
    </details>
    <p class="muted">Columns read: {columns}. Batches checked since the model was fitted:
    {fitted}.</p>
  </section>
</main>
<div id="tip" role="status"></div>
<script>
  const tip = document.getElementById("tip");
  for (const cell of document.querySelectorAll(".cell")) {{
    const show = (x, y) => {{
      tip.textContent = cell.dataset.tip;
      tip.style.display = "block";
      tip.style.left = Math.min(x + 12, window.innerWidth - tip.offsetWidth - 8) + "px";
      tip.style.top = (y + 16) + "px";
    }};
    cell.addEventListener("mousemove", (event) => show(event.clientX, event.clientY));
    cell.addEventListener("focus", () => {{
      const box = cell.getBoundingClientRect();
      show(box.left, box.bottom - 12);
    }});
    for (const name of ["mouseleave", "blur"]) {{
      cell.addEventListener(name, () => {{ tip.style.display = "none"; }});
    }}
  }}
</script>
</body>
</html>
"""
