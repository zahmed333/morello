"""Send alerts to people: a Slack channel or an email.

Both carry the findings and the command that accepts a batch. Accepting from a button inside
Slack needs a web endpoint that Slack can call, which a command line tool does not have, so
the message gives the command to run.

Sending is best effort. A failed notification is logged and never changes the result of a
check: the exit code and the results table are what gates a pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from email.message import EmailMessage

from .report import Report

log = logging.getLogger(__name__)

MAX_FINDINGS, MAX_BATCHES = 3, 10
TIMEOUT_SECONDS = 10


def message_text(
    table: str, reports: list[Report], unresolved: list[str], accept_command: str = ""
) -> tuple[str, str]:
    """(subject, body) describing what is wrong with a table. Body is empty when nothing is."""
    alerting = [report for report in reports if report.alert]
    earlier = [batch for batch in unresolved if batch not in {r.batch for r in alerting}]
    if not alerting and not earlier:
        return "", ""
    count = len({report.batch for report in alerting}) + len(earlier)
    subject = f"Morello: {count} batch{'es' if count != 1 else ''} of {table} need a look"
    lines: list[str] = []
    for report in alerting[:MAX_BATCHES]:
        segment = f" [{report.segment}]" if report.segment else ""
        lines.append(f"Batch {report.batch}{segment} ({report.rows_checked} rows checked)")
        for finding in report.findings[:MAX_FINDINGS]:
            lines.append(f"  - {finding.summary}")
            if finding.where:
                lines.append(f"    {finding.where}")
        lines.append("")
    if earlier:
        lines.append("Still open from earlier runs: " + ", ".join(earlier))
        lines.append("")
    if accept_command:
        lines.append("If a batch is fine, accept it so the pipeline continues:")
        lines.append(f"  {accept_command}")
    return subject, "\n".join(lines).rstrip()


def slack_payload(subject: str, body: str) -> dict:
    """The message as Slack blocks, with a plain-text fallback for notifications."""
    return {
        "text": subject,
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": subject[:150]}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"```{body[:2900]}```"}},
        ],
    }


def post_slack(webhook_url: str, subject: str, body: str) -> None:
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(slack_payload(subject, body)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        response.read()


def email_message(subject: str, body: str, sender: str, recipients: list[str]) -> EmailMessage:
    message = EmailMessage()
    message["Subject"], message["From"], message["To"] = subject, sender, ", ".join(recipients)
    message.set_content(body)
    return message


def send_email(message: EmailMessage, environ: dict[str, str] | None = None) -> None:
    """Send through the SMTP server named by MORELLO_SMTP_HOST, _PORT, _USER and _PASSWORD."""
    environ = os.environ if environ is None else environ
    host = environ.get("MORELLO_SMTP_HOST")
    if not host:
        raise ValueError("set MORELLO_SMTP_HOST to send email")
    port = int(environ.get("MORELLO_SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=TIMEOUT_SECONDS) as server:
        if port != 25:
            server.starttls()
        if environ.get("MORELLO_SMTP_USER"):
            server.login(environ["MORELLO_SMTP_USER"], environ.get("MORELLO_SMTP_PASSWORD", ""))
        server.send_message(message)


def notify(
    table: str,
    reports: list[Report],
    unresolved: list[str],
    slack_webhook: str | None = None,
    email_to: list[str] | None = None,
    accept_command: str = "",
) -> list[str]:
    """Tell Slack and email about alerts. Returns the channels that were reached."""
    subject, body = message_text(table, reports, unresolved, accept_command)
    if not body:
        return []
    reached = []
    if slack_webhook:
        try:
            post_slack(slack_webhook, subject, body)
            reached.append("slack")
        except OSError as error:
            log.warning("could not post to Slack: %s", error)
    if email_to:
        try:
            sender = os.environ.get("MORELLO_EMAIL_FROM", "morello@localhost")
            send_email(email_message(subject, body, sender, email_to))
            reached.append("email")
        except (OSError, ValueError, smtplib.SMTPException) as error:
            log.warning("could not send email: %s", error)
    return reached
