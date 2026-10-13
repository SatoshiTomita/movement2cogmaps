#!/usr/bin/env python3
"""Send the final message from a completed Codex CLI turn to a Slack DM.

Codex invokes notification programs with one JSON argument. This script reads
Slack credentials from the process environment or, by default, from
~/.config/codex/slack.env. Existing environment variables take precedence over
values in the file.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENV_FILE = Path("~/.config/codex/slack.env").expanduser()
DEFAULT_LOG_FILE = Path("~/.local/state/codex-slack-notify.log").expanduser()
SLACK_API_BASE_URL = "https://slack.com/api"
SLACK_MESSAGE_CHUNK_SIZE = 3500


class NotificationError(RuntimeError):
    """Raised when the notification cannot be delivered."""


def write_log(message: str) -> None:
    """Write metadata-only diagnostics without storing prompts or credentials."""
    try:
        DEFAULT_LOG_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with DEFAULT_LOG_FILE.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamp} {message}\n")
        DEFAULT_LOG_FILE.chmod(0o600)
    except OSError:
        # Notification delivery should not fail only because diagnostics cannot
        # be written.
        pass


def _parse_env_value(value: str, *, line_number: int, path: Path) -> str:
    """Parse an env-file value with optional shell-style quotes."""
    if not value:
        return ""

    try:
        parts = shlex.split(value, comments=True, posix=True)
    except ValueError as exc:
        raise NotificationError(
            f"Invalid value in {path} on line {line_number}: {exc}"
        ) from exc

    if not parts:
        return ""
    if len(parts) != 1:
        raise NotificationError(
            f"Invalid value in {path} on line {line_number}: quote values containing spaces"
        )
    return parts[0]


def load_env_file(path: Path) -> None:
    """Load the supported Slack variables without overriding the environment."""
    if not path.exists():
        return
    if not path.is_file():
        raise NotificationError(f"Environment file is not a regular file: {path}")

    supported_keys = {"SLACK_BOT_TOKEN", "SLACK_USER_ID"}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise NotificationError(
                f"Invalid line in {path} on line {line_number}: expected KEY=VALUE"
            )

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in supported_keys:
            continue
        value = _parse_env_value(raw_value.strip(), line_number=line_number, path=path)
        os.environ.setdefault(key, value)


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise NotificationError(
            f"{name} is not set. Add it to {DEFAULT_ENV_FILE} or export it before starting Codex."
        )
    return value


def slack_api(token: str, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{SLACK_API_BASE_URL}/{method}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise NotificationError(
            f"Slack API HTTP error for {method}: {exc.code} {response_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise NotificationError(f"Could not reach Slack API for {method}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise NotificationError(f"Slack returned invalid JSON for {method}") from exc

    if not result.get("ok"):
        error = result.get("error", "unknown_error")
        raise NotificationError(f"Slack API error for {method}: {error}")
    return result


def split_message(message: str) -> list[str]:
    """Split long answers into comfortably sized Slack messages."""
    return [
        message[index : index + SLACK_MESSAGE_CHUNK_SIZE]
        for index in range(0, len(message), SLACK_MESSAGE_CHUNK_SIZE)
    ]


def clean_assistant_message(message: str) -> str:
    """Extract the answer when a TUI transcript is included in the payload."""
    text = message.strip()
    if not text:
        return ""

    lines = text.splitlines()

    # Some clients append a generated chat title after the visible transcript.
    # Remove it only when the complete final non-empty line is a title object.
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        try:
            trailing_value = json.loads(lines[-1].strip())
        except json.JSONDecodeError:
            trailing_value = None
        if (
            isinstance(trailing_value, dict)
            and set(trailing_value) == {"title"}
            and isinstance(trailing_value["title"], str)
        ):
            lines.pop()

    user_marker = re.compile(r"^\s*›(?:\s+|$)")
    assistant_marker = re.compile(r"^\s*•(?:\s+|$)")
    assistant_lines = [
        index for index, line in enumerate(lines) if assistant_marker.match(line)
    ]

    # Interpret the bullet as TUI chrome only when a matching user prompt marker
    # is present. This preserves ordinary answers that legitimately start with •.
    if assistant_lines and any(user_marker.match(line) for line in lines):
        answer_start = assistant_lines[-1]
        first_line = assistant_marker.sub("", lines[answer_start], count=1)
        lines = [first_line, *lines[answer_start + 1 :]]

    return "\n".join(lines).strip()


def send_dm(event: dict[str, Any]) -> int:
    token = require_env("SLACK_BOT_TOKEN")
    user_id = require_env("SLACK_USER_ID")

    raw_answer = event.get("last-assistant-message")
    if raw_answer is None:
        raw_answer = event.get("last_assistant_message", "")
    answer = clean_assistant_message(str(raw_answer))
    write_log(
        f"event_received type=agent-turn-complete "
        f"raw_chars={len(str(raw_answer))} cleaned_chars={len(answer)}"
    )
    if not answer:
        write_log("skipped reason=empty_answer")
        return 0

    conversation = slack_api(token, "conversations.open", {"users": user_id})
    channel_id = conversation.get("channel", {}).get("id")
    if not channel_id:
        raise NotificationError("Slack did not return a DM channel ID")

    cwd = str(event.get("cwd", "")).strip()
    project_name = Path(cwd).name if cwd else "unknown project"

    for index, chunk in enumerate(split_message(answer)):
        if index == 0:
            text = f"*Codex CLI completed — {project_name}*\n\n{chunk}"
        else:
            text = f"*Codex CLI response continued*\n\n{chunk}"
        slack_api(token, "chat.postMessage", {"channel": channel_id, "text": text})
    chunk_count = len(split_message(answer))
    write_log(f"sent chunks={chunk_count} project={project_name!r}")
    return chunk_count


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise NotificationError(
            "Expected one Codex notification JSON argument. "
            "For example: codex_slack_notify.py '{\"type\":\"agent-turn-complete\",...}'"
        )

    env_file = Path(
        os.environ.get("CODEX_SLACK_ENV_FILE", str(DEFAULT_ENV_FILE))
    ).expanduser()
    load_env_file(env_file)

    try:
        event = json.loads(argv[1])
    except json.JSONDecodeError as exc:
        raise NotificationError(f"Invalid notification JSON: {exc}") from exc
    if not isinstance(event, dict):
        raise NotificationError("Notification JSON must be an object")

    if event.get("type") != "agent-turn-complete":
        return 0

    send_dm(event)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except NotificationError as exc:
        write_log(f"failed error={exc}")
        print(f"Slack notification failed: {exc}", file=sys.stderr)
        sys.exit(1)
