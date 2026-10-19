#!/usr/bin/env python3
"""Send the final answer of a Claude Code turn to Slack (Stop hook).

Claude Code passes the hook event as JSON on stdin. The answer is posted with
the Claude bot token (SLACK_BOT_CLAUDE_TOKEN) so it appears under a different
bot than Codex notifications; the channel settings are shared with Codex.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from codex_slack_notify import (
    DEFAULT_ENV_FILE,
    NotificationError,
    load_env_file,
    send_to_slack,
    write_log,
)


def is_user_prompt(entry: dict[str, Any]) -> bool:
    """Return True for a typed prompt, False for tool results sent as user turns."""
    if entry.get("type") != "user":
        return False
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return True
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "text" for block in content
    )


def last_answer(transcript_path: Path) -> str:
    """Join the assistant text blocks written after the last user prompt."""
    texts: list[str] = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if is_user_prompt(entry):
            texts = []
        elif entry.get("type") == "assistant":
            content = entry.get("message", {}).get("content")
            if isinstance(content, list):
                texts += [
                    block["text"]
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
    return "\n\n".join(text.strip() for text in texts if text.strip())


def main() -> int:
    event = json.load(sys.stdin)
    load_env_file(
        Path(os.environ.get("CODEX_SLACK_ENV_FILE", str(DEFAULT_ENV_FILE))).expanduser()
    )

    answer = event.get("last_assistant_message")
    if not answer and event.get("transcript_path"):
        answer = last_answer(Path(event["transcript_path"]).expanduser())

    send_to_slack(
        {"last-assistant-message": answer or "", "cwd": event.get("cwd", "")},
        token_env="SLACK_BOT_CLAUDE_TOKEN",
        title="Claude Code",
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (NotificationError, OSError, ValueError) as exc:
        write_log(f"claude_failed error={exc}")
        print(f"Slack notification failed: {exc}", file=sys.stderr)
        # Exit 0 so a Slack failure never blocks Claude from stopping.
        sys.exit(0)
