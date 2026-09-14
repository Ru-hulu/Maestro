"""Append-only JSONL logging for every message and lifecycle event in an agent session.

Message records use the provider-compatible shape (``role``/``content``/...).
Lifecycle records (for example a context-compression summary) carry an ``event``
field instead of ``role``, so a reader can tell them apart with one key.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roboclaw_next.agent.message import AgentMessage


class ConversationLog:
    """Persist session messages and events in provider-compatible form."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Start a fresh log file for a new client process."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def append(
        self,
        *,
        session_id: str,
        sequence: int,
        message: AgentMessage,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "sequence": sequence,
            **message.to_provider_dict(),
        }
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")

    def append_event(
        self,
        *,
        session_id: str,
        event: str,
        **fields: Any,
    ) -> None:
        """Append a lifecycle event that is not a model message.

        Uses an ``event`` key instead of ``role`` so consumers can filter the
        stream with ``if "event" in record``. Extra keyword arguments become
        fields on the record (for example the summary cursor and its size).
        """

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "event": event,
            **fields,
        }
        encoded = json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), default=str
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
