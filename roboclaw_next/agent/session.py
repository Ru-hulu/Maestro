"""Session state for one continuous Agent interaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from roboclaw_next.agent.message import AgentMessage

if TYPE_CHECKING:
    from roboclaw_next.agent.conversation_log import ConversationLog


@dataclass
class AgentSession:
    """保存一次连续交互中的消息、摘要和唯一标识。"""

    session_id: str = field(default_factory=lambda: uuid4().hex)
    messages: list[AgentMessage] = field(default_factory=list)
    summary: str | None = None
    summary_cursor: int = 0
    conversation_log: "ConversationLog | None" = None

    def __post_init__(self) -> None:
        if self.conversation_log is not None:
            for sequence, message in enumerate(self.messages):
                self.conversation_log.append(
                    session_id=self.session_id,
                    sequence=sequence,
                    message=message,
                )

    def append(self, message: AgentMessage) -> None:
        """将一条新消息追加到当前会话历史。"""

        self.messages.append(message)
        if self.conversation_log is not None:
            self.conversation_log.append(
                session_id=self.session_id,
                sequence=len(self.messages) - 1,
                message=message,
            )
