"""Build bounded model context from an Agent Session."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from roboclaw_next.agent.budget import Budget, ContextOverflow, TokenEstimator
from roboclaw_next.agent.message import AgentMessage
from roboclaw_next.agent.session import AgentSession
from roboclaw_next.llm.openai_compatible import LLMProvider


# 摘要调用的输出上限，与普通回复分开。摘要要背的历史随会话单调增长，而普通回复的长度不变；共用一个上限会让摘要在会话变长后被截断。
SUMMARY_MAX_TOKENS = 4096

# 写进提示词的软目标
SUMMARY_TARGET_CHARS = 1200


@dataclass(frozen=True)
class _ConversationTurn:
    """一轮对话的消息区间。

    区间采用 [start, end)，start 是 user，end 是下一条 user 的位置。
    """

    start: int
    end: int
    complete: bool


class ContextBuilder:
    """保留近期完整轮次，并将更早的轮次压缩为滚动摘要。"""

    def __init__(
        self,
        provider: LLMProvider,
        keep_recent_turns: int = 2,
        *,
        estimator: TokenEstimator,
        budget: Budget,
    ) -> None:
        if keep_recent_turns < 1:
            raise ValueError("keep_recent_turns must be at least 1.")
        self.provider = provider
        # keep_recent_turns 的角色是下界：保证最近这几轮永远保持原文，
        # 无论预算多紧。真正决定「要不要压」的是 budget。
        self.keep_recent_turns = keep_recent_turns
        self.estimator = estimator
        self.budget = budget

    async def build(
        self,
        session: AgentSession,
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> list[AgentMessage]:
        """构造本次模型调用使用的消息，并在需要时更新历史摘要。

        `tool_definitions` 参与预算估算 —— 工具定义每轮都发，是不可忽略的固定
        开销，不计入就会显著低估这次请求的实际体积。

        压缩之后仍然超出硬上限时抛 `ContextOverflow`。
        """

        system_end, turns = _split_turns(session.messages)

        completed_turns = [turn for turn in turns if turn.complete]
        compressible_turns = completed_turns[: -self.keep_recent_turns]
        # 这里拿到的是除了最近几轮的turn,因为最近几轮不压缩
        new_turns = [
            turn for turn in compressible_turns if turn.end > session.summary_cursor
        ] # 这里拿到的是还没有压缩的turn
        # TODO: 尤其这里的压缩单位是<user> <user>，所以如果用户下达了一次指令，这次本身的上下文很长，就会爆。
        # 但是我们不考虑这种情况
        if new_turns and self._over_budget(session, system_end, tool_definitions):
            await self._update_summary(
                session, new_turns, system_end, tool_definitions
            )

        context = self._assemble(session, system_end)

        # 超出硬上限时明确失败，而不是照发让 Provider 返回 400 —— 后者会把
        # 报错文本当成模型回答写进历史，那一轮还会被判定为完整轮次、参与摘要。
        estimated = self._estimate(context, tool_definitions)
        if estimated > self.budget.limit:
            raise ContextOverflow(
                f"context is {estimated} tokens, limit is {self.budget.limit}; "
                "reduce the size of recent tool results or use a larger model"
            )
        return context

    def _assemble(self, session: AgentSession, system_end: int) -> list[AgentMessage]:
        """按当前的 summary 与 cursor 装配出这次要发送的消息。"""

        context = list(session.messages[:system_end]) # 拿到messages数组中最前面的systems
        if session.summary:
            context.append(
                AgentMessage(
                    role="system",
                    content=f"以下是此前对话的摘要：\n{session.summary}",
                )
            )
        # 在system 后面进行追加summary的结果
        history_start = max(system_end, session.summary_cursor)
        context.extend(session.messages[history_start:]) # 这里是messages数组中被压缩信息后面的内容（近几次的对话）
        return context

    def _over_budget(
        self,
        session: AgentSession,
        system_end: int,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> bool:
        """判断是否会超出预算触发线。"""

        candidate = self._assemble(session, system_end)
        return self._estimate(candidate, tool_definitions) > self.budget.trigger

    def _estimate(
        self,
        context: list[AgentMessage],
        tool_definitions: list[dict[str, Any]] | None,
    ) -> int:
        return self.estimator.estimate(
            [message.to_provider_dict() for message in context],
            tool_definitions,
        )

    async def _update_summary(
        self,
        session: AgentSession,
        turns: list[_ConversationTurn],
        system_end: int,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> None:
        # 压缩前的估算：必须在推进 cursor 之前取，否则量到的是压缩后的上下文。
        estimated_before: int | None = None
        if self.estimator is not None and self.budget is not None:
            estimated_before = self._estimate(
                self._assemble(session, system_end), tool_definitions
            )

        messages_to_summarize: list[AgentMessage] = []
        for turn in turns:
            messages_to_summarize.extend(session.messages[turn.start : turn.end])

        previous_summary = session.summary or "（暂无历史摘要）"
        serialized_messages = json.dumps(
            [message.to_provider_dict() for message in messages_to_summarize],
            ensure_ascii=False,
        )
        response = await self.provider.chat_with_retry(
            [
                {
                    "role": "system",
                    "content": (
                        "你负责压缩 Agent 的历史对话。"
                        "请结合已有摘要和新增轮次，"
                        "只输出更新后的累计摘要。"
                        "保留用户目标、已确认决定、重要工具结果、"
                        "失败原因和未解决问题，不要添加推测。"
                        f"整段摘要控制在 {SUMMARY_TARGET_CHARS} 字以内。"
                        "空间不够时按这个顺序取舍："
                        "未解决问题和失败原因必须完整保留，"
                        "其次是用户目标和已确认决定，"
                        "已完成步骤的细节可以合并成一句。"
                        "宁可写得更概括，也不要写到一半被截断。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"已有摘要：\n{previous_summary}\n\n"
                        f"需要合并的新轮次：\n{serialized_messages}"
                    ),
                },
            ],
            tools=None,
            temperature=0,
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        if response.finish_reason == "error":
            raise RuntimeError(response.content or "Failed to summarize session history.")
        if response.finish_reason == "length":
            # 生成的摘要超出了 SUMMARY_MAX_TOKENS，说明模型没有按要求压缩。
            # 通常是随着对话增加以及`未解决问题和失败原因必须完整保留`的要求，导致压缩后摘要还是超出限制。
            raise RuntimeError(
                "Session summarization was truncated at the "
                f"{SUMMARY_MAX_TOKENS}-token output limit."
            )

        summary = (response.content or "").strip()
        if not summary:
            raise RuntimeError("Session summarization returned empty content.")

        session.summary = summary
        session.summary_cursor = turns[-1].end
        self._log_compression(session, turns, estimated_before)

    def _log_compression(
        self,
        session: AgentSession,
        turns: list[_ConversationTurn],
        estimated_before: int | None,
    ) -> None:
        """把一次压缩写进会话日志，便于事后核对压缩是否按预期发生。"""

        log = session.conversation_log
        if log is None:
            return
        fields: dict[str, Any] = {
            "summary_cursor": session.summary_cursor,
            "summarized_turns": len(turns),
            "messages_total": len(session.messages),
            "summary_chars": len(session.summary or ""),
        }
        if self.budget is not None:
            fields["trigger_tokens"] = self.budget.trigger
        if estimated_before is not None:
            fields["estimated_tokens_before"] = estimated_before
        log.append_event(
            session_id=session.session_id,
            event="context_compression",
            **fields,
        )


# 示例：索引 0 是系统提示词，1–4 是已完成的第一轮，5 是尚未完成的第二轮。
# call = ToolCall(id="call_001", name="get_status", arguments={})
# messages = [
#     AgentMessage(role="system", content="你是一个机器人助手。"),
#     AgentMessage(role="user", content="查询机器人状态。"),
#     AgentMessage(role="assistant", tool_calls=[call]),
#     AgentMessage(role="tool", tool_call_id="call_001", name="get_status", content='{"battery": 80, "state": "idle"}'),
#     AgentMessage(role="assistant", content="机器人当前空闲，剩余电量 80%。"),
#     AgentMessage(role="user", content="它现在在哪里？"),
# ]
# 划分结果：system_end=1，turns=[(start=1, end=5, complete=True), (start=5, end=6, complete=False)]。

def _split_turns(
    messages: list[AgentMessage],
) -> tuple[int, list[_ConversationTurn]]:
    system_end = 0
    while system_end < len(messages) and messages[system_end].role == "system":
        system_end += 1
    # 检查消息列表开头有多少条连续的 System Message。
    if system_end < len(messages) and messages[system_end].role != "user":
        raise ValueError("A conversation turn must start with a user message.")

    # 每一轮对话都从一条 user message 开始，
    # 因此所有 user 的索引就是 Turn.start。
    user_message_indexes = [
        index
        for index in range(system_end, len(messages))
        if messages[index].role == "user"
    ]
    # 下一条 user 是当前 Turn 的右边界（不包含），同时也是下一轮的 start。
    # 最后一轮没有下一条 user，因此使用 len(messages) 作为右边界。
    turn_end_indexes = user_message_indexes[1:] + [len(messages)]
    turns = [
        _make_turn(messages, start, end) # 两个role = user之间的部分
        for start, end in zip(user_message_indexes, turn_end_indexes)
    ]

    if any(not turn.complete for turn in turns[:-1]):
        raise ValueError("A new user turn cannot start before the previous turn completes.")
    return system_end, turns


def _make_turn(
    messages: list[AgentMessage],
    start: int,
    end: int,
) -> _ConversationTurn:
    final_message = messages[end - 1]
    complete = final_message.role == "assistant" and not final_message.tool_calls
    # 最后一条的角色是Assistant，并且没有调用工具。
    return _ConversationTurn(start=start, end=end, complete=complete)
