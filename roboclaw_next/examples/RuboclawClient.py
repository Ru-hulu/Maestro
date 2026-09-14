"""Let an LLM choose and call tools exposed by a local MCP server.

Run with DeepSeek:

    DEEPSEEK_API_KEY=... ROBOCLAW_LLM_PROVIDER=deepseek PYTHONPATH=. \
        uv run --no-project --with openai --with "mcp[cli]<2" \
        python roboclaw_next/examples/RuboclawClient.py

Run with OpenAI:

    OPENAI_API_KEY=... ROBOCLAW_LLM_PROVIDER=openai PYTHONPATH=. \
        uv run --no-project --with openai --with "mcp[cli]<2" \
        python roboclaw_next/examples/RuboclawClient.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import cast

from roboclaw_next.agent import (
    AgentMessage,
    AgentRuntime,
    AgentSession,
    ContextBuilder,
    ConversationLog,
)
from roboclaw_next.agent.budget import Budget, TokenEstimator
from roboclaw_next.llm import create_llm_provider
from roboclaw_next.llm.types import ProviderName
from roboclaw_next.tools import (
    MCPClientRuntime,
    StdioMCPServerConfig,
    ToolRegistry,
    load_mcp_tools,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# 单次回复的最大输出长度。它同时是输入预算要扣掉的预留量 —— 窗口是输入和
# 输出共享的，只按输入算会在生成阶段撞上限。
MAX_OUTPUT_TOKENS = 1024


def load_dotenv(path: Path) -> None:
    """把仓库根目录 .env 中的键值读进 os.environ。

    只支持 `KEY=VALUE` 和 `#` 注释这两种最基本的形式，不做变量展开、多行值和
    转义处理；需要更完整的语义时再换成 python-dotenv。

    已经存在于环境中的变量优先，因此 shell 里显式设置的值不会被 .env 覆盖。
    """

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_provider_name() -> ProviderName:
    raw_provider = os.environ.get("ROBOCLAW_LLM_PROVIDER", "deepseek").lower()
    if raw_provider not in ("openai", "deepseek"):
        raise ValueError("ROBOCLAW_LLM_PROVIDER must be 'openai' or 'deepseek'.")
    return cast(ProviderName, raw_provider)


def should_autostart_perception() -> bool:
    return os.environ.get("ROBOCLAW_AUTOSTART_PERCEPTION", "0") == "1"


async def start_required_service(
    runtime: MCPClientRuntime,
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    print(f"[startup] {tool_name} ...", flush=True)
    result = await runtime.call_tool(tool_name, arguments)
    status = getattr(result, "structuredContent", None)
    payload = status.get("result", status) if isinstance(status, dict) else None
    if (
        bool(getattr(result, "isError", False))
        or not isinstance(payload, dict)
        or payload.get("state") != "running"
    ):
        raise RuntimeError(f"{tool_name} failed: {status or result}")
    print(f"[startup] {payload.get('message', tool_name + ' is running.')}")


async def autostart_perception(runtime: MCPClientRuntime) -> None:
    await start_required_service(
        runtime,
        "start_gazebo_realsense_camera",
        {"frame_timeout_sec": 30.0},
    )
    await start_required_service(runtime, "start_sam3_perception", {})


async def main() -> None:
    # 在读取任何配置之前载入 .env；MCP server 子进程通过 os.environ.copy()
    # 继承这些变量，因此工具侧也能看到。
    load_dotenv(REPOSITORY_ROOT / ".env")
    conversation_log_path = Path(
        os.environ.get(
            "ROBOCLAW_CONVERSATION_LOG",
            str(REPOSITORY_ROOT / "runtime_data" / "conversations" / "latest.jsonl"),
        )
    )
    conversation_log = ConversationLog(conversation_log_path)
    conversation_log.reset()
    print(f"[agent] conversation log: {conversation_log_path.resolve()}")

    config = StdioMCPServerConfig(
        name="roboclaw_tools",
        command=sys.executable,
        args=["-m", "roboclaw_next.tools.mcp_server"],
        env=os.environ.copy(),
    )

    print(f"[mcp] connect server: {config.name}")
    print(f"[mcp] command: {config.command} {config.args}")
    async with MCPClientRuntime(config) as runtime:
        registry = ToolRegistry(await load_mcp_tools(runtime))

        print("[mcp] registered tools:")
        for name in registry.names:
            print(f"- {name}")

        if should_autostart_perception():
            await autostart_perception(runtime)

        provider = create_llm_provider(
            resolve_provider_name(),
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        session = AgentSession(
            messages=[
                AgentMessage(
                    role="system",
                    content=(
                        "You are a tool-using assistant. Use the provided tools "
                        "when appropriate, and answer the user in Chinese. "
                        "When the requested operation has succeeded, give a final "
                        "answer instead of repeating status checks."
                    ),
                ),
            ],
            conversation_log=conversation_log,
        )
        # estimator 由 ContextBuilder 与 AgentRuntime 共用，这样校正系数在一处
        # 累积：Runtime 每轮用真实用量校准它，ContextBuilder 用它判断预算。
        estimator = TokenEstimator()
        budget = Budget(
            context_window=int(os.environ.get("ROBOCLAW_CONTEXT_WINDOW", "65536")),
            reserve_output=MAX_OUTPUT_TOKENS,
        )
        context_builder = ContextBuilder(
            provider,
            keep_recent_turns=2,
            estimator=estimator,
            budget=budget,
        )
        agent_runtime = AgentRuntime(provider, registry, context_builder, estimator=estimator)

        print("\nEnter /exit to quit.")
        while True:
            try:
                user_input = (await asyncio.to_thread(input, "\nYou: ")).strip()
            except EOFError:
                break
            if user_input == "/exit":
                break
            if not user_input:
                continue

            session.append(AgentMessage(role="user", content=user_input))
            answer = await agent_runtime.run(session, trace=True)
            if answer is None:
                print("\nAssistant:")
                print(
                    "工具调用已经达到本轮上限，但客户端会继续运行。"
                    "你可以继续输入下一条指令，或查询当前状态。"
                )
                continue

            print("\nAssistant:")
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
