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

from roboclaw_next.agent import AgentMessage, AgentRuntime, AgentSession, ContextBuilder
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


async def main() -> None:
    # 在读取任何配置之前载入 .env；MCP server 子进程通过 os.environ.copy()
    # 继承这些变量，因此工具侧也能看到。
    load_dotenv(REPOSITORY_ROOT / ".env")

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
                        "After the requested operation has succeeded, give a final "
                        "answer instead of repeating status checks. "
                        "Arm motion: call plan_openarm_pose, then execute_openarm_plan. "
                        "It takes relative EE motion and resolves the absolute target itself; "
                        "normally do not call get_openarm_ee_pose. For a requested final "
                        "fingertip direction, call get_openarm_ee_pose first and take its "
                        "fingertip_direction as the current unit vector v; never derive v "
                        "from the quaternion. With the requested unit vector t in arm_origin "
                        "(+X forward, +Y left, +Z up), pass rotation_vector "
                        "r=unit(v cross t)*acos(clamp(v dot t)) with orientation_mode=relative "
                        "and rotation_frame=arm_origin; if t is opposite to v, use angle pi "
                        "about any axis perpendicular to v. "
                        "When no rotation is requested leave orientation_mode=arm_line: the "
                        "fingertips then follow the shoulder-to-target line like a straight arm. "
                        "Use keep only to hold an orientation set earlier, tolerant when "
                        "variation is allowed, and free only when any orientation is acceptable. "
                        "On failure you may retry once with small XYZ offsets unless the user "
                        "fixed position, keeping any requested rotation unchanged. Never "
                        "execute a smaller or different rotation than requested; if the retry "
                        "also fails, tell the user the motion failed and why. Stop on "
                        "START_STATE_IN_COLLISION and never execute failed plans. For "
                        "maximum-range requests compare several candidates and execute only "
                        "the best. "
                    ),
                ),
            ]
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
            answer = await agent_runtime.run(session, max_iterations=15, trace=True)
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
