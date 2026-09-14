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
                        "Arm motion: plan with plan_openarm_pose and run the returned "
                        "plan_id with execute_openarm_plan. They use MoveIt 2, which "
                        "checks robot self-collision and planning-scene collisions. "
                        "plan_openarm_reach and execute_openarm_reach are a legacy "
                        "custom-IK fallback with NO collision checking: use them only "
                        "when plan_openarm_pose fails because move_group is not "
                        "running, or when the user explicitly asks for the custom IK, "
                        "and tell the user that you used the fallback. "
                        "Never pass coordinates you estimated from camera images or "
                        "perception results to a motion tool. For relative moves "
                        "(e.g. 'up 5 cm'), read get_openarm_ee_pose first and offset "
                        "the current wrist position. "
                        "When choosing a target you are free to adjust x, y and z "
                        "together - do not anchor on the current x/y and change only "
                        "one axis, because a single direction is often limited while "
                        "varying the other axes (e.g. moving forward or inward) unlocks "
                        "much more range. "
                        "When a plan fails, read its error: NO_IK_SOLUTION or "
                        "GOAL_CONSTRAINTS_VIOLATED means the target is unreachable "
                        "with the current wrist orientation, so pick a closer or "
                        "different target; PLANNING_FAILED may succeed on one retry "
                        "because the planner is randomized; START_STATE_IN_COLLISION "
                        "cannot be fixed by changing the target, so stop and report "
                        "it. Treat a move as done only when execute_openarm_plan "
                        "returns success=true. "
                        "For extreme goals ('as much/high/far as possible', "
                        "尽可能/最大/最高/最远), do NOT stop at the first feasible "
                        "plan: propose several more aggressive candidates (you may "
                        "issue several plan_openarm_pose calls in one turn), compare "
                        "the feasible plans, then execute only the single best "
                        "plan_id."
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
