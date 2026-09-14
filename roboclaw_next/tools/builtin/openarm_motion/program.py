"""MoveIt 2 motion planning and execution for the OpenArm MCP tools.

Plans are stored with the same store_plan/load_plan helpers as the
custom-IK reach tools, just in their own directory, so a plan_id always names the
exact trajectory MoveIt returned.
"""

from __future__ import annotations

import asyncio

from robot_runtime.openarm_control import moveit_client
from robot_runtime.openarm_ik import ORIGIN_FRAME, fk
from roboclaw_next.tools.builtin.openarm_reach.program import (
    REPOSITORY_ROOT,
    load_plan,
    read_joints,
    store_plan,
)


PLAN_ROOT = REPOSITORY_ROOT / "runtime_data" / "openarm_motion"
SPEED_SCALE = 0.2  # fraction of the joint velocity and acceleration limits


async def plan_pose(arm: str, x: float, y: float, z: float, *, plan_root=PLAN_ROOT) -> dict[str, object]:
    """Plan a MoveIt motion of the wrist to (x, y, z) in arm_origin.

    The wrist keeps its current orientation, read from the real joints, like
    plan_openarm_reach. Failed plans are stored too so every attempt has a
    plan_id; execute_plan refuses them.
    """

    joints = await read_joints(arm)
    quat = list(fk(arm, joints)[3:])
    plan = await asyncio.to_thread(moveit_client.plan_pose, arm, (x, y, z), quat, SPEED_SCALE)
    payload = {"arm": arm, "frame": ORIGIN_FRAME, "target_pose": [x, y, z, *quat], **plan}
    return store_plan(payload, plan_root=plan_root)


async def execute_plan(plan_id: str, *, plan_root=PLAN_ROOT) -> dict[str, object]:
    """Run one stored MoveIt plan.

    move_group itself rejects the trajectory if the arm moved after planning,
    so a plan cannot be replayed from a different posture.
    """

    plan = load_plan(plan_id, plan_root=plan_root)
    if not plan.get("ok"):
        raise ValueError(
            f"Plan {plan_id} failed ({plan.get('error')}) and cannot be executed. "
            "Plan again with plan_openarm_pose."
        )
    duration = plan["points"][-1]["t"]
    result = await asyncio.to_thread(
        moveit_client.execute,
        plan["joint_names"],
        plan["points"],
        duration * 2.0 + 10.0,  # generous limit; the simulator may run slower than real time
    )
    return {"plan_id": plan["plan_id"], "arm": plan["arm"], "duration_sec": duration, **result}
