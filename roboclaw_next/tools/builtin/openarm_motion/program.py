"""MoveIt 2 motion planning and execution for the OpenArm MCP tools.

Plans are stored with the same store_plan/load_plan helpers as the
custom-IK reach tools, just in their own directory, so a plan_id always names the
exact trajectory MoveIt returned.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

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


async def plan_pose(
    arm: str,
    x: float,
    y: float,
    z: float,
    *,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    plan_root=PLAN_ROOT,
) -> dict[str, object]:
    """Plan a MoveIt motion of the wrist to (x, y, z) in arm_origin.

    The target orientation is the current wrist orientation, read from the
    real joints, rotated by roll_deg, pitch_deg and yaw_deg about the fixed
    arm_origin x, y and z axes; all three at zero keep it unchanged. Failed
    plans are stored too so every attempt has a plan_id; execute_plan refuses
    them.
    """

    joints = await read_joints(arm)
    quat = rotate_orientation(fk(arm, joints)[3:], roll_deg, pitch_deg, yaw_deg)
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


def rotate_orientation(
    quat_wxyz: Sequence[float],
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> list[float]:
    """Rotate a wxyz quaternion about the fixed arm_origin x, y and z axes.

    Roll about x is applied first, then pitch about y, then yaw about z, the
    same convention as ROS setRPY. The rotation is multiplied on the left, so
    the axes are those of arm_origin, not of the wrist.
    """

    half_roll, half_pitch, half_yaw = (
        math.radians(angle) / 2.0 for angle in (roll_deg, pitch_deg, yaw_deg)
    )
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    # Quaternion of Rz(yaw) * Ry(pitch) * Rx(roll).
    dw = cr * cp * cy + sr * sp * sy
    dx = sr * cp * cy - cr * sp * sy
    dy = cr * sp * cy + sr * cp * sy
    dz = cr * cp * sy - sr * sp * cy

    # Hamilton product: rotation * current orientation.
    qw, qx, qy, qz = (float(value) for value in quat_wxyz)
    return [
        dw * qw - dx * qx - dy * qy - dz * qz,
        dw * qx + dx * qw + dy * qz - dz * qy,
        dw * qy - dx * qz + dy * qw + dz * qx,
        dw * qz + dx * qy - dy * qx + dz * qw,
    ]
