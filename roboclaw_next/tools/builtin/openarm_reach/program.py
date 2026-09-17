"""In-process OpenArm FK/IK used by the MCP tools.

Joint angles come from a process-wide listener that continuously caches the
robot's `/joint_states` topic. They are never a tool parameter: the model cannot
observe them, so anything it typed would be a guess. This layer reads the real
state, calls the pure kinematics, and converts the planner's tuples into
JSON-friendly dictionaries.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from robot_runtime.openarm_control import (
    read_arm_joint_state,
    retime_joint_positions,
    send_joint_trajectory,
)
from robot_runtime.openarm_ik import ORIGIN_FRAME, ReachPlan, fk, plan_reach
from robot_runtime.openarm_ik.model import arm as arm_model
from robot_runtime.openarm_ik.planner import (
    JOINT_LIMIT_BLOCKED,
    NO_PROGRESS,
    OUT_OF_REACH,
    SINGULARITY_LOCKED,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PLAN_ROOT = REPOSITORY_ROOT / "runtime_data" / "openarm"
# 读取关节状态时的"最大允许数据新鲜度"，避免使用过时的关节状态数据
EXECUTION_STATE_MAX_AGE_SEC = 0.25
# 起始容差（0.05 弧度 ≈ 2.9°）。执行时会把当前关节角度和 plan 的第一个点做比较，取 7 个关节里误差最大的那个（start_error）。如果它超过 tolerance，就拒绝执行并报错："机械臂在规划之后动过了，请重新规划"。
START_TOLERANCE_RAD = 0.05


async def read_joints(
    arm: str,
    *,
    max_age_sec: float = 1.0,
) -> list[float]:
    """Read the latest fresh 7-joint snapshot, off the event loop."""

    state = await asyncio.to_thread(read_arm_joint_state, arm, 2.0, max_age_sec)
    return list(state.positions)


async def get_ee_pose(arm: str) -> dict[str, object]:
    """Read the EE pose[7] in the arm_origin frame from the current joints."""

    return build_ee_pose(arm, await read_joints(arm))


async def plan_to_xyz(
    arm: str,
    x: float,
    y: float,
    z: float,
) -> dict[str, object]:
    """Plan a reach from the current joints and persist its full trajectory."""

    joints = await read_joints(arm)
    payload = await asyncio.to_thread(plan_from_joints, arm, joints, x, y, z)
    return store_plan(payload)


async def execute_plan(
    plan_id: str,
    *,
    speed_scale: float = 0.2,
    plan_root: Path = DEFAULT_PLAN_ROOT,
) -> dict[str, object]:
    """Load, validate, retime, and send one stored reach trajectory."""

    payload = load_plan(plan_id, plan_root=plan_root)
    if payload.get("ok") is not True:
        raise ValueError("Cannot execute an OpenArm plan that did not converge.")
    if payload.get("frame") != ORIGIN_FRAME:
        raise ValueError(
            f"OpenArm plan frame must be {ORIGIN_FRAME!r}, got {payload.get('frame')!r}."
        )

    side = payload.get("arm")
    positions = _plan_positions(payload)

    current = await read_joints(side, max_age_sec=EXECUTION_STATE_MAX_AGE_SEC)
    start_error = max(
        abs(actual - planned) for actual, planned in zip(current, positions[0])
    )
    if start_error > START_TOLERANCE_RAD:
        raise RuntimeError(
            "OpenArm moved after planning: maximum start-joint error is "
            f"{start_error:.4f} rad, exceeding {START_TOLERANCE_RAD:.4f} rad. "
            "Create a new reach plan before executing."
        )

    timed_points = retime_joint_positions(positions, speed_scale=speed_scale)
    execution = await asyncio.to_thread(send_joint_trajectory, side, timed_points)
    message = (
        f"OpenArm trajectory completed in {execution.duration_sec:.2f} s."
        if execution.success
        else (
            "OpenArm trajectory controller reported failure: "
            f"status={execution.status}, error_code={execution.error_code}, "
            f"detail={execution.error_string}"
        )
    )
    return {
        "success": execution.success,
        "sent": execution.sent,
        "plan_id": payload["plan_id"],
        "arm": side,
        "action_name": execution.action_name,
        "point_count": execution.point_count,
        "duration_sec": execution.duration_sec,
        "speed_scale": speed_scale,
        "start_error_rad": start_error,
        "status": execution.status,
        "error_code": execution.error_code,
        "error_string": execution.error_string,
        "collision_checked": False,
        "message": message,
    }


def build_ee_pose(arm: str, joints: Sequence[float]) -> dict[str, object]:
    """FK payload for a given joint vector. Pure; ROS is the caller's problem."""

    values = [float(value) for value in joints]
    pose = fk(arm, values)
    return {
        "arm": arm,
        "frame": ORIGIN_FRAME,
        "pose": list(pose),
        "rpy_deg": quaternion_to_rpy_deg(pose[3:]),
        "joints": values,
    }


def quaternion_to_rpy_deg(quat_wxyz: Sequence[float]) -> list[float]:
    """Convert a wxyz quaternion to [roll, pitch, yaw] in degrees.

    Roll, pitch and yaw are rotations about the fixed arm_origin x, y and z
    axes applied in that order, R = Rz(yaw) * Ry(pitch) * Rx(roll), the same
    convention as ROS setRPY. Near pitch = +-90 degrees roll and yaw are not
    unique (gimbal lock).
    """

    w, x, y, z = (float(value) for value in quat_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def plan_from_joints(
    arm: str,
    joints: Sequence[float],
    x: float,
    y: float,
    z: float,
) -> dict[str, object]:
    """Plan from an explicit start, keeping the current EE orientation.

    Pure, so tests can pin a start configuration. The MCP tools never expose
    this start to the model.
    """

    return serialize_plan(plan_reach(arm, joints, (x, y, z)))


def serialize_plan(plan: ReachPlan) -> dict[str, object]:
    """Convert a ReachPlan into a structured tool payload."""

    return {
        "ok": plan.ok,
        "failure_reason": plan.failure_reason,
        "frame": plan.frame,
        "arm": plan.arm,
        "dt": plan.dt,
        "initial_error_m": plan.initial_error_m,
        "final_error_m": plan.final_error_m,
        "escape_blend": plan.escape_blend,
        "target_pose": list(plan.target_pose),
        "message": describe_plan(plan),
        "points": [
            {
                "t": point.t,
                "q": list(point.q),
                "ee_pose": list(point.ee_pose),
            }
            for point in plan.points
        ],
    }


def describe_plan(plan: ReachPlan) -> str:
    """Explain the outcome in the terms the model actually needs.

    A bare "max_steps" made every failure look alike, so the model invented its
    own explanation. Each reason below says what blocked the solve and whether
    the error moved at all.
    """

    steps = max(0, len(plan.points) - 1)
    start_mm = plan.initial_error_m * 1000.0
    final_mm = plan.final_error_m * 1000.0
    frame = plan.frame

    if plan.ok:
        message = (
            f"Reached the target in the {frame} frame with {final_mm:.1f} mm "
            f"error after {steps} steps."
        )
        if plan.escape_blend > 0.0:
            message += (
                f" The start configuration was blocked, so the trajectory first "
                f"moves {plan.escape_blend * 100:.0f}% of the way toward the home "
                "posture and reaches the target from there."
            )
        return message

    if plan.failure_reason == OUT_OF_REACH:
        kinematics = arm_model(plan.arm)
        distance = math.dist(plan.target_pose[:3], kinematics.base_from_origin)
        return (
            f"The target is {distance:.3f} m from the {plan.arm} arm base, beyond "
            f"its {kinematics.max_reach:.3f} m maximum link length, so no posture "
            "can reach it and no IK was attempted. Pick a closer target, or move "
            "the mobile base first."
        )
    if plan.failure_reason == SINGULARITY_LOCKED:
        return (
            f"The arm is at a kinematic singularity, so the solver could not move "
            f"it: position error stayed at {final_mm:.1f} mm. Retrying the same "
            "target will not help. Move the arm to a different posture first, for "
            "example by reaching a nearby point that bends the elbow."
        )
    if plan.failure_reason == JOINT_LIMIT_BLOCKED:
        return (
            f"A joint reached its travel limit with {final_mm:.1f} mm still to go "
            f"(from {start_mm:.1f} mm). The target is outside this arm's range in "
            "the current orientation."
        )
    if plan.failure_reason == NO_PROGRESS:
        return (
            f"The solver stopped making progress at {final_mm:.1f} mm error "
            f"(from {start_mm:.1f} mm) after {steps} steps, including retries from "
            "other starting postures. Treat this target as unreachable while the "
            "end-effector orientation is held fixed."
        )
    return (
        f"The step budget ran out while still converging: {final_mm:.1f} mm error "
        f"remains (from {start_mm:.1f} mm) after {steps} steps. The solver was still "
        "reducing the error, so the target is not proven out of range; re-planning "
        "or adjusting the target (for example its height z) may succeed."
    )


def store_plan(
    payload: dict[str, object],
    *,
    plan_root: Path = DEFAULT_PLAN_ROOT,
) -> dict[str, object]:
    """Persist one immutable plan under a UUID and return the stored payload."""

    plan_id = str(uuid.uuid4())
    plan_directory = plan_root / plan_id
    plan_file = plan_directory / "plan.json"
    stored = {
        **payload,
        "schema_version": 1,
        "plan_id": plan_id,
        "plan_file": str(plan_file.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    plan_directory.mkdir(parents=True, exist_ok=False)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=plan_directory,
            prefix=".plan.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(stored, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(plan_file)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to store OpenArm plan: {plan_file}") from exc
    return stored


def load_plan(
    plan_id: str,
    *,
    plan_root: Path = DEFAULT_PLAN_ROOT,
) -> dict[str, object]:
    """Load an OpenArm plan by UUID without accepting arbitrary file paths."""

    try:
        canonical_id = str(uuid.UUID(plan_id))
    except (ValueError, AttributeError) as exc:
        raise ValueError("plan_id must be a valid UUID") from exc
    plan_file = plan_root / canonical_id / "plan.json"
    try:
        payload = json.loads(plan_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"OpenArm plan does not exist: {canonical_id}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"OpenArm plan is unreadable: {canonical_id}") from exc
    if not isinstance(payload, dict) or payload.get("plan_id") != canonical_id:
        raise ValueError(f"OpenArm plan identity is invalid: {canonical_id}")
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported OpenArm plan schema: {payload.get('schema_version')!r}"
        )
    return payload


def _plan_positions(payload: dict[str, object]) -> tuple[tuple[float, ...], ...]:
    raw_points = payload.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("Stored OpenArm plan has no trajectory points.")
    positions: list[tuple[float, ...]] = []
    for index, point in enumerate(raw_points):
        if not isinstance(point, dict) or not isinstance(point.get("q"), list):
            raise ValueError(f"Stored OpenArm trajectory point {index} is invalid.")
        values = tuple(float(value) for value in point["q"])
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Stored OpenArm trajectory point {index} has invalid joints."
            )
        positions.append(values)
    return tuple(positions)
