"""MoveIt 2 motion planning and execution for the OpenArm MCP tools.

Plans are stored with the same store_plan/load_plan helpers as the
custom-IK reach tools, just in their own directory, so a plan_id always names the
exact trajectory MoveIt returned.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Literal

from robot_runtime.openarm_control import moveit_client
from robot_runtime.openarm_ik import ORIGIN_FRAME, arm as arm_model, fk
from roboclaw_next.tools.builtin.openarm_reach.program import (
    REPOSITORY_ROOT,
    load_plan,
    read_joints,
    store_plan,
)


PLAN_ROOT = REPOSITORY_ROOT / "runtime_data" / "openarm_motion"
SPEED_SCALE = 0.2  # fraction of the joint velocity and acceleration limits

RelativeOrientationMode = Literal["arm_line", "keep", "relative", "tolerant", "free"]
RelativeFrame = Literal["arm_origin", "tool"]
DEFAULT_RELATIVE_ORIENTATION_TOLERANCE_RAD = 0.20


async def get_ee_pose(arm: str) -> dict[str, object]:
    """Read the wrist pose in arm_origin from the real joints.

    Besides pose[7], the result carries the fingertip direction, so the model
    never has to work it out from the quaternion.
    """

    joints = await read_joints(arm)
    pose = fk(arm, joints)
    return {
        "arm": arm,
        "frame": ORIGIN_FRAME,
        "pose": list(pose),
        "fingertip_direction": fingertip_direction(pose[3:]),
        "joints": list(joints),
    }


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


async def plan_relative(
    arm: str,
    translation: Sequence[float],
    *,
    translation_frame: RelativeFrame = "arm_origin",
    orientation_mode: RelativeOrientationMode = "arm_line",
    rotation_vector: Sequence[float] | None = None,
    rotation_frame: RelativeFrame = "tool",
    orientation_tolerance_rad: float = DEFAULT_RELATIVE_ORIENTATION_TOLERANCE_RAD,
    plan_root=PLAN_ROOT,
) -> dict[str, object]:
    """Plan from the latest EE pose using a relative SE(3) command.

    ``translation`` is expressed in ``translation_frame``. By default the
    target orientation is not derived from the current one: the fingertips
    point along the line from the shoulder centre to the target position, as
    they would on a straight arm. ``rotation_vector`` is axis multiplied by
    angle in radians and is composed on the right in the tool frame or on the
    left in ``arm_origin``. The function snapshots the joints once, computes
    the absolute target internally, and stores the MoveIt plan for the
    existing ``execute_openarm_plan`` tool.

    Args:
        arm: Arm to plan for, either ``"right"`` or ``"left"``.
        translation: Relative EE translation ``[dx, dy, dz]`` in metres.
        translation_frame: Frame of ``translation``. ``"arm_origin"`` uses
            fixed robot axes; ``"tool"`` uses the current wrist-local axes.
        orientation_mode: How to constrain the target orientation.
            ``"arm_line"`` uses the straight-arm orientation from
            ``arm_line_orientation``, ``"keep"`` preserves the current
            orientation, ``"relative"`` applies ``rotation_vector``,
            ``"tolerant"`` applies it with a relaxed tolerance, and ``"free"``
            sends no orientation constraint.
        rotation_vector: Relative axis-angle rotation ``[rx, ry, rz]`` in
            radians, where the direction is the rotation axis and the norm is
            the angle. ``None`` means zero rotation.
        rotation_frame: Frame of ``rotation_vector``. ``"tool"`` rotates about
            current wrist-local axes; ``"arm_origin"`` rotates about fixed axes.
        orientation_tolerance_rad: Angular tolerance in radians for
            ``orientation_mode="tolerant"``. Other modes use their fixed policy.
        plan_root: Directory in which the generated plan JSON is stored.

    Returns:
        A stored-plan dictionary containing the source and resolved target poses,
        normalized relative-command metadata, the MoveIt result and trajectory,
        and the generated ``plan_id``, ``plan_file``, and ``created_at`` fields.
        The trajectory is planned but is not executed by this function.
    """

    joints = await read_joints(arm)
    current_pose = tuple(float(value) for value in fk(arm, joints))
    delta_translation = _vector3(translation, "translation")
    delta_rotation = _vector3(
        rotation_vector if rotation_vector is not None else (0.0, 0.0, 0.0),
        "rotation_vector",
    )
    target_position, target_quat = compose_relative_target(
        current_pose,
        delta_translation,
        translation_frame=translation_frame,
        orientation_mode=orientation_mode,
        rotation_vector=delta_rotation,
        rotation_frame=rotation_frame,
        shoulder=shoulder_position(arm),
    )

    if orientation_mode == "tolerant":
        tolerance = float(orientation_tolerance_rad)
        if not math.isfinite(tolerance) or not 0.0 < tolerance <= math.pi:
            raise ValueError("orientation_tolerance_rad must be in (0, pi]")
    elif orientation_mode == "free":
        tolerance = None
    else:
        tolerance = moveit_client.ORIENTATION_TOLERANCE_RAD

    plan = await asyncio.to_thread(
        moveit_client.plan_pose,
        arm,
        target_position,
        target_quat,
        SPEED_SCALE,
        orientation_tolerance_rad=tolerance or moveit_client.ORIENTATION_TOLERANCE_RAD,
    )
    payload = {
        "arm": arm,
        "frame": ORIGIN_FRAME,
        "motion_kind": "relative_ee",
        "source_pose": list(current_pose),
        "translation": list(delta_translation),
        "translation_frame": translation_frame,
        "orientation_mode": orientation_mode,
        "rotation_vector": list(delta_rotation),
        "rotation_frame": rotation_frame,
        "orientation_tolerance_rad": tolerance,
        "target_pose": [*target_position, *(target_quat or ())],
        **plan,
    }
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


def compose_relative_target(
    current_pose: Sequence[float],
    translation: Sequence[float],
    *,
    translation_frame: RelativeFrame,
    orientation_mode: RelativeOrientationMode,
    rotation_vector: Sequence[float],
    rotation_frame: RelativeFrame,
    shoulder: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float] | None]:
    """Compose a relative translation and rotation with a pose[7] in arm_origin.

    ``shoulder`` is the start of the line that ``orientation_mode="arm_line"``
    points the fingertips along.
    """

    pose = tuple(float(value) for value in current_pose)
    if len(pose) != 7 or not all(math.isfinite(value) for value in pose):
        raise ValueError("current_pose must be finite [x, y, z, qw, qx, qy, qz]")
    delta_position = _vector3(translation, "translation")
    delta_rotation = _vector3(rotation_vector, "rotation_vector")
    current_quat = _normalize_quaternion(pose[3:])

    if translation_frame == "tool":
        delta_position = _rotate_vector(current_quat, delta_position)
    elif translation_frame != "arm_origin":
        raise ValueError("translation_frame must be 'arm_origin' or 'tool'")
    target_position = tuple(pose[index] + delta_position[index] for index in range(3))

    rotation_angle = math.sqrt(sum(component * component for component in delta_rotation))
    if orientation_mode in {"arm_line", "keep", "free"} and rotation_angle > 1e-12:
        raise ValueError(
            f"rotation_vector requires orientation_mode='relative' or 'tolerant', got {orientation_mode!r}"
        )
    if orientation_mode == "free":
        return target_position, None
    if orientation_mode == "keep":
        return target_position, current_quat
    if orientation_mode == "arm_line":
        return target_position, arm_line_orientation(shoulder, target_position)
    if orientation_mode not in {"relative", "tolerant"}:
        raise ValueError(f"unsupported relative orientation mode: {orientation_mode!r}")

    delta_quat = _quaternion_from_rotation_vector(delta_rotation)
    if rotation_frame == "tool":
        target_quat = _hamilton_product(current_quat, delta_quat)
    elif rotation_frame == "arm_origin":
        target_quat = _hamilton_product(delta_quat, current_quat)
    else:
        raise ValueError("rotation_frame must be 'arm_origin' or 'tool'")
    return target_position, _normalize_quaternion(target_quat)


def shoulder_position(arm: str) -> tuple[float, float, float]:
    """Return the shoulder centre of ``arm`` in arm_origin.

    This is the pivot of joint 2, where the axes of joints 1 to 3 meet. Joint 2
    sits on the axis of joint 1, so the point never moves, and the upper arm
    always starts there.
    """

    kinematics = arm_model(arm)
    first, second = kinematics.hinges[:2]
    return tuple(
        kinematics.base_from_origin[index] + first.origin[index] + second.origin[index]
        for index in range(3)
    )


def arm_line_orientation(
    shoulder: Sequence[float], target: Sequence[float]
) -> tuple[float, float, float, float]:
    """Return the wrist orientation of a straight arm from shoulder towards target.

    The fingertips (wrist-local -Z) point along the shoulder-to-target line.
    The roll about that line is the one given by joints 1 and 2 alone, with
    every other joint at zero, so wrist-local X stays in the arm_origin XZ
    plane. A line along arm_origin Y leaves that roll undefined; joint 1 is
    then taken as zero.
    """

    start = _vector3(shoulder, "shoulder")
    end = _vector3(target, "target")
    line = tuple(end[index] - start[index] for index in range(3))
    length = math.sqrt(sum(component * component for component in line))
    if length <= 1e-9:
        raise ValueError("target is at the shoulder, so the arm line has no direction")
    dx, dy, dz = (component / length for component in line)

    # The arm hanging straight down has the identity orientation. Joint 2 lifts
    # it about arm_origin X, then joint 1 swings it about arm_origin Y. The axis
    # signs differ between the arms, but the resulting orientation does not.
    about_x = math.asin(max(-1.0, min(1.0, dy))) # 已知方向向量里的 Y 分量，反推出绕 X 轴需要多少角度。
    about_y = math.atan2(-dx, -dz) if math.hypot(dx, dz) > 1e-9 else 0.0 # 给一个 (x,z) 点，atan2 告诉默认-z方向绕y轴转多少度。
    return _normalize_quaternion(
        _hamilton_product( # 两个旋转合并为一个旋转
            (math.cos(about_y / 2.0), 0.0, math.sin(about_y / 2.0), 0.0),
            (math.cos(about_x / 2.0), math.sin(about_x / 2.0), 0.0, 0.0),
        )# 最后归一化出四元数
    )


def fingertip_direction(quat_wxyz: Sequence[float]) -> list[float]:
    """Return the unit vector in arm_origin along which the fingertips point.

    The fingertips point along wrist-local -Z, so [0, 0, -1] is straight down.
    Values are rounded to 4 decimals so float noise such as 6e-17 does not read
    as a direction component.
    """

    direction = _rotate_vector(quat_wxyz, (0.0, 0.0, -1.0))
    # Adding 0.0 turns a rounded -0.0 into 0.0.
    return [round(component, 4) + 0.0 for component in direction]


def _vector3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain exactly 3 finite values")
    return result


def _quaternion_from_rotation_vector(
    rotation_vector: Sequence[float],
) -> tuple[float, float, float, float]:
    vector = _vector3(rotation_vector, "rotation_vector")
    angle = math.sqrt(sum(component * component for component in vector))
    if angle <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    half_angle = angle / 2.0
    scale = math.sin(half_angle) / angle
    return _normalize_quaternion(
        (math.cos(half_angle), *(component * scale for component in vector))
    )


def _rotate_vector(
    quat_wxyz: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    q = _normalize_quaternion(quat_wxyz)
    rotated = _hamilton_product(
        _hamilton_product(q, (0.0, *vector)),
        (q[0], -q[1], -q[2], -q[3]),
    )
    return (rotated[1], rotated[2], rotated[3])


def _hamilton_product(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = left
    bw, bx, by, bz = right
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _normalize_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in values)
    if len(values) != 4:
        raise ValueError(f"quaternion must have 4 components, got {len(values)}")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("quaternion must not be zero")
    return (values[0] / norm, values[1] / norm, values[2] / norm, values[3] / norm)
