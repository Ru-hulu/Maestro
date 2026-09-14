"""MCP contracts for OpenArm forward and inverse kinematics."""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .program import execute_plan, get_ee_pose, plan_to_xyz


class EePoseResult(BaseModel):
    """Current end-effector pose in the arm_origin frame."""

    arm: Literal["right", "left"] = Field(description="Which OpenArm chain was queried.")
    frame: str = Field(description="Pose frame. Always arm_origin.")
    pose: list[float] = Field(
        description="EE pose [px, py, pz, qw, qx, qy, qz] in metres and unit quaternion.",
    )
    rpy_deg: list[float] = Field(
        description=(
            "EE orientation as [roll, pitch, yaw] in degrees: rotations about the "
            "fixed arm_origin x, y and z axes, applied in that order. The gripper "
            "points straight down at [0, 0, any yaw]. Near pitch +-90 roll and yaw "
            "are not unique. The roll_deg, pitch_deg and yaw_deg of "
            "plan_openarm_pose are rotations added on top of the current "
            "orientation, not absolute angles."
        ),
    )
    joints: list[float] = Field(
        description=(
            "The 7 arm joints this pose was computed from, read from "
            "/joint_states at call time."
        ),
    )


class TrajectoryPointResult(BaseModel):
    """One sample along the planned joint trajectory."""

    t: float = Field(description="Time from the start of the plan, in seconds.")
    q: list[float] = Field(description="Joint command at this sample.")
    ee_pose: list[float] = Field(description="EE pose[7] in the arm_origin frame.")


class ReachPlanResult(BaseModel):
    """Joint trajectory produced by one atomic OpenArm reach."""

    ok: bool = Field(description="True when the final position error is within tolerance.")
    failure_reason: str = Field(
        description=(
            "Why the solve stopped: converged, max_steps (still improving when the step budget ran out), singularity_locked (the arm sits at a singularity and could not move), joint_limit_blocked (a joint hit its stop), or no_progress (the solver stalled, including after retries from other starting postures)."
        ),
    )
    frame: str = Field(description="Pose frame. Always arm_origin.")
    arm: Literal["right", "left"] = Field(description="Which OpenArm chain was planned.")
    dt: float = Field(description="Outer-loop sample period in seconds.")
    initial_error_m: float = Field(
        description=(
            "Euclidean position error before the first step. Equal to "
            "final_error_m means the solver never moved the arm."
        ),
    )
    final_error_m: float = Field(description="Final Euclidean position error in metres.")
    escape_blend: float = Field(
        description=(
            "0.0 when the plan started solving directly. Otherwise the "
            "fraction toward the home posture that the trajectory detours "
            "through to leave a blocked start configuration."
        ),
    )
    target_pose: list[float] = Field(description="Assembled target pose[7] in arm_origin.")
    message: str = Field(description="Short summary for the model, including millimetre error.")
    points: list[TrajectoryPointResult] = Field(
        description="Trajectory samples, starting at the current configuration.",
    )
    schema_version: int = Field(description="Stored plan schema version.")
    plan_id: str = Field(description="Unique immutable plan identifier.")
    plan_file: str = Field(description="Absolute path of the stored full trajectory.")
    created_at: str = Field(description="UTC timestamp at which the plan was stored.")


class ReachPlanSummary(BaseModel):
    """`ReachPlanResult` 面向模型的投影，不含 points 轨迹数组。

    模型驱动不了电机，逐点关节角对它没有用处；它真正需要判断的只是
    （够不够得着、差多少、生成了几个点）。
    """

    ok: bool = Field(description="True when the final position error is within tolerance.")
    failure_reason: str = Field(
        description=(
            "Why the solve stopped: converged, max_steps (still improving when the step budget ran out), singularity_locked (the arm sits at a singularity and could not move), joint_limit_blocked (a joint hit its stop), or no_progress (the solver stalled, including after retries from other starting postures)."
        ),
    )
    frame: str = Field(description="Pose frame. Always arm_origin.")
    arm: Literal["right", "left"] = Field(description="Which OpenArm chain was planned.")
    dt: float = Field(description="Outer-loop sample period in seconds.")
    initial_error_m: float = Field(
        description=(
            "Euclidean position error before the first step. Equal to "
            "final_error_m means the solver never moved the arm."
        ),
    )
    final_error_m: float = Field(description="Final Euclidean position error in metres.")
    escape_blend: float = Field(
        description=(
            "0.0 when the plan started solving directly. Otherwise the "
            "fraction toward the home posture that the trajectory detours "
            "through to leave a blocked start configuration."
        ),
    )
    target_pose: list[float] = Field(description="Assembled target pose[7] in arm_origin.")
    message: str = Field(description="Short summary for the model, including millimetre error.")
    point_count: int = Field(
        description=(
            "Number of trajectory samples the plan produced. The trajectory "
            "itself is not returned."
        ),
    )
    plan_id: str = Field(description="Unique identifier accepted by execute_openarm_reach.")
    plan_file: str = Field(description="Absolute path of the stored full trajectory.")
    created_at: str = Field(description="UTC timestamp at which the plan was stored.")

    @classmethod
    def from_plan(cls, plan: ReachPlanResult) -> ReachPlanSummary:
        """从完整规划结果投影出面向模型的摘要。"""

        return cls(
            ok=plan.ok,
            failure_reason=plan.failure_reason,
            frame=plan.frame,
            arm=plan.arm,
            dt=plan.dt,
            initial_error_m=plan.initial_error_m,
            final_error_m=plan.final_error_m,
            escape_blend=plan.escape_blend,
            target_pose=plan.target_pose,
            message=plan.message,
            point_count=len(plan.points),
            plan_id=plan.plan_id,
            plan_file=plan.plan_file,
            created_at=plan.created_at,
        )


class OpenArmExecutionResult(BaseModel):
    """Result of sending one stored plan to FollowJointTrajectory."""

    success: bool = Field(description="Whether the trajectory action reported success.")
    sent: bool = Field(description="Whether a non-empty action goal was sent.")
    plan_id: str = Field(description="The immutable plan that was executed.")
    arm: Literal["right", "left"] = Field(description="The commanded OpenArm chain.")
    action_name: str = Field(description="ROS 2 FollowJointTrajectory action name.")
    point_count: int = Field(description="Number of position commands sent.")
    duration_sec: float = Field(description="Retimed trajectory duration in seconds.")
    speed_scale: float = Field(description="Fraction of documented joint velocity limits.")
    start_error_rad: float = Field(description="Largest start-joint mismatch in radians.")
    status: int = Field(description="ROS action terminal status code.")
    error_code: int = Field(description="FollowJointTrajectory result error code.")
    error_string: str = Field(description="Controller-provided result detail.")
    collision_checked: bool = Field(
        description="Always false in this minimal executor; no collision checking is done."
    )
    message: str = Field(description="Short execution summary.")


def register_openarm_reach_tools(mcp: FastMCP) -> None:
    """Register OpenArm pose and reach Tools."""

    @mcp.tool(
        name="get_openarm_ee_pose",
        title="Get OpenArm EE Pose",
        description=(
            "Read the OpenArm end-effector pose. The tool reads the arm's real "
            "joint angles from a continuously updated /joint_states cache; do "
            "not supply them. Poses are in the arm_origin frame, in metres. "
            "Fails if the robot is not publishing fresh joint states."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def get_openarm_ee_pose(
        arm: Annotated[
            Literal["right", "left"],
            Field(description="Which arm to read."),
        ],
    ) -> EePoseResult:
        """Return the current EE pose in the arm_origin frame."""

        return EePoseResult.model_validate(await get_ee_pose(arm))

    @mcp.tool(
        name="plan_openarm_reach",
        title="Plan OpenArm Reach",
        description=(
            "Plan a joint trajectory that moves one OpenArm chain to an xyz "
            "target in the arm_origin frame. The plan starts from the arm's real "
            "joint angles in a continuously updated /joint_states cache; do not "
            "supply them. Orientation is kept from the current end-effector pose. "
            "This is an IK calculation only; it does not command motors. Provide "
            "x, y, z in metres (x forward, y left, z up/height). A max_steps "
            "failure means the solver was still converging, so the target is not "
            "proven out of range; re-planning or adjusting the target (for example "
            "its height z) may succeed. The full joint trajectory is stored under "
            "a unique plan_id but is not returned to the model. Use ok and "
            "final_error_m to judge whether the target is reachable, and "
            "failure_reason to tell an unreachable target apart from a blocked arm "
            "posture; do not infer a reachability limit from a few failed attempts. "
            "Then pass plan_id to execute_openarm_reach."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    )
    async def plan_openarm_reach(
        arm: Annotated[
            Literal["right", "left"],
            Field(description="Which arm to plan."),
        ],
        x: Annotated[float, Field(description="Target x (forward) in arm_origin, metres.")],
        y: Annotated[float, Field(description="Target y (left) in arm_origin, metres.")],
        z: Annotated[float, Field(description="Target z (up/height) in arm_origin, metres.")],
    ) -> ReachPlanSummary:
        """Plan a reach and return a summary of the joint trajectory.

        轨迹本身不进模型上下文：81 个采样点实测约 6700 token，而模型驱动不了
        电机，拿到逐点关节角也用不上。
        """

        plan = ReachPlanResult.model_validate(await plan_to_xyz(arm, x, y, z))
        return ReachPlanSummary.from_plan(plan)

    @mcp.tool(
        name="execute_openarm_reach",
        title="Execute OpenArm Reach",
        description=(
            "Send one previously generated OpenArm reach plan to the ROS 2 "
            "FollowJointTrajectory controller. Provide the plan_id returned by "
            "plan_openarm_reach. Before sending, the tool reloads the immutable "
            "full trajectory, requires a converged plan, reads fresh joint state, "
            "and rejects the plan if the arm moved more than 0.05 rad from its "
            "planned start. Points are retimed against the documented joint "
            "velocity limits. This first executor does NOT perform self-collision "
            "or environment-collision checking and does not command the gripper."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=False,
        ),
    )
    async def execute_openarm_reach(
        plan_id: Annotated[
            str,
            Field(description="Plan identifier returned by plan_openarm_reach."),
        ],
        speed_scale: Annotated[
            float,
            Field(
                gt=0.0,
                le=1.0,
                description=(
                    "Fraction of documented joint velocity limits; 0.2 is the "
                    "conservative default."
                ),
            ),
        ] = 0.2,
    ) -> OpenArmExecutionResult:
        """Execute a stored plan through FollowJointTrajectory."""

        return OpenArmExecutionResult.model_validate(
            await execute_plan(plan_id, speed_scale=speed_scale)
        )
