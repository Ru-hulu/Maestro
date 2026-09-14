"""MCP contracts for MoveIt-backed OpenArm motion planning and execution."""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .program import execute_plan, plan_pose


class MotionPlanSummary(BaseModel):
    """What the model needs from a MoveIt plan; the trajectory stays on disk."""

    ok: bool = Field(description="True when MoveIt found a collision-free trajectory.")
    error: str = Field(
        description=(
            "MoveIt result: SUCCESS, NO_IK_SOLUTION, GOAL_CONSTRAINTS_VIOLATED, "
            "PLANNING_FAILED, GOAL_IN_COLLISION, START_STATE_IN_COLLISION, ..."
        ),
    )
    arm: Literal["right", "left"] = Field(description="Which arm was planned.")
    frame: str = Field(description="Frame of target_pose. Always arm_origin.")
    target_pose: list[float] = Field(
        description="Wrist target [x, y, z, qw, qx, qy, qz] in arm_origin.",
    )
    point_count: int = Field(description="Number of trajectory samples stored.")
    duration_sec: float = Field(description="Planned motion duration in seconds.")
    planning_time_sec: float = Field(description="Time MoveIt spent planning.")
    plan_id: str = Field(description="Identifier accepted by execute_openarm_plan.")
    plan_file: str = Field(description="Absolute path of the stored plan.")
    created_at: str = Field(description="UTC time at which the plan was stored.")


class MotionExecutionResult(BaseModel):
    """Result of running one stored MoveIt plan."""

    success: bool = Field(description="True when MoveIt reported the motion finished.")
    error: str = Field(description="MoveIt result, e.g. SUCCESS or CONTROL_FAILED.")
    plan_id: str = Field(description="The plan that was executed.")
    arm: Literal["right", "left"] = Field(description="The arm that moved.")
    duration_sec: float = Field(description="Planned motion duration in seconds.")


def register_openarm_motion_tools(mcp: FastMCP) -> None:
    """Register the MoveIt-backed OpenArm motion tools."""

    @mcp.tool(
        name="plan_openarm_pose",
        title="Plan OpenArm Motion (MoveIt)",
        description=(
            "Plans a trajectory with MoveIt 2 that moves one arm's wrist "
            "(openarm_<arm>_ee_base_link) to x, y, z in the arm_origin frame, in "
            "metres (x forward, y left, z up). The gripper fingertips point along "
            "the negative z axis of the wrist frame openarm_<arm>_ee_base_link. "
            "The wrist orientation starts from "
            "the current one: roll_deg, pitch_deg and yaw_deg rotate it about the "
            "arm_origin x, y and z axes (applied in that order), and leaving all "
            "three at 0 keeps it unchanged. To only tilt or turn the wrist, pass "
            "the current x, y, z from get_openarm_ee_pose. The start state is read "
            "from the robot: never supply joint angles, and never type coordinates "
            "you estimated from camera images or perception results. MoveIt checks "
            "robot self-collision and objects in its planning scene; the table is "
            "not in the scene yet, so keep targets clear of it. The trajectory is "
            "stored under plan_id and not returned. If ok is false, error says "
            "why: NO_IK_SOLUTION or GOAL_CONSTRAINTS_VIOLATED means the pose is "
            "unreachable with the requested orientation, PLANNING_FAILED may "
            "succeed on one retry, and START_STATE_IN_COLLISION cannot be fixed "
            "by changing the target. Then pass plan_id to execute_openarm_plan."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    )
    async def plan_openarm_pose(
        arm: Annotated[Literal["right", "left"], Field(description="Which arm to move.")],
        x: Annotated[float, Field(description="Wrist target x (forward) in arm_origin, metres.")],
        y: Annotated[float, Field(description="Wrist target y (left) in arm_origin, metres.")],
        z: Annotated[float, Field(description="Wrist target z (up) in arm_origin, metres.")],
        roll_deg: Annotated[
            float,
            Field(
                ge=-180.0,
                le=180.0,
                description="Rotate the current wrist orientation about the arm_origin x axis, degrees.",
            ),
        ] = 0.0,
        pitch_deg: Annotated[
            float,
            Field(
                ge=-180.0,
                le=180.0,
                description="Rotate the current wrist orientation about the arm_origin y axis, degrees.",
            ),
        ] = 0.0,
        yaw_deg: Annotated[
            float,
            Field(
                ge=-180.0,
                le=180.0,
                description="Rotate the current wrist orientation about the arm_origin z axis, degrees.",
            ),
        ] = 0.0,
    ) -> MotionPlanSummary:
        """Plan with MoveIt and return a summary without the trajectory samples."""

        plan = await plan_pose(
            arm,
            x,
            y,
            z,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )
        points = plan["points"]
        return MotionPlanSummary(
            ok=plan["ok"],
            error=plan["error"],
            arm=plan["arm"],
            frame=plan["frame"],
            target_pose=plan["target_pose"],
            point_count=len(points),
            duration_sec=points[-1]["t"] if points else 0.0,
            planning_time_sec=plan["planning_time_sec"],
            plan_id=plan["plan_id"],
            plan_file=plan["plan_file"],
            created_at=plan["created_at"],
        )

    @mcp.tool(
        name="execute_openarm_plan",
        title="Execute OpenArm Motion (MoveIt)",
        description=(
            "Run a plan from plan_openarm_pose through MoveIt 2 "
            "(/execute_trajectory). Provide the plan_id. move_group refuses the "
            "plan if the arm moved since planning, so plan again for every move. "
            "Treat the move as done only when success is true. Does not command "
            "the gripper."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=False,
        ),
    )
    async def execute_openarm_plan(
        plan_id: Annotated[str, Field(description="Plan identifier returned by plan_openarm_pose.")],
    ) -> MotionExecutionResult:
        """Execute a stored MoveIt plan."""

        return MotionExecutionResult.model_validate(await execute_plan(plan_id))
