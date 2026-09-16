"""MCP contracts for MoveIt-backed OpenArm motion planning and execution."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .program import (
    RelativeFrame,
    RelativeOrientationMode,
    execute_plan,
    plan_relative,
)


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
        description="Resolved wrist target position or pose in arm_origin.",
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
        title="Plan Relative OpenArm EE Motion (MoveIt)",
        description=(
            "Plan a relative wrist motion from the latest measured pose with MoveIt. "
            "dx, dy, dz are metres in arm_origin (fixed) or tool (wrist-local). "
            "keep preserves orientation; relative applies the axis-angle rotation_vector; "
            "tolerant relaxes that orientation; free omits it. The trajectory is stored, "
            "not executed; pass a successful plan_id to execute_openarm_plan."
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
        dx: Annotated[float, Field(description="Relative x translation in metres.")] = 0.0,
        dy: Annotated[float, Field(description="Relative y translation in metres.")] = 0.0,
        dz: Annotated[float, Field(description="Relative z translation in metres.")] = 0.0,
        translation_frame: Annotated[
            RelativeFrame,
            Field(description="arm_origin for fixed axes; tool for wrist-local axes."),
        ] = "arm_origin",
        orientation_mode: Annotated[
            RelativeOrientationMode,
            Field(description="keep, relative, tolerant, or free."),
        ] = "keep",
        rotation_vector: Annotated[
            list[float] | None,
            Field(
                min_length=3,
                max_length=3,
                description="Axis times angle in radians; used by relative or tolerant.",
            ),
        ] = None,
        rotation_frame: Annotated[
            RelativeFrame,
            Field(description="Frame of rotation_vector."),
        ] = "tool",
        orientation_tolerance_rad: Annotated[
            float,
            Field(gt=0.0, le=math.pi, description="Tolerance for tolerant mode."),
        ] = 0.20,
    ) -> MotionPlanSummary:
        """Resolve a relative EE command and store its MoveIt plan."""

        plan = await plan_relative(
            arm,
            (dx, dy, dz),
            translation_frame=translation_frame,
            orientation_mode=orientation_mode,
            rotation_vector=rotation_vector,
            rotation_frame=rotation_frame,
            orientation_tolerance_rad=orientation_tolerance_rad,
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
