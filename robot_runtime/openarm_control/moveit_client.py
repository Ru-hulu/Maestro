"""Plan and execute OpenArm motions with MoveIt 2's move_group.

The tools only need two calls:

- plan_pose(): ask /plan_kinematic_path for a collision-free trajectory that
  brings the wrist to a pose given in arm_origin. MoveIt does the IK, the
  collision checking and the time parameterisation.
- execute(): send a stored trajectory to /execute_trajectory.

Each call opens a short-lived ROS node, like trajectory.py. ROS imports stay
inside the functions so this module loads on machines without ROS.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

from robot_runtime.openarm_ik import arm


PLAN_SERVICE = "/plan_kinematic_path"
EXECUTE_ACTION = "/execute_trajectory"
WAIT_SEC = 3.0  # how long to wait for move_group to show up
PLANNING_TIME_SEC = 5.0  # OMPL time budget for one plan
POSITION_TOLERANCE_M = 0.002
ORIENTATION_TOLERANCE_RAD = 0.01

MOVE_GROUP_MISSING = (
    "move_group is not running, so no collision-aware plan is possible. Start "
    "the simulation with moveit:=true."
)

# MoveItErrorCodes values reported to the model by name.
ERROR_NAMES = {
    1: "SUCCESS",
    -1: "PLANNING_FAILED",
    -2: "INVALID_MOTION_PLAN",
    -4: "CONTROL_FAILED",
    -6: "TIMED_OUT",
    -7: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION",
    -12: "GOAL_IN_COLLISION",
    -14: "GOAL_CONSTRAINTS_VIOLATED",
    -16: "INVALID_GOAL_CONSTRAINTS",
    -31: "NO_IK_SOLUTION",
}


def plan_pose(
    side: str,
    position: Sequence[float],
    quat_wxyz: Sequence[float],
    speed_scale: float,
) -> dict[str, object]:
    """Plan a collision-free wrist motion to a pose in arm_origin.

    The goal is handed to MoveIt in openarm_<side>_base_link: that frame is
    arm_origin shifted by base_from_origin with the same orientation, and it is
    always part of MoveIt's robot model. Planning starts from move_group's
    current robot state. Returns the MoveIt result name and, on success, the
    joint trajectory as plain lists that can be stored as JSON.
    """

    from moveit_msgs.msg import (
        Constraints,
        MotionPlanRequest,
        OrientationConstraint,
        PositionConstraint,
    )
    from moveit_msgs.srv import GetMotionPlan
    from geometry_msgs.msg import Pose
    from shape_msgs.msg import SolidPrimitive

    kinematics = arm(side)
    frame = f"openarm_{kinematics.side}_base_link"
    link = f"openarm_{kinematics.side}_ee_base_link"
    offset = kinematics.base_from_origin

    # Position goal: a small sphere around the target wrist position.
    center = Pose()
    center.position.x = float(position[0]) - offset[0]
    center.position.y = float(position[1]) - offset[1]
    center.position.z = float(position[2]) - offset[2]
    center.orientation.w = 1.0
    position_goal = PositionConstraint(link_name=link, weight=1.0)
    position_goal.header.frame_id = frame
    position_goal.constraint_region.primitives.append(
        SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[POSITION_TOLERANCE_M])
    )
    position_goal.constraint_region.primitive_poses.append(center)

    # Orientation goal: the requested wrist orientation with a small tolerance.
    orientation_goal = OrientationConstraint(
        link_name=link,
        weight=1.0,
        absolute_x_axis_tolerance=ORIENTATION_TOLERANCE_RAD,
        absolute_y_axis_tolerance=ORIENTATION_TOLERANCE_RAD,
        absolute_z_axis_tolerance=ORIENTATION_TOLERANCE_RAD,
    )
    orientation_goal.header.frame_id = frame
    w, x, y, z = (float(value) for value in quat_wxyz)
    orientation_goal.orientation.w = w
    orientation_goal.orientation.x = x
    orientation_goal.orientation.y = y
    orientation_goal.orientation.z = z

    request = MotionPlanRequest(
        group_name=f"{kinematics.side}_arm",
        pipeline_id="ompl",
        num_planning_attempts=1,
        allowed_planning_time=PLANNING_TIME_SEC,
        max_velocity_scaling_factor=float(speed_scale),
        max_acceleration_scaling_factor=float(speed_scale),
    )
    request.start_state.is_diff = True  # start from move_group's current state
    request.goal_constraints.append(
        Constraints(
            position_constraints=[position_goal],
            orientation_constraints=[orientation_goal],
        )
    )

    with _ros_node("openarm_moveit_planner") as (node, executor):
        client = node.create_client(GetMotionPlan, PLAN_SERVICE)
        if not client.wait_for_service(timeout_sec=WAIT_SEC):
            raise RuntimeError(MOVE_GROUP_MISSING)
        future = client.call_async(GetMotionPlan.Request(motion_plan_request=request))
        executor.spin_until_future_complete(future, timeout_sec=PLANNING_TIME_SEC + 10.0)
        if not future.done():
            raise RuntimeError("move_group did not answer the planning request in time.")
        response = future.result().motion_plan_response

    code = int(response.error_code.val)
    trajectory = response.trajectory.joint_trajectory
    return {
        "ok": code == 1,
        "error": ERROR_NAMES.get(code, f"CODE_{code}"),
        "planning_time_sec": float(response.planning_time),
        "joint_names": list(trajectory.joint_names),
        "points": [
            {
                "t": point.time_from_start.sec + point.time_from_start.nanosec * 1e-9,
                "positions": list(point.positions),
                "velocities": list(point.velocities),
                "accelerations": list(point.accelerations),
            }
            for point in trajectory.points
        ],
    }


def execute(
    joint_names: Sequence[str],
    points: Sequence[dict[str, object]],
    timeout_sec: float,
) -> dict[str, object]:
    """Run a stored trajectory through move_group's /execute_trajectory action.

    move_group refuses to start when the arm is no longer at the trajectory's
    first point, and stops the motion if it takes much longer than planned.
    Returns whether MoveIt reported success and the MoveIt result name.
    """

    from action_msgs.msg import GoalStatus
    from builtin_interfaces.msg import Duration
    from moveit_msgs.action import ExecuteTrajectory
    from rclpy.action import ActionClient
    from trajectory_msgs.msg import JointTrajectoryPoint

    goal = ExecuteTrajectory.Goal()
    goal.trajectory.joint_trajectory.joint_names = list(joint_names)
    for point in points:
        seconds = int(point["t"])
        goal.trajectory.joint_trajectory.points.append(
            JointTrajectoryPoint(
                positions=point["positions"],
                velocities=point["velocities"],
                accelerations=point["accelerations"],
                time_from_start=Duration(
                    sec=seconds,
                    nanosec=int((point["t"] - seconds) * 1e9),
                ),
            )
        )

    with _ros_node("openarm_moveit_executor") as (node, executor):
        client = ActionClient(node, ExecuteTrajectory, EXECUTE_ACTION)
        try:
            if not client.wait_for_server(timeout_sec=WAIT_SEC):
                raise RuntimeError(MOVE_GROUP_MISSING)
            sent = client.send_goal_async(goal)
            executor.spin_until_future_complete(sent, timeout_sec=WAIT_SEC)
            handle = sent.result() if sent.done() else None
            if handle is None or not handle.accepted:
                raise RuntimeError("move_group rejected the trajectory.")
            finished = handle.get_result_async()
            executor.spin_until_future_complete(finished, timeout_sec=timeout_sec)
            if not finished.done():
                executor.spin_until_future_complete(handle.cancel_goal_async(), timeout_sec=WAIT_SEC)
                raise RuntimeError(
                    f"Execution did not finish within {timeout_sec:.0f} s and was cancelled."
                )
            status = finished.result().status
            code = int(finished.result().result.error_code.val)
        finally:
            client.destroy()

    return {
        "success": status == GoalStatus.STATUS_SUCCEEDED and code == 1,
        "error": ERROR_NAMES.get(code, f"CODE_{code}"),
    }


@contextmanager
def _ros_node(name: str):
    """Create a throwaway node on its own ROS context and clean it up afterwards."""

    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    context = Context()
    rclpy.init(context=context, signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(name, context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    try:
        yield node, executor
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)
