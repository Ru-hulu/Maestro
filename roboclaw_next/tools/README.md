# Tools

This package defines the first minimal tool boundary for RoboClaw Next.

The core idea is:

```text
AgentTool
    -> ToolRegistry
    -> LLM tool schema / tool execution
```

## Files

- `base.py`: defines `AgentTool`, `ToolExecutionContext`, and `ToolResult`.
- `registry.py`: registers tools and invokes them by name.
- `mcp_runtime.py`: connects to a stdio MCP server with the official Python MCP SDK.
- `mcp_adapter.py`: converts MCP tools into the same `AgentTool` shape.
- `mcp_server.py`: creates the MCP server and registers concrete tools.
- `builtin/`: groups each concrete Tool's executable program and MCP contract.

Current built-in capabilities:

- `mock_localization/`: simulated localization process and lifecycle Tools.
- `path_tracking/`: MPC path tracking process management Tools.
- `hybrid_astar_planner/`: one-shot runner and direct path-planning Tool backed by
  the standalone C++ Hybrid A* executable.
- `openarm_reach/`: OpenArm forward and inverse kinematics planning Tools.
- `openarm_motion/`: OpenArm arm-motion Tools backed by MoveIt 2.
  `plan_openarm_pose` plans a collision-checked motion of the wrist to x, y, z
  in `arm_origin`, keeping the current orientation. `execute_openarm_plan` runs
  the stored plan through `/execute_trajectory`. Requires the simulator
  launched with `moveit:=true`.
- `gazebo_realsense_camera/`: Gazebo RealSense RGB-D listener lifecycle Tools:
  `start_gazebo_realsense_camera`, `get_gazebo_realsense_camera_status`, and
  `stop_gazebo_realsense_camera`. The listener relays images to LCM channels
  `ROBOCLAW_REALSENSE_COLOR_IMAGE` and `ROBOCLAW_REALSENSE_DEPTH_IMAGE`.
- `sam3_segmentation/`: SAM3 perception service lifecycle Tools
  (`start_sam3_perception`, `get_sam3_perception_status`,
  `stop_sam3_perception`). SAM3 inference RPC clients belong to separate
  business Tools and are not part of this lifecycle manager.
- `target_object_position/`: business Tool `get_target_object_position` that
  calls the SAM3 LCM RPC interface with a natural-language prompt and returns a
  position-shaped result. The current implementation reports `position_valid`
  as false until depth projection is implemented.

## MCP Tool-Call Chain

The main end-to-end example is:

```bash
DEEPSEEK_API_KEY=... ROBOCLAW_LLM_PROVIDER=deepseek PYTHONPATH=. \
    uv run --no-project --with openai --with "mcp[cli]<2" \
    python roboclaw_next/examples/RuboclawClient.py
```

On the native ROS 2 Humble host workflow, run the client from a shell that has
already sourced `/opt/ros/humble/setup.bash` and `install/setup.bash`.
