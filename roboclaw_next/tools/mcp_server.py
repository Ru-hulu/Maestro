"""MCP server that registers and exposes RoboClaw tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .builtin.gazebo_realsense_camera.tool import register_gazebo_realsense_camera_tool
from .builtin.hybrid_astar_planner.tool import register_hybrid_astar_planner_tool
from .builtin.mock_localization.program import MockLocalizationProcessManager
from .builtin.mock_localization.tool import register_mock_localization_tools
from .builtin.openarm_motion.tool import register_openarm_motion_tools
from .builtin.openarm_reach.tool import register_openarm_reach_tools
from .builtin.path_tracking.program import PathTrackingProcessManager
from .builtin.path_tracking.tool import register_path_tracking_tools
from .builtin.sam3_segmentation.program import Sam3PerceptionManager
from .builtin.sam3_segmentation.tool import register_sam3_segmentation_tools
from .builtin.target_object_position.tool import register_target_object_position_tool


mcp = FastMCP("RoboClaw Tool Server", json_response=True)

localization_manager = MockLocalizationProcessManager()
tracking_manager = PathTrackingProcessManager(localization_manager)
sam3_manager = Sam3PerceptionManager()

register_mock_localization_tools(mcp, localization_manager)
register_path_tracking_tools(mcp, tracking_manager)
register_hybrid_astar_planner_tool(mcp)
register_openarm_reach_tools(mcp)
register_openarm_motion_tools(mcp)
register_sam3_segmentation_tools(mcp, sam3_manager)
register_gazebo_realsense_camera_tool(mcp)
register_target_object_position_tool(mcp)


if __name__ == "__main__":
    mcp.run(transport="stdio")
