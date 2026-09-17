import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from openarm2_minimal_gazebo.description import build_robot_description
from openarm2_minimal_gazebo.moveit_params import move_group_parameters


def _launch_setup(context):
    package_share = get_package_share_directory("openarm2_minimal_gazebo")
    controllers_path = os.path.join(package_share, "config", "openarm2_controllers.yaml")
    config_dir = os.path.join(package_share, "config")
    world_path = os.path.join(package_share, "worlds", "empty.world")

    robot_preset = context.perform_substitution(LaunchConfiguration("robot_preset"))
    gui = context.perform_substitution(LaunchConfiguration("gui")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    robot_description_xml = build_robot_description(robot_preset, controllers_path)
    robot_description = {"robot_description": robot_description_xml}

    gazebo_executable = "gazebo" if gui else "gzserver"
    gazebo_cmd = [
        gazebo_executable,
        "--verbose",
        world_path,
        "-s",
        "libgazebo_ros_init.so",
        "-s",
        "libgazebo_ros_factory.so",
    ]

    gazebo = ExecuteProcess(
        cmd=gazebo_cmd,
        additional_env={"GAZEBO_MODEL_DATABASE_URI": ""},
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )
    arm_origin_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="arm_origin_tf",
        output="screen",
        arguments=[
            "--x", "0",
            "--y", "0.031",
            "--z", "0",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "openarm_right_base_link",
            "--child-frame-id", "arm_origin",
        ],
        parameters=[{"use_sim_time": True}],
    )
    head_realsense_optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="head_realsense_optical_tf",
        output="screen",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", "0",
            "--roll", "-1.5707963267948966",
            "--pitch", "0",
            "--yaw", "-1.5707963267948966",
            "--frame-id", "head_realsense_link",
            "--child-frame-id", "head_realsense_optical_frame",
        ],
        parameters=[{"use_sim_time": True}],
    )
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description", "-entity", "openarm_v20"],
        output="screen",
    )

    controllers_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "openarm_joint_trajectory_controller",
            "--controller-manager",
            "/controller_manager",
            "--activate-as-group",
        ],
        output="screen",
    )
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=move_group_parameters(robot_description_xml, config_dir),
        condition=IfCondition(LaunchConfiguration("moveit")),
    )

    return [
        gazebo,
        robot_state_publisher,
        arm_origin_tf,
        head_realsense_optical_tf,
        spawn_entity,
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_entity,
                on_exit=[controllers_spawner],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=controllers_spawner,
                on_exit=[move_group],
            )
        ),
    ]


def generate_launch_description():
    robot_preset_arg = DeclareLaunchArgument(
        "robot_preset",
        default_value="default_bimanual",
        description="OpenArm 2.0 preset to load.",
    )
    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="true",
        description="Start gzclient in addition to gzserver.",
    )
    moveit_arg = DeclareLaunchArgument(
        "moveit",
        default_value="true",
        description="Start MoveIt move_group after the controllers are ready.",
    )

    return LaunchDescription(
        [
            robot_preset_arg,
            gui_arg,
            moveit_arg,
            OpaqueFunction(function=_launch_setup),
        ]
    )
