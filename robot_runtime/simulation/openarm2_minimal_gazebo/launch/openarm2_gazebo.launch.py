import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from openarm2_minimal_gazebo.description import build_robot_description


def _launch_setup(context):
    package_share = get_package_share_directory("openarm2_minimal_gazebo")
    controllers_path = os.path.join(package_share, "config", "openarm2_controllers.yaml")
    world_path = os.path.join(package_share, "worlds", "empty.world")

    robot_preset = context.perform_substitution(LaunchConfiguration("robot_preset"))
    gui = context.perform_substitution(LaunchConfiguration("gui")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    robot_description = {
        "robot_description": build_robot_description(robot_preset, controllers_path)
    }

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
        parameters=[robot_description],
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

    return [
        gazebo,
        robot_state_publisher,
        spawn_entity,
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_entity,
                on_exit=[controllers_spawner],
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

    return LaunchDescription(
        [
            robot_preset_arg,
            gui_arg,
            OpaqueFunction(function=_launch_setup),
        ]
    )
