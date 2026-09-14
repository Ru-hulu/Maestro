"""Load the in-package MoveIt configuration for the generated runtime URDF."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping and reject ambiguous non-mapping documents."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"MoveIt config must be a mapping: {config_path}")
    return payload


def move_group_parameters(
    robot_description_xml: str,
    config_dir: str | Path,
) -> list[dict[str, Any]]:
    """Return the complete parameter list consumed by ``move_group``."""

    moveit_dir = Path(config_dir) / "moveit"
    semantic_path = moveit_dir / "openarm2.srdf"
    semantic_xml = semantic_path.read_text(encoding="utf-8")

    return [
        {"robot_description": robot_description_xml},
        {"robot_description_semantic": semantic_xml},
        {
            "robot_description_kinematics": load_yaml(
                moveit_dir / "kinematics.yaml"
            )
        },
        {
            "robot_description_planning": load_yaml(
                moveit_dir / "joint_limits.yaml"
            )
        },
        {
            "planning_pipelines": ["ompl"],
            "default_planning_pipeline": "ompl",
            "ompl": load_yaml(moveit_dir / "ompl_planning.yaml"),
        },
        load_yaml(moveit_dir / "moveit_controllers.yaml"),
        load_yaml(moveit_dir / "move_group.yaml"),
        {"use_sim_time": True},
    ]
