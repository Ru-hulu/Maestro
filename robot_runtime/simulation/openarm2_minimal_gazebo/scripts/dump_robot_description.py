#!/usr/bin/env python3
"""Export and inspect the runtime OpenArm URDF used by the simulator."""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from openarm2_minimal_gazebo.description import build_robot_description


def _parse_args() -> argparse.Namespace:
    package_share = Path(get_package_share_directory("openarm2_minimal_gazebo"))
    parser = argparse.ArgumentParser(
        description="Write the generated runtime URDF and print its MoveIt-relevant structure."
    )
    parser.add_argument(
        "--preset",
        default="default_bimanual",
        help="OpenArm robot preset passed to the upstream xacro.",
    )
    parser.add_argument(
        "--controllers",
        type=Path,
        default=package_share / "config" / "openarm2_controllers.yaml",
        help="Controller YAML path embedded in the Gazebo ros2_control plugin.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/openarm2_mutated.urdf"),
        help="Destination for the generated URDF.",
    )
    return parser.parse_args()


def _joint_summary(joint: ET.Element) -> str:
    fields = [
        f"name={joint.get('name', '<unnamed>')}",
        f"type={joint.get('type', '<missing>')}",
    ]
    limit = joint.find("limit")
    if limit is not None:
        for key in ("lower", "upper", "effort", "velocity"):
            fields.append(f"{key}={limit.get(key, '<missing>')}")
    else:
        fields.append("limit=<missing>")

    mimic = joint.find("mimic")
    if mimic is not None:
        fields.append(f"mimic={mimic.get('joint', '<missing>')}")
        fields.append(f"multiplier={mimic.get('multiplier', '1')}")
        fields.append(f"offset={mimic.get('offset', '0')}")
    else:
        fields.append("mimic=None")
    return " ".join(fields)


def _find_link_chain(root: ET.Element, start: str, goal: str) -> tuple[str, ...] | None:
    children: dict[str, list[str]] = defaultdict(list)
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.get("link")
        child_link = child.get("link")
        if parent_link and child_link:
            children[parent_link].append(child_link)

    pending: list[tuple[str, tuple[str, ...]]] = [(start, (start,))]
    visited: set[str] = set()
    while pending:
        link, path = pending.pop()
        if link == goal:
            return path
        if link in visited:
            continue
        visited.add(link)
        for child in reversed(children.get(link, [])):
            pending.append((child, (*path, child)))
    return None


def _print_report(root: ET.Element, output: Path) -> bool:
    print(f"output: {output}")
    print(f"robot name: {root.get('name', '<missing>')}")
    print("non-fixed joints:")
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            print(f"  {_joint_summary(joint)}")

    chains_ok = True
    print("arm chains:")
    for side in ("right", "left"):
        start = f"openarm_{side}_base_link"
        goal = f"openarm_{side}_ee_base_link"
        chain = _find_link_chain(root, start, goal)
        if chain is None:
            chains_ok = False
            print(f"  {side}: MISSING ({start} -> {goal})")
        else:
            print(f"  {side}: OK ({' -> '.join(chain)})")
    return chains_ok


def main() -> int:
    args = _parse_args()
    xml_text = build_robot_description(args.preset, str(args.controllers))
    root = ET.fromstring(xml_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml_text, encoding="utf-8")
    return 0 if _print_report(root, args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
