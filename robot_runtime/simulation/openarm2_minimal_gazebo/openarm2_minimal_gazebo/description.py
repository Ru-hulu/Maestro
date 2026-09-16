"""Build the runtime URDF shared by Gazebo, robot_state_publisher, and MoveIt."""

import os
import xml.etree.ElementTree as ET

import xacro
from ament_index_python.packages import get_package_share_directory


def _add_text(parent: ET.Element, tag: str, text: str) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = text
    return child


def _add_origin(parent: ET.Element, xyz: str, rpy: str = "0 0 0") -> None:
    ET.SubElement(parent, "origin", {"xyz": xyz, "rpy": rpy})


def _add_mesh(parent: ET.Element, filename: str, scale: str = "1 1 1") -> None:
    geometry = ET.SubElement(parent, "geometry")
    ET.SubElement(geometry, "mesh", {"filename": filename, "scale": scale})


def _add_box(parent: ET.Element, size: str) -> None:
    geometry = ET.SubElement(parent, "geometry")
    ET.SubElement(geometry, "box", {"size": size})


def _add_cylinder(parent: ET.Element, radius: str, length: str) -> None:
    geometry = ET.SubElement(parent, "geometry")
    ET.SubElement(geometry, "cylinder", {"radius": radius, "length": length})


def _add_inertial(
    link: ET.Element,
    *,
    mass: str,
    xyz: str,
    ixx: str,
    ixy: str = "0.0",
    ixz: str = "0.0",
    iyy: str,
    iyz: str = "0.0",
    izz: str,
) -> None:
    inertial = ET.SubElement(link, "inertial")
    _add_origin(inertial, xyz)
    ET.SubElement(inertial, "mass", {"value": mass})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": ixx,
            "ixy": ixy,
            "ixz": ixz,
            "iyy": iyy,
            "iyz": iyz,
            "izz": izz,
        },
    )


def _add_mobile_base(root: ET.Element) -> None:
    for element in list(root):
        if element.tag == "link" and element.get("name") == "world":
            root.remove(element)
        if element.tag == "joint" and element.get("name") == "openarm_body_link0_mount_joint":
            root.remove(element)

    rosbot_share = get_package_share_directory("rosbot_description")
    mesh_root = os.path.join(rosbot_share, "meshes", "rosbot_xl")
    wheel_radius = 0.05

    base_footprint = ET.Element("link", {"name": "base_footprint"})
    root.insert(0, base_footprint)

    base_link = ET.Element("link", {"name": "base_link"})
    root.insert(1, base_link)

    base_joint = ET.Element("joint", {"name": "base_footprint_to_base_link", "type": "fixed"})
    _add_origin(base_joint, "0 0 0")
    ET.SubElement(base_joint, "parent", {"link": "base_footprint"})
    ET.SubElement(base_joint, "child", {"link": "base_link"})
    root.insert(2, base_joint)

    body_link = ET.Element("link", {"name": "body_link"})
    body_visual = ET.SubElement(body_link, "visual", {"name": "rosbot_xl_body_visual"})
    _add_mesh(body_visual, f"file://{os.path.join(mesh_root, 'body.dae')}")

    body_collision = ET.SubElement(body_link, "collision", {"name": "base_platform_collision"})
    _add_origin(body_collision, "0 0 -0.025")
    _add_box(body_collision, "0.36 0.32 0.05")
    _add_inertial(
        body_link,
        mass="3.5",
        xyz="0.0 0.0 0.0358",
        ixx="0.01393",
        ixy="-0.000020968097",
        ixz="0.000010399694",
        iyy="0.01081",
        iyz="0.000059372953",
        izz="0.02048",
    )
    root.insert(3, body_link)

    base_to_body_joint = ET.Element("joint", {"name": "base_to_body_joint", "type": "fixed"})
    _add_origin(base_to_body_joint, f"0.0 0.0 {wheel_radius}")
    ET.SubElement(base_to_body_joint, "parent", {"link": "base_link"})
    ET.SubElement(base_to_body_joint, "child", {"link": "body_link"})
    root.insert(4, base_to_body_joint)

    body_gazebo = ET.SubElement(root, "gazebo", {"reference": "body_link"})
    _add_text(body_gazebo, "gravity", "false")
    velocity_decay = ET.SubElement(body_gazebo, "velocity_decay")
    _add_text(velocity_decay, "linear", "0.2")
    _add_text(velocity_decay, "angular", "0.2")

    cover_link = ET.Element("link", {"name": "cover_link"})
    root.insert(5, cover_link)

    body_to_cover_joint = ET.Element("joint", {"name": "body_to_cover_joint", "type": "fixed"})
    _add_origin(body_to_cover_joint, "0.0 0.0 0.08345")
    ET.SubElement(body_to_cover_joint, "parent", {"link": "body_link"})
    ET.SubElement(body_to_cover_joint, "child", {"link": "cover_link"})
    root.insert(6, body_to_cover_joint)

    cover_gazebo = ET.SubElement(root, "gazebo", {"reference": "cover_link"})
    _add_text(cover_gazebo, "gravity", "false")

    mount_joint = ET.Element(
        "joint",
        {"name": "openarm_body_link0_mount_joint", "type": "fixed"},
    )
    _add_origin(mount_joint, "-0.02 0 0.00")
    ET.SubElement(mount_joint, "parent", {"link": "cover_link"})
    ET.SubElement(mount_joint, "child", {"link": "openarm_body_link0"})
    root.insert(7, mount_joint)

    wheels = (
        ("fl", "0.085 0.135 0.0", "mecanum_b.dae", "0 0 3.14159265359"),
        ("fr", "0.085 -0.135 0.0", "mecanum_a.dae", "0 0 3.14159265359"),
        ("rl", "-0.085 0.135 0.0", "mecanum_a.dae", "0 0 0"),
        ("rr", "-0.085 -0.135 0.0", "mecanum_b.dae", "0 0 0"),
    )
    for side, xyz, mesh_name, visual_rpy in wheels:
        link_name = f"{side}_wheel_link"
        joint_name = f"{side}_wheel_joint"
        wheel_link = ET.Element("link", {"name": link_name})
        wheel_visual = ET.SubElement(wheel_link, "visual", {"name": f"{link_name}_visual"})
        _add_origin(wheel_visual, "0 0 0", visual_rpy)
        _add_mesh(wheel_visual, f"file://{os.path.join(mesh_root, mesh_name)}")
        _add_inertial(
            wheel_link,
            mass="0.05",
            xyz="0 0 0",
            ixx="0.00005",
            iyy="0.00008",
            izz="0.00005",
        )
        root.insert(8, wheel_link)

        wheel_gazebo = ET.SubElement(root, "gazebo", {"reference": link_name})
        _add_text(wheel_gazebo, "gravity", "false")

        wheel_joint = ET.Element("joint", {"name": joint_name, "type": "fixed"})
        _add_origin(wheel_joint, xyz)
        ET.SubElement(wheel_joint, "parent", {"link": "body_link"})
        ET.SubElement(wheel_joint, "child", {"link": link_name})
        root.insert(8, wheel_joint)


OPTICAL_FRAME = "head_realsense_optical_frame"


def _add_head_realsense(root: ET.Element) -> None:
    camera_link_name = "head_realsense_link"

    camera_link = ET.Element("link", {"name": camera_link_name})
    camera_body = ET.SubElement(camera_link, "visual", {"name": "realsense_body_visual"})
    _add_origin(camera_body, "0 0 0")
    _add_box(camera_body, "0.025 0.09 0.025")
    body_material = ET.SubElement(camera_body, "material", {"name": "realsense_dark_gray"})
    ET.SubElement(body_material, "color", {"rgba": "0.08 0.085 0.09 1"})

    camera_lens = ET.SubElement(camera_link, "visual", {"name": "realsense_lens_visual"})
    _add_origin(camera_lens, "0.014 0 0", "0 1.57079632679 0")
    _add_cylinder(camera_lens, "0.008", "0.006")
    lens_material = ET.SubElement(camera_lens, "material", {"name": "realsense_lens_black"})
    ET.SubElement(lens_material, "color", {"rgba": "0.005 0.006 0.008 1"})

    camera_collision = ET.SubElement(camera_link, "collision", {"name": "realsense_collision"})
    _add_origin(camera_collision, "0 0 0")
    _add_box(camera_collision, "0.025 0.09 0.025")
    _add_inertial(
        camera_link,
        mass="0.08",
        xyz="0 0 0",
        ixx="0.00005",
        iyy="0.00002",
        izz="0.00005",
    )
    root.append(camera_link)

    camera_joint = ET.Element(
        "joint",
        {"name": "head_realsense_mount_joint", "type": "fixed"},
    )
    _add_origin(camera_joint, "0.12 0 0.82", "0 0.35 0")
    ET.SubElement(camera_joint, "parent", {"link": "base_footprint"})
    ET.SubElement(camera_joint, "child", {"link": camera_link_name})
    root.append(camera_joint)

    camera_gazebo = ET.SubElement(root, "gazebo", {"reference": camera_link_name})
    _add_text(camera_gazebo, "gravity", "false")
    sensor = ET.SubElement(
        camera_gazebo,
        "sensor",
        {"name": "head_realsense_camera", "type": "camera"},
    )
    _add_text(sensor, "always_on", "true")
    _add_text(sensor, "update_rate", "15")
    _add_text(sensor, "visualize", "true")
    camera = ET.SubElement(sensor, "camera")
    _add_text(camera, "horizontal_fov", "1.211")
    image = ET.SubElement(camera, "image")
    _add_text(image, "width", "640")
    _add_text(image, "height", "480")
    _add_text(image, "format", "R8G8B8")
    clip = ET.SubElement(camera, "clip")
    _add_text(clip, "near", "0.05")
    _add_text(clip, "far", "10.0")
    plugin = ET.SubElement(
        sensor,
        "plugin",
        {
            "name": "head_realsense_camera_controller",
            "filename": "libgazebo_ros_camera.so",
        },
    )
    ros = ET.SubElement(plugin, "ros")
    _add_text(ros, "namespace", "/head_realsense")
    _add_text(plugin, "camera_name", "color")
    _add_text(plugin, "frame_name", OPTICAL_FRAME)

    depth_sensor = ET.SubElement(
        camera_gazebo,
        "sensor",
        {"name": "head_realsense_depth", "type": "depth"},
    )
    _add_text(depth_sensor, "always_on", "true")
    _add_text(depth_sensor, "update_rate", "15")
    _add_text(depth_sensor, "visualize", "true")
    depth_camera = ET.SubElement(depth_sensor, "camera")
    _add_text(depth_camera, "horizontal_fov", "1.211")
    depth_image = ET.SubElement(depth_camera, "image")
    _add_text(depth_image, "width", "640")
    _add_text(depth_image, "height", "480")
    _add_text(depth_image, "format", "R_FLOAT32")
    depth_clip = ET.SubElement(depth_camera, "clip")
    _add_text(depth_clip, "near", "0.05")
    _add_text(depth_clip, "far", "10.0")
    depth_plugin = ET.SubElement(
        depth_sensor,
        "plugin",
        {
            "name": "head_realsense_depth_controller",
            "filename": "libgazebo_ros_camera.so",
        },
    )
    depth_ros = ET.SubElement(depth_plugin, "ros")
    _add_text(depth_ros, "namespace", "/head_realsense")
    _add_text(depth_plugin, "camera_name", "depth")
    _add_text(depth_plugin, "frame_name", OPTICAL_FRAME)
    _add_text(depth_plugin, "min_depth", "0.05")
    _add_text(depth_plugin, "max_depth", "10.0")


def _add_base_livox_mid360(root: ET.Element) -> None:
    lidar_link_name = "livox_mid360_link"

    lidar_link = ET.Element("link", {"name": lidar_link_name})
    body_visual = ET.SubElement(lidar_link, "visual", {"name": "livox_mid360_body_visual"})
    _add_origin(body_visual, "0 0 0")
    _add_cylinder(body_visual, "0.0325", "0.06")
    body_material = ET.SubElement(body_visual, "material", {"name": "livox_mid360_gray"})
    ET.SubElement(body_material, "color", {"rgba": "0.18 0.19 0.2 1"})

    top_visual = ET.SubElement(lidar_link, "visual", {"name": "livox_mid360_top_visual"})
    _add_origin(top_visual, "0 0 0.034")
    _add_cylinder(top_visual, "0.024", "0.012")
    top_material = ET.SubElement(top_visual, "material", {"name": "livox_mid360_black"})
    ET.SubElement(top_material, "color", {"rgba": "0.03 0.035 0.04 1"})

    lidar_collision = ET.SubElement(lidar_link, "collision", {"name": "livox_mid360_collision"})
    _add_origin(lidar_collision, "0 0 0")
    _add_cylinder(lidar_collision, "0.0325", "0.06")
    _add_inertial(
        lidar_link,
        mass="0.265",
        xyz="0 0 0",
        ixx="0.00010",
        iyy="0.00010",
        izz="0.00014",
    )
    root.append(lidar_link)

    lidar_joint = ET.Element(
        "joint",
        {"name": "livox_mid360_mount_joint", "type": "fixed"},
    )
    _add_origin(lidar_joint, "0.13 0 0.08")
    ET.SubElement(lidar_joint, "parent", {"link": "cover_link"})
    ET.SubElement(lidar_joint, "child", {"link": lidar_link_name})
    root.append(lidar_joint)

    lidar_gazebo = ET.SubElement(root, "gazebo", {"reference": lidar_link_name})
    _add_text(lidar_gazebo, "gravity", "false")
    sensor = ET.SubElement(
        lidar_gazebo,
        "sensor",
        {"name": "livox_mid360", "type": "ray"},
    )
    _add_text(sensor, "always_on", "true")
    _add_text(sensor, "update_rate", "10")
    _add_text(sensor, "visualize", "false")
    ray = ET.SubElement(sensor, "ray")
    scan = ET.SubElement(ray, "scan")
    horizontal = ET.SubElement(scan, "horizontal")
    _add_text(horizontal, "samples", "360")
    _add_text(horizontal, "resolution", "1")
    _add_text(horizontal, "min_angle", "-3.14159265359")
    _add_text(horizontal, "max_angle", "3.14159265359")
    vertical = ET.SubElement(scan, "vertical")
    _add_text(vertical, "samples", "40")
    _add_text(vertical, "resolution", "1")
    _add_text(vertical, "min_angle", "-0.12217304764")
    _add_text(vertical, "max_angle", "0.90757121104")
    range_element = ET.SubElement(ray, "range")
    _add_text(range_element, "min", "0.1")
    _add_text(range_element, "max", "40.0")
    _add_text(range_element, "resolution", "0.01")

    plugin = ET.SubElement(
        sensor,
        "plugin",
        {
            "name": "livox_mid360_controller",
            "filename": "libgazebo_ros_ray_sensor.so",
        },
    )
    ros = ET.SubElement(plugin, "ros")
    _add_text(ros, "namespace", "/livox")
    _add_text(ros, "remapping", "~/out:=lidar")
    _add_text(plugin, "output_type", "sensor_msgs/PointCloud2")
    _add_text(plugin, "frame_name", lidar_link_name)


def _replace_ros2_control_with_single_gazebo_system(root: ET.Element) -> None:
    joints: list[ET.Element] = []
    for ros2_control in root.findall("ros2_control"):
        for joint in ros2_control.findall("joint"):
            joints.append(ET.fromstring(ET.tostring(joint, encoding="unicode")))
        root.remove(ros2_control)

    ros2_control = ET.SubElement(
        root,
        "ros2_control",
        {"name": "gazebo_hardware_interface", "type": "system"},
    )
    hardware = ET.SubElement(ros2_control, "hardware")
    plugin = ET.SubElement(hardware, "plugin")
    plugin.text = "gazebo_ros2_control/GazeboSystem"

    existing_joint_names = set()
    for joint in joints:
        joint_name = joint.get("name")
        if joint_name is None or joint_name in existing_joint_names:
            continue
        existing_joint_names.add(joint_name)
        ros2_control.append(joint)


def build_robot_description(robot_preset: str, controllers_path: str) -> str:
    description_share = get_package_share_directory("openarm_description")
    xacro_path = os.path.join(
        description_share,
        "assets",
        "robot",
        "openarm_v2.0",
        "urdf",
        "openarm_v20.urdf.xacro",
    )
    xml_text = xacro.process_file(
        xacro_path,
        mappings={
            "robot_preset": robot_preset,
            "collapse_internal_empty_links": "true",
            "emit_grasp_frame": "false",
            "use_fake_hardware": "true",
        },
    ).toxml()
    xml_text = xml_text.replace(
        "package://openarm_description/",
        f"file://{description_share}/",
    )

    root = ET.fromstring(xml_text)
    _add_mobile_base(root)
    _add_base_livox_mid360(root)
    _add_head_realsense(root)
    _replace_ros2_control_with_single_gazebo_system(root)

    gazebo = ET.SubElement(root, "gazebo")
    plugin = ET.SubElement(
        gazebo,
        "plugin",
        {
            "name": "gazebo_ros2_control",
            "filename": "libgazebo_ros2_control.so",
        },
    )
    parameters = ET.SubElement(plugin, "parameters")
    parameters.text = controllers_path

    rigid_cmd_vel_plugin = ET.SubElement(
        gazebo,
        "plugin",
        {
            "name": "openarm2_rigid_cmd_vel",
            "filename": "libopenarm2_rigid_cmd_vel_plugin.so",
        },
    )
    _add_text(rigid_cmd_vel_plugin, "command_topic", "cmd_vel")
    _add_text(rigid_cmd_vel_plugin, "command_timeout", "0.5")

    return ET.tostring(root, encoding="unicode")

