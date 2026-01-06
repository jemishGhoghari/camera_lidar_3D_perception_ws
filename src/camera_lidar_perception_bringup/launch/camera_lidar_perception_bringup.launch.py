#!/usr/bin/env python3

from launch import LaunchDescription, LaunchContext
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def launch_setup(context: LaunchContext, *args, **kwargs):

    component_container = Node(
        package="rclcpp_components",
        executable="component_container_mt",
        namespace="",
        name="camera_lidar_perception_container",
        arguments=["--ros-args", "--log-level", "INFO"],
        output="screen",
    )

    model_file_path = PathJoinSubstitution(
        [
            FindPackageShare("yolo11_inference"),
            "weights",
            "yolo11m_1280x736.onnx",
        ]
    )

    launch_yolo11_detection_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("yolo11_inference"),
                    "launch",
                    "yolo11_inference_node.launch.py",
                ]
            ),
        ),
        launch_arguments={
            "component_container_name": "camera_lidar_perception_container",
            "standalone": "False",
            "image_input_topic": "/carla/ego_vehicle/zedx_one_gs/image",
            "detections_output_topic": "/detections",
            "network_image_width": "1280",
            "network_image_height": "736",
            "model_file_path": model_file_path,
            "confidence_threshold": "0.55",
            "nms_threshold": "0.45",
            "use_cuda": "True",
        }.items(),
    )

    launch_camera_lidar_3d_detection_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("camera_lidar_3d_perception"),
                    "launch",
                    "camera_lidar_perception_node.launch.py",
                ]
            ),
        ),
        launch_arguments={
            "component_container_name": "camera_lidar_perception_container",
            "standalone": "False",
            "pointcloud_topic": "/carla/ego_vehicle/livox_mid360",
            "camera_info_topic": "/carla/ego_vehicle/zedx_one_gs/camera_info",
            "detection_2d_input_topic": "detections",
            "detections_3d_output_topic": "/detections_3d",
            "voxel_leaf_size": "0.1",
            "cluster_tolerance": "0.2",
            "min_cluster_size": "1",
            "max_cluster_size": "25000",
            "camera_optical_frame": "ego_vehicle/zedx_one_gs",
            "target_frame": "map",
        }.items(),
    )

    launch_foxglove_bridge = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("foxglove_bridge"),
                    "launch",
                    "foxglove_bridge_launch.xml",
                ],
            ),
        ),
        condition=IfCondition(LaunchConfiguration("visualize")),
    )

    return [
        component_container,
        launch_yolo11_detection_node,
        launch_camera_lidar_3d_detection_node,
        launch_foxglove_bridge,
    ]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            "visualize",
            default_value="true",
            description="Run foxglove bridge to visualize with tf tree",
        ),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
