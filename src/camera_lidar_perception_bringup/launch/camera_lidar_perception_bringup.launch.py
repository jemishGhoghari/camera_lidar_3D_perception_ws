#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def launch_setup(context):
    component_container = Node(
        package='rclcpp_components',
        executable='component_container_mt',
        namespace='',
        name='camera_lidar_perception_container',
        arguments=['--ros-args', '--log-level', 'INFO'],
        output='screen',
    )

    model_file_path = PathJoinSubstitution(
        [
            FindPackageShare('yolo11_inference'),
            'weights',
            'yolo11m_1280x736.onnx',
        ]
    )

    launch_yolo11_detection_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('yolo11_inference'),
                    'launch',
                    'yolo11_inference_node.launch.py',
                ]
            ),
        ),
        launch_arguments={
            'component_container_name': 'camera_lidar_perception_container',
            'standalone': 'False',
            'image_input_topic': '/zed/zed_node/rgb/image_rect_color/compressed',
            'detections_output_topic': '/detections',
            'network_image_width': '1280',
            'network_image_height': '736',
            'model_file_path': model_file_path,
            'confidence_threshold': '0.25',
            'nms_threshold': '0.4',
            'use_cuda': 'True',
        }.items()
    )

    launch_camera_lidar_3d_detection_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('camera_lidar_3d_perception'),
                    'launch',
                    'camera_lidar_perception_node.launch.py',
                ]
            ),
        ),
        launch_arguments={
            'component_container_name': 'camera_lidar_perception_container',
            'standalone': 'False',
            'pointcloud_topic': '/livox/lidar',
            'camera_info_topic': '/zed/zed_node/rgb/camera_info',
            'detection_2d_input_topic': 'detections',
            'detections_3d_output_topic': '/detections_3d',
        }.items()
    )

    launch_foxglove_bridge = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('foxglove_bridge'), 'launch', 'foxglove_bridge_launch.xml'],
            ),
        ),
        condition=IfCondition(LaunchConfiguration('preview')),
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
            'preview',
            default_value='false',
            description='Run foxglove bridge and robot_state_publisher to visulaize with tf tree',
        ),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])