from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import (
    ComposableNodeContainer,
    PushRosNamespace,
    LoadComposableNodes,
)
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):

    component_container_name = LaunchConfiguration("component_container_name")
    standalone = LaunchConfiguration("standalone")
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    image_input_topic = LaunchConfiguration("detection_2d_input_topic")
    detections_output_topic = LaunchConfiguration("detections_3d_output_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    voxel_leaf_size = LaunchConfiguration("voxel_leaf_size")
    cluster_tolerance = LaunchConfiguration("cluster_tolerance")
    min_cluster_size = LaunchConfiguration("min_cluster_size")
    max_cluster_size = LaunchConfiguration("max_cluster_size")
    camera_optical_frame = LaunchConfiguration("camera_optical_frame")
    target_frame = LaunchConfiguration("target_frame")

    component_container = ComposableNodeContainer(
        package="rclcpp_components",
        executable="component_container_mt",
        namespace="",
        name=component_container_name,
        arguments=["--ros-args", "--log-level", "INFO"],
        output="screen",
        condition=IfCondition(standalone),
    )

    camera_lidar_detection_node = ComposableNode(
        package="camera_lidar_3d_perception",
        plugin="camera_lidar_3d_perception::CameraLidarPerceptionNode",
        name="camera_lidar_3d_perception_node",
        parameters=[
            {
                "voxel_leaf_size": voxel_leaf_size,
                "cluster_tolerance": cluster_tolerance,
                "min_cluster_size": min_cluster_size,
                "max_cluster_size": max_cluster_size,
                "camera_optical_frame": camera_optical_frame,
                "target_frame": target_frame,
            }
        ],
        remappings=[
            ("pointcloud", pointcloud_topic),
            ("detections", image_input_topic),
            ("camera_info", camera_info_topic),
            ("detections_3d", detections_output_topic),
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    load_composable_nodes_action = GroupAction(
        [
            PushRosNamespace(""),
            LoadComposableNodes(
                composable_node_descriptions=[camera_lidar_detection_node],
                target_container=component_container_name,
            ),
        ]
    )

    return [component_container, load_composable_nodes_action]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            "component_container_name",
            default_value="camera_lidar_perception_container",
            description="The name of the component container",
        ),
        DeclareLaunchArgument(
            "standalone",
            default_value="True",
            description="Whether the component container should be launched or not",
        ),
        DeclareLaunchArgument(
            "pointcloud_topic",
            default_value="pointcloud",
            description="The topic to subscribe to for the LiDAR Pointcloud",
        ),
        DeclareLaunchArgument(
            "detection_2d_input_topic",
            default_value="detections",
            description="The topic to subscribe to for the 2D Detections",
        ),
        DeclareLaunchArgument(
            "detections_3d_output_topic",
            default_value="detections_3d",
            description="the topic to publish the 3D Detections",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="camera_info",
            description="The topic to subscribe to for the Camera Info",
        ),
        DeclareLaunchArgument(
            "voxel_leaf_size",
            default_value="0.1",
            description="The size of the voxel grid leaf for downsampling the pointcloud",
        ),
        DeclareLaunchArgument(
            "cluster_tolerance",
            default_value="0.1",
            description="The tolerance for clustering points in the pointcloud",
        ),
        DeclareLaunchArgument(
            "min_cluster_size",
            default_value="10",
            description="The minimum size of a cluster to be considered valid",
        ),
        DeclareLaunchArgument(
            "max_cluster_size",
            default_value="25000",
            description="The maximum size of a cluster to be considered valid",
        ),
        DeclareLaunchArgument(
            "camera_optical_frame",
            default_value="zed_left_camera_optical_frame",
            description="The camera optical frame",
        ),
        DeclareLaunchArgument(
            "target_frame",
            default_value="map",
            description="The Target frame in which 3D detections trandformed to",
        ),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
