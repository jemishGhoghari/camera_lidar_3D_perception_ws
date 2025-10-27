from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, PushRosNamespace, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

def launch_setup(context, *args, **kwargs):

    component_container_name = LaunchConfiguration('component_container_name')
    standalone = LaunchConfiguration('standalone')
    image_input_topic = LaunchConfiguration('image_input_topic')
    detections_output_topic = LaunchConfiguration('detections_output_topic')
    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    model_file_path = LaunchConfiguration('model_file_path')
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    nms_threshold = LaunchConfiguration('nms_threshold')
    use_cuda = LaunchConfiguration('use_cuda')

    classes_config = PathJoinSubstitution(
        [FindPackageShare('yolo11_inference'), 'config', 'params.yaml']
    )

    component_container = ComposableNodeContainer(
        package='rclcpp_components',
        executable='component_container_mt',
        namespace='',
        name=component_container_name,
        arguments=['--ros-args', '--log-level', 'INFO'],
        output='screen',
        condition=IfCondition(standalone),
    )

    yolo11_inference_node = ComposableNode(
        package="yolo11_inference",
        plugin="yolo11_inference::Yolo11InferenceNode",
        name="yolo11_inference_node",
        parameters=[
            classes_config,
            {
                'model_file_path' : model_file_path,
                'network_image_width' : network_image_width,
                'network_image_height' : network_image_height,
                'confidence_threshold' : confidence_threshold,
                'nms_threshold' : nms_threshold,
                'use_cuda' : use_cuda,
            }
        ],
        remappings=[
            ('image_raw', image_input_topic),
            ('detections', detections_output_topic)
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    load_composable_nodes_action = GroupAction(
        [
            PushRosNamespace('yolo11_inference'),
            LoadComposableNodes(
                composable_node_descriptions=[
                    yolo11_inference_node
                ],
                target_container=component_container_name,
            ),
        ]
    )

    return [component_container, load_composable_nodes_action]

def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'component_container_name',
            default_value='yolov11_inference_container',
            description='The name of the component container',
        ),
        DeclareLaunchArgument(
            'standalone',
            default_value='True',
            description='Whether the component container should be launched or not',
        ),
        DeclareLaunchArgument(
            'image_input_topic',
            default_value='image_raw',
            description='The topic to subscribe to for the input image',
        ),
        DeclareLaunchArgument(
            'detections_output_topic',
            default_value='detections',
            description='The topic to publish the detections to',
        ),
        DeclareLaunchArgument(
            'network_image_width',
            default_value='640',
            description='The width of the network image',
        ),
        DeclareLaunchArgument(
            'network_image_height',
            default_value='640',
            description='The height of the network image',
        ),
        DeclareLaunchArgument(
            'model_file_path',
            description='The absolute file path to the ONNX file',
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.25',
            description='The confidence threshold for object detection',
        ),
        DeclareLaunchArgument(
            'nms_threshold',
            default_value='0.4',
            description='The non-maximum suppression threshold for object detection',
        ),
        DeclareLaunchArgument(
            'use_cuda',
            default_value='True',
            description='The flag to set whether to use CUDA based DNN or not',
        ),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])