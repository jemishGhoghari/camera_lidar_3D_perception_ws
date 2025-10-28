from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent,
                            RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.substitutions import FindPackageShare
import lifecycle_msgs.msg

def generate_launch_description():
    nav2_map_server_node = LifecycleNode(
        package='nav2_map_server',
        executable='map_server',
        namespace='nav2',
        name='map_server',
        parameters=[
            {'yaml_filename': LaunchConfiguration('yaml_filename')},
        ],
    )

    configure_nav2_map_server_node = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(nav2_map_server_node),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    activate_nav2_map_server_node = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(nav2_map_server_node),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
        )
    )

    configure_nav2_map_server_node_event_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=nav2_map_server_node,
            on_start=[configure_nav2_map_server_node],
        )
    )

    activate_nav2_map_server_node_event_handler = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=nav2_map_server_node,
            goal_state='inactive',
            entities=[activate_nav2_map_server_node],
        ),
        condition=IfCondition(LaunchConfiguration('auto_activate')),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('auto_activate', default_value='false'),
            DeclareLaunchArgument('yaml_filename', default_value=''),
            nav2_map_server_node,
            configure_nav2_map_server_node_event_handler,
            activate_nav2_map_server_node_event_handler,
        ]
    )