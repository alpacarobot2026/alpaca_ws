from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import GroupAction

def generate_launch_description():
    return LaunchDescription([
        GroupAction([
            PushRosNamespace('ranger_mini'),
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                parameters=[
                    '/home/user/alpaca_ws/src/nav2_params/mapper_params_online_async.yaml'
                ],
            )
        ])
    ])
