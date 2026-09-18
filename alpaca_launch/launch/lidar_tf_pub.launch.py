#!/usr/bin/env python3
# Launch the odom->base_link TF broadcaster that reads /odom (nav_msgs/Odometry)

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    odom_topic = LaunchConfiguration('odom_topic')
    odom_frame = LaunchConfiguration('odom_frame')
    base_frame = LaunchConfiguration('base_frame')

    use_zed_odom_arg = DeclareLaunchArgument(
        'use_zed_odom', default_value='false', description='use odom from zed'
    )
    use_zed_odom = LaunchConfiguration('use_zed_odom')

    node = Node(
            package='alpaca_launch',
            executable='lidar_tf_pub',
            name='lidar_tf_pub',
            output='screen',
            parameters = [{'use_zed_odom' : use_zed_odom}]
        )

    return LaunchDescription([

        use_zed_odom_arg,
        node, 
    

    ])
