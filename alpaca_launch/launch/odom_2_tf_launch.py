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

    return LaunchDescription([
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odom',
            description='Odometry topic (nav_msgs/Odometry) to convert to TF.'
        ),
        DeclareLaunchArgument(
            'odom_frame',
            default_value='',
            description='Parent frame (e.g., "odom"). Leave empty to use msg.header.frame_id.'
        ),
        DeclareLaunchArgument(
            'base_frame',
            default_value='base_link',
            description='Child frame (e.g., "base_link") if msg.child_frame_id is empty.'
        ),

        Node(
            package='alpaca_launch',
            executable='odom_to_tf',     # entry point from setup.py
            name='odom_to_tf',
            output='screen',
            parameters=[{
                'odom_topic': odom_topic,
                'odom_frame': odom_frame,
                'base_frame': base_frame,
            }]
        ),
    ])
