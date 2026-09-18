#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource, FrontendLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from nav2_common.launch import ReplaceString




def generate_launch_description():
    use_lidar = LaunchConfiguration('use_lidar', default='true')
    use_camera = LaunchConfiguration('use_camera', default='true')
    use_zed_odom = LaunchConfiguration('use_zed_odom', default='false')
    use_imu_ekf = LaunchConfiguration('use_imu_ekf', default='false')

    # use_zed_odom = LaunchConfiguration('use_zed_odom')

    base_bringup_launch = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            [os.path.join(get_package_prefix("ranger_bringup"),"share", "ranger_bringup","launch","ranger_mini_v3.launch.xml")],
        )
    )
    
    def camera_bringup(context, *args, **kwargs):
        # zed_wrapper isn't installed in the docker image, so get_package_prefix()
        # raises immediately if resolved eagerly. Deferring it to an OpaqueFunction
        # means it only runs when use_camera actually resolves to true.
        if LaunchConfiguration('use_camera').perform(context).lower() not in ('true', '1'):
            return []
        return [IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(get_package_prefix("zed_wrapper"), "share", "zed_wrapper", "launch", "zed_camera.launch.py")],
            ),
            launch_arguments={'camera_model': 'zedm', 'publish_tf': use_zed_odom}.items(),
        )]

    camera_bringup_launch = OpaqueFunction(function=camera_bringup)
    
    
    # The stock velodyne_driver_node-VLP16-launch.py takes no launch arguments,
    # so device_ip can't be overridden through IncludeLaunchDescription. The
    # sensor lives on 192.168.3.x, but the apt package's default params yaml
    # (and the upstream repo before the host's local, uncommitted patch) ships
    # device_ip: 192.168.1.201, which makes the driver silently drop every
    # real packet. Build the node directly so the override is in-repo instead
    # of depending on that host-only edit.
    velodyne_device_ip = LaunchConfiguration('velodyne_device_ip', default='192.168.3.201')

    lidar_bringup_launch = Node(
        package='velodyne_driver',
        executable='velodyne_driver_node',
        output='both',
        parameters=[
            os.path.join(get_package_share_directory('velodyne_driver'), 'config', 'VLP16-velodyne_driver_node-params.yaml'),
            {'device_ip': velodyne_device_ip},
        ],
        condition=IfCondition(use_lidar),
    )

    # The stock velodyne_transform_node-VLP16-launch.py takes no launch
    # arguments either (same limitation as lidar_bringup_launch above), and
    # its default organize_cloud:=true builds a structured 2D grid every
    # scan - expensive enough, with RPEA/HST/MPPI/TalkNCE all competing for
    # CPU in this container, that /velodyne_points drops from ~10 Hz (the
    # scan rate, see /velodyne_packets) to ~1 Hz. organize_cloud:=false skips
    # that structured layout and publishes an unordered cloud instead.
    velodyne_organize_cloud = LaunchConfiguration('velodyne_organize_cloud', default='false')
    velodyne_pointcloud_share = get_package_share_directory('velodyne_pointcloud')

    lidar_pointcloud_bringup_launch = Node(
        package='velodyne_pointcloud',
        executable='velodyne_transform_node',
        output='both',
        parameters=[
            os.path.join(velodyne_pointcloud_share, 'config', 'VLP16-velodyne_transform_node-params.yaml'),
            {
                'calibration': os.path.join(velodyne_pointcloud_share, 'params', 'VLP16db.yaml'),
                'organize_cloud': velodyne_organize_cloud,
            },
        ],
        condition=IfCondition(use_lidar),
    )

    # Disabled when use_imu_ekf=true — EKF node publishes odom->base_link TF itself
    odom_2_tf_launch = IncludeLaunchDescription (
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_prefix ("alpaca_launch"), "share", "alpaca_launch", "launch", "odom_2_tf_launch.py")],
        ),
        condition=UnlessCondition(use_imu_ekf),
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[os.path.join(get_package_share_directory('nav2_params'), 'navigation_params.yaml')],
        condition=IfCondition(use_imu_ekf),
    )

    lidar_tf_pub = IncludeLaunchDescription (
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_prefix ("alpaca_launch"), "share", "alpaca_launch", "launch", "lidar_tf_pub.launch.py")],
            
        ),
        launch_arguments = {'use_zed_odom' : use_zed_odom}.items(),
    )

    point_2_laserscan_node = Node (
        package = 'pointcloud_to_laserscan',
        executable = 'pointcloud_to_laserscan_node',
        name = 'pointcloud_to_laserscan_node',
        output = 'screen',
        parameters = [{'target_frame': 'velodyne',
                       'transform_tolerance': 0.01,
                       'min_height': -0.3,
                       'max_height': 0.5,
                       'angle_min': -3.14,
                       'angle_max': 3.14,
                       'angle_increment': 0.0058,
                       'scan_time': 0.1,
                       'range_min': 0.0,
                       'range_max': 20.0,
                       'use_inf': True,
                       'inf_epsilon': 0.1,
                       'output_topic': '/scan',
                       'qos_overrides./scan.publisher.reliability': 'RELIABLE'
                       },
                       ],
        remappings = [('cloud_in', '/velodyne_points')],
        condition = IfCondition(use_lidar),
    )



    pointcloud_filter_node = Node(
        package="alpaca_launch",
        executable="pointcloud_filter",
        name="pointcloud_filter",
        parameters=[{
            "input_topic": "/velodyne_points",
            "output_topic": "/filtered_clouds",
        }],
        )


    # Removed: relayed /velodyne_points to /cloud_in, which nothing
    # subscribed to. Its RELIABLE subscription to /velodyne_points made
    # velodyne_transform_node's writer block waiting for its ACK whenever
    # topic_tools relay lagged, stalling the topic for every other
    # subscriber too.

    ld = LaunchDescription()

    ld.add_action (DeclareLaunchArgument(
        'use_imu_ekf',
        default_value = 'false',
        description = 'Fuse wheel odom + ZED Mini IMU via EKF. Disables odom_to_tf (EKF publishes TF instead).'))

    ld.add_action (DeclareLaunchArgument(
        'use_lidar',
        default_value = 'true',))

    ld.add_action (DeclareLaunchArgument(
        'use_camera',
        default_value = 'true',))

    ld.add_action (DeclareLaunchArgument(
        'use_zed_odom',
        default_value = 'false',))

    ld.add_action (DeclareLaunchArgument(
        'velodyne_device_ip',
        default_value = '192.168.3.201',
        description = 'Source IP the velodyne driver accepts packets from.'))

    ld.add_action (DeclareLaunchArgument(
        'velodyne_organize_cloud',
        default_value = 'false',
        description = 'Structured (organized) vs. unordered /velodyne_points. '
                       'true is heavier and can drop the publish rate under CPU contention.'))

    ld.add_action (base_bringup_launch)
    ld.add_action (camera_bringup_launch)
    ld.add_action (lidar_bringup_launch)
    ld.add_action (odom_2_tf_launch)
    ld.add_action (ekf_node)
    ld.add_action (lidar_tf_pub)
    ld.add_action (point_2_laserscan_node)
    ld.add_action (pointcloud_filter_node)
    ld.add_action (lidar_pointcloud_bringup_launch)
    # ld.add_action (dummy_tf_publisher)
    # ld.add_action (point_2_laserscan)   
    

    return ld
