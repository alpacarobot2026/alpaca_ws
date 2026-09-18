from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, SetRemap
import os


def generate_launch_description():
    map_arg = DeclareLaunchArgument(
        'map', default_value='/path/to/map', description='Path to the map yaml/pgm'
    )

    safety_arg = DeclareLaunchArgument(
        'safety', default_value='rc', description='Enable safety deadman switch'
    )

    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value='/home/user/alpaca_ws/src/nav2_params/navigation_params.yaml',
        description='Full path to the nav2 params file',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='False', description='Use simulation time'
    )

    mppi_pkg_arg = DeclareLaunchArgument(
        'mppi_pkg', default_value='alpaca_navigation', description='Package containing mppi controller node'
    )

    plan_pkg_arg = DeclareLaunchArgument(
        'plan_pkg', default_value='alpaca_navigation', description='Package containing plan segmenter node'
    )

    namespace = 'ranger_mini'

    mppi_node = Node(
        package=LaunchConfiguration('mppi_pkg'),
        executable='mppi_controller_node',
        name='mppi_controller_node',
        namespace=namespace,
        output='screen',
        arguments=['--safety', LaunchConfiguration('safety')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        remappings=[
            ('/tf',        f'/{namespace}/tf'),
            ('/tf_static', f'/{namespace}/tf_static'),
        ],
    )

    plan_segmenter_node = Node(
        package=LaunchConfiguration('plan_pkg'),
        executable='plan_segmenter_node',
        name='plan_segmenter_node',
        namespace=namespace,
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time'),
             'segment_length': 10.0},
        ],
    )

    # Keep nav2 off cmd_vel; the MPPI controller (via twist_mux) owns that topic.
    # IncludeLaunchDescription takes no remappings, so the rules are set on the
    # group, which applies them to every nav2 node and overrides nav2's own.
    # Names are relative so they expand inside `namespace` above; an absolute
    # /cmd_vel rule would never match /ranger_mini/cmd_vel.
    nav2_bringup_launch = GroupAction([
        SetRemap(src='cmd_vel', dst='cmd_vel_nav'),
        # velocity_smoother publishes under the original name cmd_vel_smoothed,
        # which stock nav2 remaps onto cmd_vel, so it needs a rule of its own.
        SetRemap(src='cmd_vel_smoothed', dst='cmd_vel_nav_smoothed'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'namespace':    namespace,
                'use_namespace': 'True',
                'map':          LaunchConfiguration('map'),
                'params_file':  LaunchConfiguration('params_file'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }.items(),
        ),
    ])

    ld = LaunchDescription()
    ld.add_action(map_arg)
    ld.add_action(params_arg)
    ld.add_action(use_sim_time_arg)
    ld.add_action(safety_arg)
    ld.add_action(mppi_pkg_arg)
    ld.add_action(plan_pkg_arg)

    ld.add_action(mppi_node)
    ld.add_action(plan_segmenter_node)
    ld.add_action(nav2_bringup_launch)

    return ld
