from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def launch_rpea_detector(context):
    # RPEA needs its own PyTorch build, so it runs under a separate interpreter
    # instead of as a regular ROS node from this package's environment.
    python = LaunchConfiguration('rpea_python').perform(context)
    checkpoint = LaunchConfiguration('checkpoint_path').perform(context)
    if not checkpoint:
        raise RuntimeError(
            "RPEA checkpoint not set: pass checkpoint_path:=/path/to/RPEA_JRDB2022.pth "
            "or export RPEA_CHECKPOINT"
        )

    return [ExecuteProcess(
        cmd=[
            python, '-m', 'alpaca_detection.RPEA_detection_node',
            '--ros-args',
            '-p', ['score_threshold:=', LaunchConfiguration('score_threshold')],
            '-p', f'checkpoint_path:={checkpoint}',
        ],
        output='screen',
    )]


def generate_launch_description():

    score_threshold_arg = DeclareLaunchArgument(
        'score_threshold',
        default_value='0.8',
        description='Score threshold for RPEA detections'
    )

    world_frame_arg = DeclareLaunchArgument (
        'world_frame',
        default_value='map',
        description='robot pose frame (map or odom )'
    )

    rpea_python_arg = DeclareLaunchArgument(
        'rpea_python',
        default_value=EnvironmentVariable('RPEA_PYTHON', default_value='python3'),
        description='Python interpreter with RPEA and its PyTorch installed (env: RPEA_PYTHON)'
    )

    checkpoint_path_arg = DeclareLaunchArgument(
        'checkpoint_path',
        default_value=EnvironmentVariable('RPEA_CHECKPOINT', default_value=''),
        description='Path to the RPEA checkpoint, e.g. RPEA_JRDB2022.pth (env: RPEA_CHECKPOINT)'
    )

    ld = LaunchDescription()
    ld.add_action(score_threshold_arg)
    ld.add_action(world_frame_arg)
    ld.add_action(rpea_python_arg)
    ld.add_action(checkpoint_path_arg)
    ld.add_action(OpaqueFunction(function=launch_rpea_detector))
    ld.add_action(
       Node (
            package='alpaca_detection',
            executable='SimpleTrack_node',
            name='SimpleTrack_node',
            output='screen',
            parameters=[{'use_sim_time': False}, {'world_frame': LaunchConfiguration('world_frame')}],
        )
    )
    return ld