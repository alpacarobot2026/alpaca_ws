from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    models_dir = LaunchConfiguration('models_dir')

    ld = LaunchDescription()
    ld.add_action(
        DeclareLaunchArgument(
            'prediction_backend',
            default_value='hst',
            description='Prediction backend to use: hst, cv, eqmotion, autobots, or moflow',
        )
    )
    # The prediction stack needs TensorFlow, which cannot share the controller's
    # environment, so it runs under its own interpreter.
    ld.add_action(
        DeclareLaunchArgument(
            'hst_python',
            default_value=EnvironmentVariable('HST_PYTHON', default_value='python3'),
            description='Python interpreter with TensorFlow and the prediction deps (env: HST_PYTHON)',
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            'models_dir',
            default_value=EnvironmentVariable('ALPACA_MODELS_DIR', default_value='/models'),
            description='Directory holding prediction checkpoints and configs (env: ALPACA_MODELS_DIR)',
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            'eqmotion_bundle_path',
            default_value=models_dir,
            description='Path to the EqMotion checkpoint bundle or zip file',
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            'autobots_checkpoint_path',
            default_value=PathJoinSubstitution([models_dir, 'best_models_ade_autobots_da.pth']),
            description='Path to the AutoBots checkpoint file',
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            'autobots_config_path',
            default_value=PathJoinSubstitution([models_dir, 'config_autobots_da.yaml']),
            description='Path to the AutoBots YAML config file',
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            'moflow_checkpoint_path',
            default_value=PathJoinSubstitution([models_dir, 'best_models_ade_student20_da.pth']),
            description='Path to the MoFlow checkpoint file',
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            'moflow_config_path',
            default_value=PathJoinSubstitution([models_dir, 'config_student20_da.yaml']),
            description='Path to the MoFlow YAML config file',
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            'moflow_eval_config_path',
            default_value=PathJoinSubstitution([models_dir, 'moflow_joint_student_eval.yaml']),
            description='Path to the MoFlow eval-time YAML override file',
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            'pose_source',
            default_value='odom',
            description='Robot pose source for prediction: tf, odom, or amcl',
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            'human_source',
            default_value='lidar',
            description='Human detection source for prediction: lidar or zed',
        )
    )
    ld.add_action(ExecuteProcess(
            cmd=[
                LaunchConfiguration('hst_python'),
                '-m',
                'alpaca_navigation.hst_prediction_node',
                '--ros-args',
                '-p',
                ['prediction_backend:=', LaunchConfiguration('prediction_backend')],
                '-p',
                ['eqmotion_bundle_path:=', LaunchConfiguration('eqmotion_bundle_path')],
                '-p',
                ['autobots_checkpoint_path:=', LaunchConfiguration('autobots_checkpoint_path')],
                '-p',
                ['autobots_config_path:=', LaunchConfiguration('autobots_config_path')],
                '-p',
                ['moflow_checkpoint_path:=', LaunchConfiguration('moflow_checkpoint_path')],
                '-p',
                ['moflow_config_path:=', LaunchConfiguration('moflow_config_path')],
                '-p',
                ['moflow_eval_config_path:=', LaunchConfiguration('moflow_eval_config_path')],
                '-p',
                ['pose_source:=', LaunchConfiguration('pose_source')],
                '-p',
                ['human_source:=', LaunchConfiguration('human_source')],
            ],
            output='screen'
        ))

    return ld
