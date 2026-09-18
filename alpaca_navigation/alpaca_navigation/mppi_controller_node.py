import gc
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import torch
from pytorch_mppi import MPPI
from alpaca_navigation.mppi_config import *
from alpaca_navigation.sm_mppi import SMMPPIController
import numpy as np
import math
import time
import sys
import argparse
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.vectorized import contains
from datetime import datetime
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.executors import MultiThreadedExecutor
from playsound import playsound
import pyttsx3
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from threading import Thread, Lock
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PointStamped, Point
from std_srvs.srv import Trigger
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float32
from vision_msgs.msg import Detection3D, Detection3DArray
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
import tf2_geometry_msgs  # noqa: F401 - registers PoseStamped transform support
from alpaca_navigation.viz_utils import *
from pathlib import Path
import os
import pickle
import termios
import tty
import select 
from ranger_msgs.msg import RCState


EPSILON = 1e-12



class NonBlockingStdin:
    """Lightweight non-blocking stdin reader for the deadman switch."""

    def __init__(self):
        self.fd = None
        self.old_termios = None
        self.enabled = sys.stdin.isatty()
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.old_termios = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def restore(self):
        if self.enabled and self.old_termios is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_termios)

    def get_key(self):
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            return sys.stdin.read(1)
        return None

class ControlFilter :
    def __init__ (self, x, alpha):
          self.x = x
          self.alpha = alpha 
          self.control = 0
    
    def get_control (self, new_command ):
         new_control = self.alpha * new_command + (1 - self.alpha) * self.control
        
         self.control =  new_control
         
         return self.control


class MPPLocalPlannerMPPI(Node):
    def __init__(self, pose_source_override=None, safety='off', use_hst=False, verbose=False,
                 namespace_override=None, params_file=None, publish_costs=False,
                 save_rollouts=False, rollout_save_dir='rollout_data'):

        print (f'cuda version {torch.version.cuda}, torch version {torch.__version__}, is initialized: {torch.cuda.is_initialized()}, is available {torch.cuda.is_available()}')
        super().__init__('mpc_local_planner_mppi')
        self._ns = namespace_override.strip('/') if namespace_override else ''

        # Initialize parameters

        self._params_file = params_file
        self.rollouts = torch.zeros((7, NUM_SAMPLES, 2))
        self.costs = torch.zeros((7, NUM_SAMPLES, 2))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # ROS2 setup
        self.cbgroup = MutuallyExclusiveCallbackGroup()

        self.thread_safe_group = ReentrantCallbackGroup()
        self.cmd_vel_pub = self.create_publisher(Twist, self._topic('/cmd_vel'), 1)
        self.cmd_vel_debug = self.create_publisher(Twist, self._topic('/cmd_vel_debug'), 1)


        self.timer = self.create_timer(HZ, self.plan_and_publish, callback_group=self.cbgroup)
        self.time_steps = []
        self.linear_velocities = []
        self.angular_velocities = []
        self.start_time = time.time()



        params = self._load_params(use_hst)

        self.controller = SMMPPIController(STATIC_OBSTACLES, self.device, verbose = False, params = params)
        self.controller.publish_cost_breakdown = publish_costs
        self.get_logger().info(f'Horizion lenght :{self.controller.horizon}')
        self.counter = 0
        
        self.x_controller = ControlFilter (True, 1.0)
        self.z_controller = ControlFilter (False, 0.5) 

        self.alpha = 0.7
    
        
        self.current_state = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).to(self.device)  # [x, y, yaw]
        self.robot_velocity = torch.tensor([0.0, 0.0], dtype=torch.float32).to(self.device)
        self.previous_robot_state = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).to(self.device)
        
        self.agents = {}
        self.agents_last_seen = {} 
        self.agent_velocities = {}
        self.agent_time = 0.0
        self.prev_agent_time = 0.0

        self.prev_control = 0.0
        self.control_variation = 0.0
        self.speak = False
        self.current_robot_pose = None

        # pose source parameter: 'amcl' (default), 'odom', or 'mocap'
        if pose_source_override is not None:
            self.pose_source = str(pose_source_override).lower()
            self.get_logger().info(f"Pose source override provided: '{self.pose_source}'")
        else:
            self.declare_parameter('pose_source', 'amcl')
            try:
                self.pose_source = str(self.get_parameter('pose_source').value).lower()
            except Exception:
                self.pose_source = 'amcl'

        # subscribe to current position based on pose_source
        if self.pose_source == 'odom':
            self.get_logger().info("Pose source set to 'odom' - subscribing to /odom")
            self.robot_pose_subscriber = self.create_subscription(Odometry, self._topic('/odom'), self.robot_pose_cb, 10, callback_group=self.thread_safe_group)
        elif self.pose_source == 'mocap':
            self.get_logger().info (self.pose_source)
            self.get_logger().info("Pose source set to 'mocap' - polling map -> base_link_mocap TF")
            self.robot_pose_subscriber = None
            self.mocap_pose_timer = self.create_timer(
                HZ, self._mocap_pose_timer_cb, callback_group=self.thread_safe_group
            )
        else:
            self.get_logger().info("Pose source set to 'amcl' - subscribing to /amcl_pose")
            self.robot_pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, self._topic('/amcl_pose'), self.robot_pose_cb, 10, callback_group=self.thread_safe_group)

        # `PlanSegmenter` publishes the current goal as a `PoseStamped` on `/goal`.
        # Subscribe to `/goal` to receive the published goals.
        self.goal_subscriber = self.create_subscription(PoseStamped, self._topic('/goal'), self.goal_cb, 10, callback_group=self.thread_safe_group)
        self.curr_goal = None
        self.stop_requested = False
        self.stop_cooldown_until = 0.0
        self._next_goal_pending = False     # prevents flooding plan_segmenter on every tick
        self._next_goal_sent_at = 0.0       # timestamp when next_goal was last sent
        self._next_goal_timeout = 5.0       # reset flag if goal_cb hasn't fired in 5s

        # Create a client for the `/next_goal` Trigger service offered by `plan_segmenter`.
        self.next_goal_client = self.create_client(Trigger, self._topic('/next_goal'))
        if not self.next_goal_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('`/next_goal` service not available yet. Will try when needed.')

        self.reached_goal_client = self.create_client(Trigger, self._topic('/reached_goal'))
        if not self.reached_goal_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('`/reached_goal` service not available yet. Will try when needed.')

        self.stop_navigation_srv = self.create_service(
            Trigger,
            self._topic('/stop_navigation'),
            self.stop_navigation_callback,
            callback_group=MutuallyExclusiveCallbackGroup()
        )
        
        # subscribe to local costmap 
        self.local_costmap_sub = self.create_subscription(OccupancyGrid, self._topic('/local_costmap/costmap'), self.local_costmap_cb, 10, callback_group=self.thread_safe_group)
        # publish cost to debug
        self.cost_pub = self.create_publisher(Float32, self._topic('/mppi_controller/cost'), 10)

        self._save_rollouts = save_rollouts
        self._rollout_save_dir = Path(rollout_save_dir)
        self._rollout_file = None
        if self._save_rollouts:
            self._rollout_save_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self._rollout_save_path = self._rollout_save_dir / f'rollouts_{ts}.pkl'
            self._rollout_file = open(self._rollout_save_path, 'ab')

        self._publish_costs = publish_costs
        self._cost_pubs: dict = {}
        if self._publish_costs:
            _cost_names = [
                "goal", "terminal_goal", "action", "action_t",
                "steering", "steering_t", "heading", "cv",
                "costmap", "control_z", "control_x", "total",
            ]
            for name in _cost_names:
                self._cost_pubs[name] = self.create_publisher(
                    Float32, self._topic(f'/mppi_costs/{name}'), 10
                )

        # tf buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # publisher for visualizing rollouts (MarkerArray)
        self.rollouts_pub = self.create_publisher(MarkerArray, self._topic('/mppi_rollouts'), 10)
        
        self.global_frame = 'odom' if self.pose_source == 'odom' else 'map'
        # visualization utils
        self.viz_tool = VisualizationUtils(self, global_frame = self.global_frame)
        
        # safety no movement if safety is on 
        self.safety = safety

        # Start a separate thread to monitor keyboard input deadman switch
        self.space_timeout = 0.15
        self.last_space_time = 0.0
        self.running = False
        self.keyboard = None
        self.key_thread = None
        self.running = True
        self.keyboard = NonBlockingStdin()
        self.key_thread = Thread(target=self.key_loop, daemon=True)
        self.key_thread.start()
        
        if use_hst:
             from std_msgs.msg import Float32MultiArray
             self.create_subscription(Float32MultiArray, self._topic('/hst/predictions'), self.hst_prediction_cb, 10, callback_group=self.thread_safe_group)
    
        self.rc_sub = None
        if self.safety == 'rc':
            self.rc_sub = self.create_subscription(RCState, self._topic('/rc_state'), self.rc_state_cb, 10, callback_group=self.thread_safe_group)
            while self.count_publishers(self._topic('/rc_state')) == 0:
                self.get_logger().warn('waiting for /rc_state publisher...')
                time.sleep(1.0)
            self.get_logger().info('/rc_state publisher found, RC safety active')

        self.swa = None
        self.verbose = verbose

        self._mppi_compute_active = False
        self._viz_publish_busy = False
        self._viz_lock = Lock()
        self._viz_candidate_samples = 20
        self._latest_min_cost = None
        self.publish_timer = self.create_timer(0.2, self.publish_costs_and_viz, callback_group=self.thread_safe_group)

    
    def _load_params(self, use_hst: bool) -> dict:
        defaults = {
            "DT": DT,
            "horizon_length": HORIZON_LENGTH,
            "num_sample": NUM_SAMPLES,
            "cov_x": 0.2,
            "cov_z": 0.06,
            "lambda_": 0.18,
            "goal_weight": 2000,
            "terminal_goal_weight": 100,
            "action_weight": 100,
            "action_t_weight": 0, #50
            "control_x_weight": 0,
            "steering_weight": 100, 
            "steering_t_weight": 0, #200
            "control_z_weight": 0,  #200
            "sm_weight": 1,
            "cv_weight": 1200,
            "costmap_weight": 100,
            "sigma_collision": 0.15,
            "beta_collision": 7.0,
            "decay_rate": -0.000001,
            "heading_weight": 0, #800
            "alpha_x": 0.1,
            "alpha_z": 0.3,
            "use_hst": use_hst,
        }

        if self._params_file:
            import yaml
            try:
                with open(self._params_file, 'r') as f:
                    file_data = yaml.safe_load(f) or {}
                overrides = file_data.get('mppi', file_data)
                defaults.update({k: v for k, v in overrides.items() if k in defaults})
                self.get_logger().info(f'Loaded params from {self._params_file}')
            except Exception as e:
                self.get_logger().warn(f'Failed to load params file {self._params_file}: {e}')
        defaults['use_hst'] = use_hst  # always honour runtime flag
        return defaults

    def _topic(self, name):
        if self._ns:
            return f'/{self._ns}/{name.lstrip("/")}'
        return name

    def rc_state_cb(self, msg):
        self.swa = msg.swa
    
    def hst_prediction_cb(self, msg):
        self.controller.update_hst_prediction(msg)
    
    def key_loop(self):
        while self.running and rclpy.ok():
            if self.keyboard is None:
                time.sleep(0.05)
                continue
            key = self.keyboard.get_key()
            if key == " ":
                self.last_space_time = time.time()
            time.sleep(0.01)

    def deadman_active(self):
        """Return True when motion is allowed under the safety deadman switch."""

        return (time.time() - self.last_space_time) <= self.space_timeout

    def publish_twist_command(self, twist_msg):
        """Publish `twist_msg` or zeros depending on the deadman state."""

        if self.safety == 'on':
            if self.deadman_active():
                self.cmd_vel_pub.publish(twist_msg)
                self.cmd_vel_debug.publish(twist_msg)
                
            else:
                self.cmd_vel_debug.publish(twist_msg)
                self.cmd_vel_pub.publish(Twist())

        elif self.safety == 'off': 
            
            if not self.deadman_active():
                
                self.cmd_vel_pub.publish(twist_msg)
                self.cmd_vel_debug.publish(twist_msg)
                
            else:
                self.cmd_vel_debug.publish(twist_msg)
                self.cmd_vel_pub.publish(Twist())
        
        else: 
            if self.swa == 0: 
                self.cmd_vel_pub.publish(twist_msg)
                self.cmd_vel_debug.publish(twist_msg)
            else:
                self.cmd_vel_debug.publish(twist_msg)
                self.cmd_vel_pub.publish(Twist())

        

    def destroy_node(self):
        self.running = False
        try:
            thread = getattr(self, "key_thread", None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            keyboard = getattr(self, "keyboard", None)
            if keyboard is not None:
                keyboard.restore()
        except Exception:
            pass
        try:
            if getattr(self, "_rollout_file", None) is not None:
                self._rollout_file.close()
        except Exception:
            pass
        super().destroy_node()


    def goal_cb (self, msg):
        now = time.time()
        if self.stop_requested and now < self.stop_cooldown_until:
            self.get_logger().info('Ignoring goal during stop cooldown window')
            return

        if msg.header.frame_id and msg.header.frame_id != self.global_frame:
            try:
                msg = self.tf_buffer.transform(msg, self.global_frame)
            except Exception as e:
                self.get_logger().warn(
                    f'Failed to transform goal from {msg.header.frame_id} to {self.global_frame}: {e}')
                return

        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = 2.0 * math.atan2(
            msg.pose.orientation.z, msg.pose.orientation.w
        )  # NOTE: assuming roll and pitch are negligible

        self.curr_goal = torch.tensor ([x, y], dtype=torch.float32).to(self.controller.device)
        self.stop_requested = False
        self._next_goal_pending = False   # new waypoint received — allow next_goal call again
        self.controller.set_goal(self.curr_goal)
        self.get_logger().debug(f'new goal received: {self.curr_goal}')

    def stop_navigation_callback(self, request, response):
        self.stop_requested = True
        self.stop_cooldown_until = time.time() + 1.0
        self.curr_goal = None
        self.publish_twist_command(Twist())
        response.success = True
        response.message = 'Navigation stopped.'
        return response
    
    def human_detection_cb (self, msg):
        self.prev_agent_time = self.agent_time
        self.agent_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.agents_last_seen = self.agents.copy()
        
        for det in msg.detections:
            id = int(det.results[0].hypothesis.class_id)
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y

            x_map, y_map = self.convert_to_map_frame (x, y, msg.header.frame_id)


            if x_map is None or y_map is None:
                self.get_logger().warn (f' No detections or failed to convert to map frame')
                continue
            print ("received human detection id: ", id)
            self.agents[id] = torch.tensor ([x_map, y_map, 0.0], dtype=torch.float32).to(self.device)
            if id in self.agents_last_seen:
                dt = self.agent_time - self.prev_agent_time
                if dt > 0:
                    prev_pos = self.agents_last_seen[id]
                    curr_pos = self.agents[id]
                    velocity = (curr_pos[:2] - prev_pos[:2])/dt

                    if velocity.norm() < 0.1:
                        velocity = torch.tensor ([0.0, 0.0], dtype=torch.float32).to(self.device)

                    self.agent_velocities[id] = velocity
            else:
                self.agent_velocities[id] = torch.tensor ([0.0, 0.0], dtype=torch.float32).to(self.device)
    
    def publish_costs_and_viz(self):
        if self._mppi_compute_active:
            return

        if not self._viz_lock.acquire(blocking=False):
            return

        self._viz_publish_busy = True
        try:
            if self._publish_costs:
                breakdown = getattr(self.controller, 'last_cost_breakdown_tensors', {})
                if breakdown:
                    self.controller.last_cost_breakdown = {
                        name: float(val.cpu().item()) for name, val in breakdown.items()
                    }

                for name, val in self.controller.last_cost_breakdown.items():
                    pub = self._cost_pubs.get(name)
                    if pub is not None:
                        pub.publish(Float32(data=val))

            candidate_states, candidate_costs = self.controller.get_candidate_states_and_costs()
            if candidate_costs is not None:
                self._latest_min_cost = float(torch.min(candidate_costs.detach()).cpu().item())

            candidate_states, candidate_costs = self._sample_for_viz(
                candidate_states,
                candidate_costs,
                max_samples=self._viz_candidate_samples,
            )

            if candidate_costs is not None and candidate_states is not None:
                self.viz_tool.visualize_rollouts(
                    candidate_states,
                    candidate_costs,
                    is_rollout=False,
                    clear=True,
                )

            rollout_states, rollout_costs = self._sample_for_viz(
                self.rollouts,
                self.costs,
                max_samples=1,
            )
            if rollout_costs is not None and rollout_states is not None:
                self.viz_tool.visualize_rollouts(
                    rollout_states,
                    rollout_costs,
                    is_rollout=True,
                    clear=False,
                )

            if self._latest_min_cost is None and rollout_costs is not None:
                self._latest_min_cost = float(torch.min(rollout_costs).item())

            if self._latest_min_cost is not None:
                self.cost_pub.publish(Float32(data=self._latest_min_cost))
        except Exception as e:
            self.get_logger().warn(f'publish_costs_and_viz exception: {e}')
        finally:
            self._viz_publish_busy = False
            self._viz_lock.release()

    def _sample_for_viz(self, states, costs, max_samples):
        if states is None or costs is None:
            return None, None

        with torch.no_grad():
            states = states.detach()
            costs = costs.detach().flatten()

            if states.ndim == 4 and states.shape[0] == 1:
                states = states.squeeze(0)

            if states.ndim != 3 or states.shape[-1] < 2:
                return None, None

            sample_count = states.shape[0]
            if sample_count == 0:
                return None, None

            if max_samples is not None and sample_count > max_samples:
                idx = torch.linspace(
                    0,
                    sample_count - 1,
                    max_samples,
                    device=states.device,
                ).round().long()
                states = states.index_select(0, idx)
                if costs.numel() >= sample_count:
                    costs = costs.index_select(0, idx)
                else:
                    costs = costs[:1]
            elif costs.numel() > sample_count:
                costs = costs[:sample_count]

            if costs.numel() == 0:
                costs = torch.zeros((states.shape[0],), device=states.device, dtype=states.dtype)
            elif costs.numel() < states.shape[0]:
                costs = costs[:1].expand(states.shape[0])

            return states.cpu(), costs.cpu()
        
    def convert_to_map_frame (self, x, y, frame_id):
        point = PointStamped()
        point.header.frame_id = frame_id
        point.point.x = x
        point.point.y = y
        point.point.z = 0.0
        try: 
            transform = self.tf_buffer.lookup_transform('map', frame_id, rclpy.time.Time())
            transformed_point = do_transform_point(point, transform)
            return transformed_point.point.x, transformed_point.point.y
        
        except Exception as e:
            self.get_logger().warn (f'Failed to transform point from {frame_id} to map frame: {e}')
            return None, None

    def _mocap_pose_timer_cb(self):
        """Poll map -> base_link_mocap TF and update current_state (mocap pose source)."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().debug(f'mocap TF lookup failed: {e}')
            return

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w
        yaw = 2.0 * math.atan2(qz, qw)

        self.current_state = torch.tensor(
            [tx, ty, yaw], dtype=torch.float32
        ).to(self.device)

    def robot_pose_cb (self, msg):
        self.get_logger().debug('found robot pose')
        self.current_robot_pose = msg

        if self.current_robot_pose is None:
            self.get_logger().warn("Can't find robot pose")
            return

        # If pose_source is odom, msg will be an Odometry message. Use it to
        # update current_state immediately and use odom twist for velocity.
        if getattr(self, 'pose_source', 'amcl') == 'odom':
            try:
                px = msg.pose.pose.position.x
                py = msg.pose.pose.position.y
                qz = msg.pose.pose.orientation.z
                qw = msg.pose.pose.orientation.w
                yaw = 2.0 * math.atan2(qz, qw)

                self.current_state = torch.tensor([
                    px,
                    py,
                    yaw,
                ], dtype=torch.float32).to(self.device)

                # If odometry contains twist information, use it to set robot_velocity
                try:
                    linx = msg.twist.twist.linear.x
                    liny = msg.twist.twist.linear.y
                except Exception:
                    linx = 0.0
                    liny = 0.0

                self.robot_velocity = torch.tensor([linx, liny], dtype=torch.float32).to(self.device)
                return
            except Exception as e:
                self.get_logger().warn(f'Failed to parse Odometry message: {e}')

        # Fallback / AMCL (PoseWithCovarianceStamped) handling
        try:
            yaw = 2.0 * math.atan2(
                self.current_robot_pose.pose.pose.orientation.z, self.current_robot_pose.pose.pose.orientation.w
            )  # NOTE: assuming roll and pitch are negligible

            self.current_state = torch.tensor(
                [
                    self.current_robot_pose.pose.pose.position.x,
                    self.current_robot_pose.pose.pose.position.y,
                    yaw,
                ],
                dtype=torch.float32,
            ).to(self.device)

        except Exception as e:
            self.get_logger().warn(f'Failed to parse pose message: {e}')

    def local_costmap_cb (self, msg):
        self.controller.set_local_costmap (msg)
        
    def publish_rollouts(self):
        """Append current rollouts to the session's pkl file (no publishing)."""
        if not self._save_rollouts or self._rollout_file is None:
            return

        if getattr(self, 'rollouts', None) is None:
            return

        try:
            rollouts = self.rollouts

            # convert torch -> numpy if needed
            if isinstance(rollouts, torch.Tensor):
                rollouts = rollouts.detach().cpu().numpy()

            # handle possible batch dimension added elsewhere
            if rollouts.ndim == 4 and rollouts.shape[0] == 1:
                # shape like (1, nsamples, horizon, dim) -> squeeze batch
                rollouts = rollouts[0]

            # At this point common shapes include:
            #  - (nsamples, horizon, dim)
            #  - (horizon, nsamples, dim)
            # Normalize to (horizon, nsamples, dim)
            if rollouts.ndim == 3:
                a, b, c = rollouts.shape
                # If first axis is likely nsamples (large), and second is likely horizon (smaller), transpose
                if a > b:
                    # assume (nsamples, horizon, dim) -> transpose
                    rollouts = rollouts.transpose(1, 0, 2)
            else:
                # unsupported shape
                return

            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            costs = self.costs
            if isinstance(costs, torch.Tensor):
                costs = costs.detach().cpu().numpy()
            pickle.dump({'timestamp': ts, 'rollouts': rollouts, 'costs': costs}, self._rollout_file)
            self._rollout_file.flush()

        except Exception as e:
            self.get_logger().warn(f'publish_rollouts exception: {e}')

    def plan_and_publish(self):
        if self.stop_requested or self.curr_goal is None:
            self.publish_twist_command(Twist())
            return

        self.counter += 1        
        if self.counter == 3:
            self.previous_robot_state = self.current_state
            
        if torch.any(self.previous_robot_state):
            self.robot_velocity = (self.current_state[:2] - self.previous_robot_state[:2])/HZ
            self.previous_robot_state = self.current_state
        now = time.time()

        try:
            self._mppi_compute_active = True
            action, self.rollouts, self.costs, termination = self.controller.compute_control(
                self.current_state, self.previous_robot_state, self.robot_velocity, self.agents, self.agents_last_seen, self.agent_velocities
            )
        except Exception as e:
            self.get_logger().error(f'compute_control exception: {e}')
            self.publish_twist_command(Twist())
            return
        finally:
            self._mppi_compute_active = False

        self.publish_rollouts()

        after = time.time ()
        self.get_logger().info (f'MPPI compute_control hz is {1 / (after - now)}')

        now = time.time()
        if action.dim() != 1:
            action = action[0]

        if action is not None and not termination:
            twist_stamped = Twist()
            
            x_effort= action[0].item() if abs(action[0].item()) < VMAX else np.sign(action[0].item())*VMAX 
            y_effort = action[1].item() if abs(action[1].item()) < VMAX else np.sign(action[1].item())*VMAX 

            if abs (y_effort) < 0.01:
                y_effort = 0.0

            x = action[0].item()
            phi = action[2].item()
            
            twist_stamped.linear.x = x 
            twist_stamped.linear.y = y_effort
            twist_stamped.angular.z = phi

            self.control_variation += np.sqrt((self.prev_control - y_effort)**2)
            self.prev_control = y_effort
            self.publish_twist_command(twist_stamped)
            self.time_steps.append(time.time() - self.start_time)

           
        elif termination:
            # Call next_goal ONCE per waypoint — flag prevents flooding plan_segmenter on every timer tick while waiting for the new goal to arrive.
            # Reset pending flag if goal_cb hasn't fired within timeout (message lost)
            if self._next_goal_pending and (time.time() - self._next_goal_sent_at) > self._next_goal_timeout:
                self.get_logger().warn('next_goal pending timeout — retrying')
                self._next_goal_pending = False

            if not self._next_goal_pending:
                self._next_goal_pending = True
                self._next_goal_sent_at = time.time()
                self.get_logger().info(f'Waypoint reached: {self.curr_goal} — requesting next')
                try:
                    if self.next_goal_client.service_is_ready():
                        req = Trigger.Request()
                        future = self.next_goal_client.call_async(req)

                        def _on_next_goal_done(fut):
                            try:
                                res = fut.result()
                                if res.success:
                                    self.get_logger().info(f"/next_goal: {res.message}")
                                else:
                                    # No more waypoints — final goal reached
                                    self.get_logger().info(f"/next_goal done: {res.message}")
                            except Exception as e:
                                self.get_logger().error(f"Failed calling /next_goal: {e}")
                                self._next_goal_pending = False  # allow retry on error

                        future.add_done_callback(_on_next_goal_done)
                    else:
                        self.get_logger().warn('/next_goal service not ready — will retry')
                        self._next_goal_pending = False  # allow retry next tick
                except Exception as e:
                    self.get_logger().error(f'Exception calling /next_goal: {e}')
                    self._next_goal_pending = False
            self.publish_twist_command(Twist())
           
        else:
            self.get_logger().warn("Failed to compute optimal controls")
            self.publish_twist_command(Twist())
        after = time.time ()

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--safety', choices=['on', 'off', 'rc'], default='off')
    parser.add_argument('--use-hst', action='store_true', help='Enable HST prediction')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--pose-source', choices=['amcl', 'odom', 'mocap'], default='amcl',
                        help="Pose source for the controller. 'mocap' polls map->base_link_mocap TF.")
    parser.add_argument('--namespace', default=None,
                        help='ROS topic namespace prefix (e.g. ranger_mini). All topics become /ns/topic.')
    parser.add_argument('--params-file', default=None,
                        help='Path to YAML params file (mppi: section). Overrides hardcoded defaults.')
    parser.add_argument('--publish-costs', action='store_true',
                        help='Publish per-term weighted costs to /mppi_costs/<name>')
    parser.add_argument('--save-rollouts', action='store_true',
                        help='Save each rollout batch to a timestamped pkl file in --rollout-save-dir')
    parser.add_argument('--rollout-save-dir', default='rollout_data',
                        help='Directory to save rollout pkl files when --save-rollouts is set')

    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)

    node = MPPLocalPlannerMPPI(pose_source_override=parsed.pose_source, safety=parsed.safety,
                               use_hst=parsed.use_hst, verbose=parsed.verbose,
                               namespace_override=parsed.namespace, params_file=parsed.params_file,
                               publish_costs=parsed.publish_costs,
                               save_rollouts=parsed.save_rollouts,
                               rollout_save_dir=parsed.rollout_save_dir)

    executor = MultiThreadedExecutor()
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
