import gc
import os
import psutil
import rclpy
import argparse
import sys
from rclpy.node import Node
import torch
import numpy as np
import time
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from vision_msgs.msg import Detection3DArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
import math
from zed_msgs.msg import ObjectsStamped

from alpaca_navigation.prediction.prediction import (
    DEFAULT_AUTOBOTS_CHECKPOINT,
    DEFAULT_AUTOBOTS_CONFIG,
    DEFAULT_EQMOTION_BUNDLE,
    DEFAULT_MOFLOW_CHECKPOINT,
    DEFAULT_MOFLOW_CONFIG,
    DEFAULT_MOFLOW_EVAL_CONFIG,
    create_predictor,
)
from alpaca_navigation.prediction.prediction_config import (
    HISTORY_LENGTH,
    TIMESTEP,
    WINDOW_LENGTH,
)

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

# Frequency for HST Inference
HST_HZ = 10.0


class CVPredictor:
    """Constant-velocity predictor. Extrapolates from recent positions — no neural net."""

    def __init__(
        self,
        horizon_steps: int = WINDOW_LENGTH - (HISTORY_LENGTH + 1),
        history_len: int = HISTORY_LENGTH + 1,
        update_dt: float = 1.0 / HST_HZ,
        pred_dt: float = TIMESTEP,
    ):
        self.horizon_steps = horizon_steps
        self.history_len = history_len
        self.update_dt = update_dt  # time between calls to update_and_predict
        self.pred_dt = pred_dt      # time between output prediction steps
        self._history = {}  # {agent_id: list[tuple[float, np.ndarray (2,)]]}

    def update_and_predict(self, agents, robot_pose, observation_times=None, now=None):
        active = set(agents.keys())
        for aid in list(self._history.keys()):
            if aid not in active:
                del self._history[aid]

        if not agents:
            return None, None

        for aid in sorted(agents.keys()):
            pos = agents[aid]
            stamp = None if observation_times is None else observation_times.get(aid)
            if stamp is None:
                stamp = time.time() if now is None else now
            buf = self._history.setdefault(aid, [])
            pos_np = np.array(pos, dtype=np.float32)
            if buf and stamp <= buf[-1][0]:
                buf[-1] = (buf[-1][0], pos_np)
            else:
                buf.append((float(stamp), pos_np))
            if len(buf) > self.history_len:
                buf.pop(0)

        pred_ids = sorted(agents.keys())
        T = self.horizon_steps
        A = len(pred_ids)
        preds = np.zeros((T, A, 2), dtype=np.float32)

        for idx, aid in enumerate(pred_ids):
            buf = self._history[aid]
            cur_time, cur = buf[-1]
            if len(buf) >= 2:
                first_time, first_pos = buf[0]
                elapsed = cur_time - first_time
                if elapsed > 1e-6:
                    vel = (cur - first_pos) / elapsed
                else:
                    vel = np.zeros(2, dtype=np.float32)
            else:
                vel = np.zeros(2, dtype=np.float32)
            for t in range(T):
                preds[t, idx] = cur + vel * (t + 1) * self.pred_dt

        pred_tensor = torch.from_numpy(preds).unsqueeze(0).unsqueeze(0)
        return pred_tensor, pred_ids


class HSTPredictionNode(Node):
    def __init__(self, initial_prediction_backend: str = 'hst'):
        node_name = f"hst_prediction_node_{str(initial_prediction_backend).strip().lower()}"
        super().__init__(node_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter('prediction_backend', initial_prediction_backend)
        self.declare_parameter('eqmotion_bundle_path', str(DEFAULT_EQMOTION_BUNDLE))
        self.declare_parameter('autobots_checkpoint_path', str(DEFAULT_AUTOBOTS_CHECKPOINT))
        self.declare_parameter('autobots_config_path', str(DEFAULT_AUTOBOTS_CONFIG))
        self.declare_parameter('moflow_checkpoint_path', str(DEFAULT_MOFLOW_CHECKPOINT))
        self.declare_parameter('moflow_config_path', str(DEFAULT_MOFLOW_CONFIG))
        self.declare_parameter('moflow_eval_config_path', str(DEFAULT_MOFLOW_EVAL_CONFIG))
        self.declare_parameter('human_source', 'lidar')


        self.prediction_backend = str(self.get_parameter('prediction_backend').value).strip().lower()
        self.eqmotion_bundle_path = str(self.get_parameter('eqmotion_bundle_path').value).strip()
        self.autobots_checkpoint_path = str(self.get_parameter('autobots_checkpoint_path').value).strip()
        self.autobots_config_path = str(self.get_parameter('autobots_config_path').value).strip()
        self.moflow_checkpoint_path = str(self.get_parameter('moflow_checkpoint_path').value).strip()
        self.moflow_config_path = str(self.get_parameter('moflow_config_path').value).strip()
        self.moflow_eval_config_path = str(self.get_parameter('moflow_eval_config_path').value).strip()
        self.human_source = str(self.get_parameter('human_source').value).strip().lower()

        self.get_logger().info(f"human source is {self.human_source}")

        self.get_logger().info(
            f"Initializing prediction backend '{self.prediction_backend}' on {self.device}"
        )

        predictor_kwargs = {}
        if self.prediction_backend == 'eqmotion':
            predictor_kwargs['bundle_path'] = self.eqmotion_bundle_path
        elif self.prediction_backend == 'autobots':
            predictor_kwargs['checkpoint_path'] = self.autobots_checkpoint_path
            predictor_kwargs['config_path'] = self.autobots_config_path
        elif self.prediction_backend == 'moflow':
            predictor_kwargs['checkpoint_path'] = self.moflow_checkpoint_path
            predictor_kwargs['config_path'] = self.moflow_config_path
            predictor_kwargs['eval_config_path'] = self.moflow_eval_config_path


        # Initialize predictor
        if self.prediction_backend == 'cv':
            self.predictor = CVPredictor()
        else:
            self.predictor = create_predictor(self.prediction_backend, **predictor_kwargs)
        
        # State storage
        self.current_robot_pose = None # [x, y, theta]
        self.agents = {} # {id: [x, y]}
        self.agent_observed_at = {} # {id: ROS header stamp in seconds}

        # Histories (numpy-based) for each callback
        # Each entry is a np.ndarray snapshot at callback time
        self.robot_pose_history = []          # list[np.ndarray] with shape (3,)
        self.agent_ids_history = []           # list[np.ndarray] with shape (N,)
        self.agent_positions_history = []     # list[np.ndarray] with shape (N, 2)
        self.prediction_history = []          # list[np.ndarray] with shape (T, A, 2)
        self.prediction_ids_history = []      # list[np.ndarray] with shape (A,)
        
        # TF Buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscribers
        self.declare_parameter('pose_source', 'amcl')
        self.pose_source = self.get_parameter('pose_source').value
        self.world_frame = 'odom' if self.pose_source == 'odom' else 'map'
        
        if self.pose_source == 'odom':
            print ('pose source odom')
            self.create_subscription(Odometry, '/odom', self.robot_odom_cb, 10)
        elif self.pose_source == 'tf':
            print('pose source tf (map -> base_link)')
        else:
            self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.robot_amcl_cb, 10)

        
        if self.human_source == 'lidar':    
            self.create_subscription(Detection3DArray, '/SimpleTrack_results', self.human_detection_cb, 10)
        elif self.human_source == 'zed':
            self.create_subscription (ObjectsStamped, '/zed/zed_node/obj_det/objects', self.human_detection_cb, 10)
        else:
            self.get_logger().error(f"Unsupported human_source '{self.human_source}'")
        
        # Publishers
        self.pred_pub = self.create_publisher(Float32MultiArray, f'/hst/predictions_{self.prediction_backend}', 1)
        self.pred_marker_pub = self.create_publisher(MarkerArray, f'/hst/prediction_markers_{self.prediction_backend}', 1)
        self.history_marker_pub = self.create_publisher(MarkerArray, f'/hst/history_markers_{self.prediction_backend}', 1)
        self.robot_marker_pub = self.create_publisher(MarkerArray, f'/hst/robot_markers_{self.prediction_backend}', 1)
        self.marker_frame = 'map'
        self.declare_parameter('history_marker_steps', 20)
        self.history_marker_steps = int(self.get_parameter('history_marker_steps').value)
        
        # Timer
        self.create_timer(1.0 / HST_HZ, self.timer_cb)
        
        self.get_logger().info(
            f"Prediction node started with backend '{self.prediction_backend}'"
        )

        if self.prediction_backend == 'hst':
            if tf is None:
                raise RuntimeError("TensorFlow is required when prediction_backend is 'hst'")
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
              try:
                for gpu in gpus:
                  tf.config.experimental.set_memory_growth(gpu, True)
                self.get_logger().info(f"TF Memory Growth Enabled for {len(gpus)} GPUs")
              except RuntimeError as e:
                self.get_logger().error(f"TF Memory Growth Fail: {e}")


        self.hst_counter = 0 
        self.hst_robot_pose = None

    @staticmethod
    def _agent_dim_label(pred_ids):
        if pred_ids is None:
            return "Agents"
        try:
            ids = [str(int(agent_id)) for agent_id in pred_ids]
        except Exception:
            return "Agents"
        return "Agents:" + ",".join(ids)

    @staticmethod
    def _stamp_to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9


    def robot_odom_cb(self, msg):
        # print ("getting odom msg")
        try:
            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            yaw = 2.0 * math.atan2(qz, qw)
            self.current_robot_pose = [px, py, yaw]

            # Save as numpy array snapshot
            self.robot_pose_history.append(
                np.array(self.current_robot_pose, dtype=np.float32)
            )
        except Exception as e:
            self.get_logger().warn(f"Odom parse fail: {e}")

    def robot_amcl_cb(self, msg):
        try:
            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            yaw = 2.0 * math.atan2(qz, qw)
            self.current_robot_pose = [px, py, yaw]

            # Save as numpy array snapshot
            self.robot_pose_history.append(
                np.array(self.current_robot_pose, dtype=np.float32)
            )
        except Exception as e:
            self.get_logger().warn(f"AMCL parse fail: {e}")
    

    def human_detection_cb(self, msg):
        # Refresh agents from latest detection
        # Note: We rely on the tracker ID persistence.
        self.agents = {}
        current_agents = {}
        current_observed_at = {}
        msg_time = self._stamp_to_seconds(msg.header.stamp)
        if msg_time <= 0.0:
            msg_time = self._stamp_to_seconds(self.get_clock().now().to_msg())
        if self.human_source == 'lidar':
            for det in msg.detections:
                id = int(det.results[0].hypothesis.class_id)
                x = det.bbox.center.position.x
                y = det.bbox.center.position.y
                
                x_world, y_world = self.convert_to_world_frame(x, y, msg.header.frame_id)
                if x_world is not None:
                    current_agents[id] = [x_world, y_world]
                    current_observed_at[id] = msg_time

        elif self.human_source == 'zed':
            zed_frame = msg.header.frame_id
            for obj in msg.objects:
                
                id = int(obj.label_id)
                x = obj.position[0]
                y = obj.position[1]

                x_world, y_world = self.convert_to_world_frame(x, y, zed_frame)
                if x_world is not None:
                    current_agents[id] = [x_world, y_world]
                    current_observed_at[id] = msg_time
                
        # Update state
        self.agents = current_agents
        self.agent_observed_at = current_observed_at

        # Save a numpy snapshot of current agents
        if current_agents:
            ids = np.fromiter(current_agents.keys(), dtype=np.int64)
            positions = np.array(list(current_agents.values()), dtype=np.float32)
        else:
            ids = np.zeros((0,), dtype=np.int64)
            positions = np.zeros((0, 2), dtype=np.float32)
        self.agent_ids_history.append(ids)
        self.agent_positions_history.append(positions)

    def convert_to_map_frame(self, x, y, frame_id):
        return self.convert_to_world_frame(x, y, frame_id, 'map')

    def convert_to_world_frame(self, x, y, frame_id, target_frame=None):
        target_frame = target_frame or self.world_frame
        if frame_id == target_frame:
            return x, y
            
        point = PointStamped()
        point.header.frame_id = frame_id
        point.point.x = float (x)
        point.point.y = float (y)
        point.point.z = 0.0
        try: 
            transform = self.tf_buffer.lookup_transform(target_frame, frame_id, rclpy.time.Time())
            transformed_point = do_transform_point(point, transform)
            return transformed_point.point.x, transformed_point.point.y
        except Exception as e:
            self.get_logger().warn(
                f"TF {target_frame}<-{frame_id} lookup fail: {e}",
                throttle_duration_sec=2.0,
            )
            return None, None

    def _update_pose_from_tf(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            t = transform.transform.translation
            r = transform.transform.rotation
            yaw = 2.0 * math.atan2(r.z, r.w)
            self.current_robot_pose = [t.x, t.y, yaw]
            self.robot_pose_history.append(
                np.array(self.current_robot_pose, dtype=np.float32)
            )
        except Exception as e:
            self.get_logger().warn(f"TF map->base_link lookup fail: {e}", throttle_duration_sec=2.0)

    def timer_cb(self):
        if self.pose_source == 'tf':
            self._update_pose_from_tf()

        if self.current_robot_pose is None:
            print ('No robot pose yet, skipping HST inference')
            return
            
        
        # Run Inference
        try:
            # prediction.py expects robot_pose as list/array [x,y]
            # human_poses as dict {id: [x,y]}
            
            t0 = time.time()
            now_sec = self._stamp_to_seconds(self.get_clock().now().to_msg())
            if self.prediction_backend == 'cv':
                pred_tensor, pred_ids = self.predictor.update_and_predict(
                    self.agents,
                    self.current_robot_pose,
                    observation_times=self.agent_observed_at,
                    now=now_sec,
                )
            else:
                pred_tensor, pred_ids = self.predictor.update_and_predict(self.agents, self.current_robot_pose)

            # pred_tensor shape: (1, 1, Time, Agents, 2) typically
            # Agents dimension includes Robot at idx 0, then humans.
            # We want to publish this tensor.
            
            if pred_tensor is None:
                self.get_logger().info(
                    f"No prediction yet for backend '{self.prediction_backend}'; waiting for enough tracked history."
                )
                pred_tensor = torch.zeros((1, 1, 1, 1, 2), dtype=torch.float32)
                data_np = pred_tensor.detach().cpu().numpy()
                data_squeezed = data_np.squeeze(0).squeeze(0) # (T, A, 2)
                T, A, D = data_squeezed.shape

                # Save prediction snapshot as numpy array
                self.prediction_history.append(data_squeezed.copy())
                if pred_ids is not None:
                    self.prediction_ids_history.append(
                        np.array(list(pred_ids), dtype=np.int64)
                    )
                else:
                    self.prediction_ids_history.append(
                        np.zeros((A,), dtype=np.int64)
                    )
            
                msg = Float32MultiArray()
                msg.layout.dim.append(MultiArrayDimension(label="Time", size=T, stride=T*A*D))
                msg.layout.dim.append(MultiArrayDimension(label=self._agent_dim_label(pred_ids), size=A, stride=A*D))
                msg.layout.dim.append(MultiArrayDimension(label="Coords", size=D, stride=D))
                
                msg.data = data_squeezed.flatten().tolist()         
                self.pred_pub.publish(msg)
                self.publish_prediction_markers(data_squeezed, pred_ids)
                self.publish_history_markers(pred_ids)
                self.publish_robot_markers()
                return

            # Flatten and publish
            # We preserve dimensions in layout
            
            # Tensor on GPU/CPU torch -> CPU numpy
            data_np = pred_tensor.detach().cpu().numpy()
            
            # Check shape
            # expected (1, 1, T, A, 2)
            # We can squeeze first two dims
            data_squeezed = data_np.squeeze(0).squeeze(0) # (T, A, 2)
            if self.prediction_backend not in {'eqmotion', 'autobots', 'moflow', 'cv'}:
                data_squeezed = data_squeezed + np.array(self.current_robot_pose)[None, None, :2]

            if self.prediction_backend in {'eqmotion', 'autobots', 'moflow'}:
                self.get_logger().info(
                    f"{self.prediction_backend} inference: detections={len(self.agents)}, selected={0 if pred_ids is None else len(pred_ids)}, output_agents={data_squeezed.shape[1]}, dt={time.time() - t0:.3f}s"
                )

            # Save prediction snapshot as numpy array
            self.prediction_history.append(data_squeezed.copy())
            if pred_ids is not None:
                self.prediction_ids_history.append(
                    np.array(list(pred_ids), dtype=np.int64)
                )
            else:
                self.prediction_ids_history.append(
                    np.zeros((data_squeezed.shape[1],), dtype=np.int64)
                )

            T, A, D = data_squeezed.shape
            
            msg = Float32MultiArray()
            
            # Define Layout
            # We define 3 dimensions: Time, Agents, Coordinates
            msg.layout.dim.append(MultiArrayDimension(label="Time", size=T, stride=T*A*D))
            msg.layout.dim.append(MultiArrayDimension(label=self._agent_dim_label(pred_ids), size=A, stride=A*D))
            msg.layout.dim.append(MultiArrayDimension(label="Coords", size=D, stride=D))
            
            msg.data = data_squeezed.flatten().tolist()
            
            self.pred_pub.publish(msg)
            self.publish_prediction_markers(data_squeezed, pred_ids)
            self.publish_history_markers(pred_ids)
            self.publish_robot_markers()

            dur = time.time() - t0

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            self.hst_counter += 1

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

    def publish_prediction_markers(self, data_squeezed: np.ndarray, pred_ids):
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.header.frame_id = self.world_frame
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        if pred_ids is None or data_squeezed.size == 0:
            self.pred_marker_pub.publish(marker_array)
            return

        stamp = self.get_clock().now().to_msg()

        # Use all modes if available, otherwise fall back to the published trajectory.
        all_modes = getattr(self.predictor, 'last_all_modes_pred', None)
        selected_agents = getattr(self.predictor, 'last_selected_agents', None)

        if all_modes is not None and selected_agents is not None:
            # all_modes shape: (modes, agents_including_robot, T, 2)
            num_modes = all_modes.shape[0]
            expected_pred = getattr(self.predictor, 'last_expected_pred', None)
            for agent_idx, agent_id in enumerate(selected_agents):
                model_slot = agent_idx + 1  # slot 0 is robot
                if model_slot >= all_modes.shape[1]:
                    break
                r, g, b = self._color_from_id(agent_id)

                # All modes — faint
                for mode_idx in range(num_modes):
                    marker = Marker()
                    marker.header.frame_id = self.world_frame
                    marker.header.stamp = stamp
                    marker.ns = 'hst_predictions'
                    marker.id = int(agent_id) * 100 + mode_idx
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    marker.scale.x = 0.02
                    marker.pose.orientation.w = 1.0
                    marker.color.r = r
                    marker.color.g = g
                    marker.color.b = b
                    marker.color.a = 0.25
                    for t_idx in range(all_modes.shape[2]):
                        point = Point()
                        point.x = float(all_modes[mode_idx, model_slot, t_idx, 0])
                        point.y = float(all_modes[mode_idx, model_slot, t_idx, 1])
                        point.z = 0.0
                        marker.points.append(point)
                    marker_array.markers.append(marker)

                # Expected trajectory — bold
                if expected_pred is not None and model_slot < expected_pred.shape[0]:
                    exp_marker = Marker()
                    exp_marker.header.frame_id = self.world_frame
                    exp_marker.header.stamp = stamp
                    exp_marker.ns = 'hst_predictions_expected'
                    exp_marker.id = int(agent_id)
                    exp_marker.type = Marker.LINE_STRIP
                    exp_marker.action = Marker.ADD
                    exp_marker.scale.x = 0.06
                    exp_marker.pose.orientation.w = 1.0
                    exp_marker.color.r = r
                    exp_marker.color.g = g
                    exp_marker.color.b = b
                    exp_marker.color.a = 0.95
                    for t_idx in range(expected_pred.shape[1]):
                        point = Point()
                        point.x = float(expected_pred[model_slot, t_idx, 0])
                        point.y = float(expected_pred[model_slot, t_idx, 1])
                        point.z = 0.05
                        exp_marker.points.append(point)
                    marker_array.markers.append(exp_marker)
        else:
            # Fallback: single best mode from data_squeezed
            T = data_squeezed.shape[0]
            for agent_idx, agent_id in enumerate(pred_ids):
                if agent_idx >= data_squeezed.shape[1]:
                    break
                marker = Marker()
                marker.header.frame_id = self.world_frame
                marker.header.stamp = stamp
                marker.ns = 'hst_predictions'
                marker.id = int(agent_id)
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.03
                marker.pose.orientation.w = 1.0
                r, g, b = self._color_from_id(agent_id)
                marker.color.r = r
                marker.color.g = g
                marker.color.b = b
                marker.color.a = 0.9
                for t_idx in range(T):
                    point = Point()
                    point.x = float(data_squeezed[t_idx, agent_idx, 0])
                    point.y = float(data_squeezed[t_idx, agent_idx, 1])
                    point.z = 0.0
                    marker.points.append(point)
                marker_array.markers.append(marker)

        self.pred_marker_pub.publish(marker_array)

    def publish_history_markers(self, pred_ids):
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.header.frame_id = self.marker_frame
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        if pred_ids is None:
            target_ids = [int(agent_id) for agent_id in self.agents.keys()]
        else:
            target_ids = [int(agent_id) for agent_id in pred_ids]
        if not target_ids:
            self.history_marker_pub.publish(marker_array)
            return

        # Pull the model's actual sampled history (obs_len points at 0.4s stride)
        # from the adapter's buffer so the visualization matches exactly what the
        # model received, rather than the raw dense detection list.
        adapter = self.predictor
        has_model_buffer = (
            self.prediction_backend in {'eqmotion', 'autobots', 'moflow'}
            and hasattr(adapter, 'human_history_buffer')
            and hasattr(adapter, '_sample_agent_history')
        )

        for agent_id in target_ids:
            marker = Marker()
            marker.header.frame_id = self.marker_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'hst_histories'
            marker.id = int(agent_id)
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.pose.orientation.w = 1.0

            r, g, b = self._color_from_id(agent_id)
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.6

            if has_model_buffer:
                sampled = adapter._sample_agent_history(int(agent_id))
                if isinstance(sampled, tuple):
                    history_np, history_mask = sampled
                else:
                    history_np = sampled
                    history_mask = np.isfinite(history_np).all(axis=1)
                for t in range(history_np.shape[0]):
                    if not history_mask[t]:
                        continue
                    point = Point()
                    point.x = float(history_np[t, 0])
                    point.y = float(history_np[t, 1])
                    point.z = 0.05
                    marker.points.append(point)
            else:
                # Fallback: raw detection list for backends without sampled-history access.
                if not self.agent_ids_history or not self.agent_positions_history:
                    continue
                history_len = min(self.history_marker_steps, len(self.agent_ids_history), len(self.agent_positions_history))
                for frame_ids, frame_positions in zip(
                    self.agent_ids_history[-history_len:],
                    self.agent_positions_history[-history_len:],
                ):
                    if frame_ids.size == 0:
                        continue
                    matches = np.where(frame_ids == agent_id)[0]
                    if matches.size == 0:
                        continue
                    idx = int(matches[0])
                    if idx >= frame_positions.shape[0]:
                        continue
                    point = Point()
                    point.x = float(frame_positions[idx, 0])
                    point.y = float(frame_positions[idx, 1])
                    point.z = 0.05
                    marker.points.append(point)

            if marker.points:
                marker_array.markers.append(marker)

        self.history_marker_pub.publish(marker_array)

    def publish_robot_markers(self):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        clear_marker = Marker()
        clear_marker.header.frame_id = self.marker_frame
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        # Robot history
        history_len = min(self.history_marker_steps, len(self.robot_pose_history))
        if history_len > 1:
            hist_marker = Marker()
            hist_marker.header.frame_id = self.marker_frame
            hist_marker.header.stamp = stamp
            hist_marker.ns = 'robot_history'
            hist_marker.id = 0
            hist_marker.type = Marker.LINE_STRIP
            hist_marker.action = Marker.ADD
            hist_marker.scale.x = 0.05
            hist_marker.pose.orientation.w = 1.0
            hist_marker.color.r = 0.2
            hist_marker.color.g = 0.6
            hist_marker.color.b = 1.0
            hist_marker.color.a = 0.8
            for pose in self.robot_pose_history[-history_len:]:
                p = Point()
                p.x = float(pose[0])
                p.y = float(pose[1])
                p.z = 0.05
                hist_marker.points.append(p)
            marker_array.markers.append(hist_marker)

        # Robot prediction if the backend provides one.
        robot_pred = getattr(self.predictor, 'last_robot_pred', None)
        if robot_pred is not None and robot_pred.ndim == 2 and robot_pred.shape[0] > 0:
            pred_marker = Marker()
            pred_marker.header.frame_id = self.marker_frame
            pred_marker.header.stamp = stamp
            pred_marker.ns = 'robot_prediction'
            pred_marker.id = 1
            pred_marker.type = Marker.LINE_STRIP
            pred_marker.action = Marker.ADD
            pred_marker.scale.x = 0.05
            pred_marker.pose.orientation.w = 1.0
            pred_marker.color.r = 0.2
            pred_marker.color.g = 1.0
            pred_marker.color.b = 0.2
            pred_marker.color.a = 0.9
            for t_idx in range(robot_pred.shape[0]):
                p = Point()
                p.x = float(robot_pred[t_idx, 0])
                p.y = float(robot_pred[t_idx, 1])
                p.z = 0.0
                pred_marker.points.append(p)
            marker_array.markers.append(pred_marker)

        self.robot_marker_pub.publish(marker_array)

    def _color_from_id(self, agent_id):
        hash_val = (int(agent_id) * 2654435761) & 0xFFFFFFFF
        r = ((hash_val >> 16) & 0xFF) / 255.0
        g = ((hash_val >> 8) & 0xFF) / 255.0
        b = (hash_val & 0xFF) / 255.0
        return r, g, b

def main(args=None):
    # Parse our CLI args (leave remaining args for rclpy)
    cli_args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--prediction-backend', help='prediction backend', default=None)
    parsed, remaining = parser.parse_known_args(cli_args)

    initial_pred = parsed.prediction_backend or 'hst'

    # Initialize rclpy with the remaining args (ROS-specific)
    rclpy.init(args=remaining)

    node = HSTPredictionNode(initial_prediction_backend=initial_pred)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

