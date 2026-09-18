"""
MPPI Metrics Collector — ROS2 node that records per-trial metrics.

Subscribes to:
  /<ns>/amcl_pose          — pose / path length (map frame, matches goal frame)
  /<ns>/cmd_vel            — control variation
  /<ns>/velodyne_points    — collision proxy (min LiDAR range)
  /<ns>/final_destination  — trial start + target pose (from plan_segmenter)

Saves a JSON file to --output-dir on trial completion or Ctrl-C.

Usage:
  ros2 run alpaca_navigation mppi_metrics_collector \
      --namespace ranger_mini \
      --trial-id 0 \
      --goal-tolerance 0.5 \
      --collision-threshold 0.25 \
      --timeout 120 \
      --output-dir /tmp/sweep_results
"""

import argparse
import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup


def _ns_topic(ns: str, name: str) -> str:
    name = name.lstrip('/')
    return f'/{ns}/{name}' if ns else f'/{name}'


class MPPIMetricsCollector(Node):
    def __init__(self, namespace: str, trial_id: int, output_dir: str,
                 goal_tolerance: float, collision_threshold: float, timeout: float):
        super().__init__('mppi_metrics_collector')
        self._ns = namespace
        self._trial_id = trial_id
        self._output_dir = output_dir
        self._goal_tolerance = goal_tolerance
        self._collision_threshold = collision_threshold
        self._timeout = timeout

        self._cbg = ReentrantCallbackGroup()

        # state
        self._goal: list | None = None       # [x, y] in map frame
        self._start_time: float | None = None
        self._pose: list | None = None       # [x, y] in map frame (from amcl_pose)
        self._done = False

        # metrics accumulators
        self._path_length = 0.0
        self._collision_count = 0
        self._min_clearance = float('inf')
        self._cmd_vels: list[list[float]] = []
        self._prev_cmd = None

        # TRANSIENT_LOCAL so sweep script's ros2 topic echo doesn't miss the one-shot publish
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
        done_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._done_pub = self.create_publisher(Bool, '/mppi_trial_complete', done_qos)

        # final_destination = last waypoint from plan_segmenter = actual destination (map frame)
        self.create_subscription(
            PoseStamped, _ns_topic(namespace, '/final_destination'),
            self._goal_cb, 10, callback_group=self._cbg)

        # amcl_pose: map-frame pose — same frame as goal, correct for distance check
        self.create_subscription(
            PoseWithCovarianceStamped, _ns_topic(namespace, '/amcl_pose'),
            self._amcl_pose_cb, 10, callback_group=self._cbg)

        self.create_subscription(
            Twist, _ns_topic(namespace, '/cmd_vel'),
            self._cmd_vel_cb, 10, callback_group=self._cbg)

        self.create_subscription(
            PointCloud2, _ns_topic(namespace, '/velodyne_points'),
            self._lidar_cb, 10, callback_group=self._cbg)

        # watchdog: check timeout
        self.create_timer(1.0, self._watchdog, callback_group=self._cbg)

        self.get_logger().info(
            f'MetricsCollector trial={trial_id} ns={namespace!r} '
            f'timeout={timeout}s collision_thresh={collision_threshold}m')

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #

    def _goal_cb(self, msg: PoseStamped):
        self._goal = [msg.pose.position.x, msg.pose.position.y]
        if self._start_time is None:
            self._start_time = time.time()
            self.get_logger().info(f'Trial {self._trial_id} started — goal {self._goal}')

    def _amcl_pose_cb(self, msg: PoseWithCovarianceStamped):
        if self._done or self._start_time is None:
            return
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        new_pose = [px, py]

        if self._pose is not None:
            dx = px - self._pose[0]
            dy = py - self._pose[1]
            self._path_length += math.hypot(dx, dy)

        self._pose = new_pose

        if self._goal is not None:
            dist = math.hypot(px - self._goal[0], py - self._goal[1])
            self.get_logger().info (f"distance to goal is {dist}")
            if dist < self._goal_tolerance:
                self._finish(reached=True)

    def _cmd_vel_cb(self, msg: Twist):
        if self._done or self._start_time is None:
            return
        cmd = [msg.linear.x, msg.linear.y, msg.angular.z]
        self._cmd_vels.append(cmd)

    def _lidar_cb(self, msg: PointCloud2):
        """Estimate minimum clearance from LiDAR point cloud."""
        if self._done or self._start_time is None:
            return
        try:
            data = bytes(msg.data)
            point_step = msg.point_step
            n_pts = msg.width * msg.height
            if n_pts == 0 or point_step < 12:
                return
            arr = np.frombuffer(data, dtype=np.uint8).reshape(n_pts, point_step)
            xs = np.frombuffer(arr[:, 0:4].tobytes(), dtype=np.float32)
            ys = np.frombuffer(arr[:, 4:8].tobytes(), dtype=np.float32)
            ranges = np.sqrt(xs ** 2 + ys ** 2)
            valid = ranges[np.isfinite(ranges) & (ranges > 0.05)]
            if valid.size == 0:
                return
            min_r = float(valid.min())
            self._min_clearance = min(self._min_clearance, min_r)
            if min_r < self._collision_threshold:
                self._collision_count += 1
                self.get_logger().warn(
                    f'Near-collision: min_range={min_r:.3f}m < threshold={self._collision_threshold}m')
        except Exception as e:
            self.get_logger().debug(f'lidar parse error: {e}')

    def _watchdog(self):
        if self._done:
            return
        if self._start_time is not None and (time.time() - self._start_time) > self._timeout:
            self.get_logger().warn(f'Trial {self._trial_id} timed out after {self._timeout}s')
            self._finish(reached=False)

    # ------------------------------------------------------------------ #
    # metrics save
    # ------------------------------------------------------------------ #

    def _finish(self, reached: bool):
        if self._done:
            return
        self._done = True

        elapsed = (time.time() - self._start_time) if self._start_time else 0.0

        # control variation: sum of L2 deltas between consecutive cmd_vel
        ctrl_var = 0.0
        if len(self._cmd_vels) > 1:
            arr = np.array(self._cmd_vels)
            ctrl_var = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

        # path efficiency: remaining straight-line dist to goal / path traveled
        if self._pose and self._goal:
            straight = math.hypot(self._pose[0] - self._goal[0],
                                  self._pose[1] - self._goal[1])
        else:
            straight = float('nan')
        efficiency = (straight / self._path_length) if self._path_length > 0 else float('nan')

        metrics = {
            "trial_id": self._trial_id,
            "reached_goal": reached,
            "traversal_time_s": round(elapsed, 3),
            "path_length_m": round(self._path_length, 3),
            "path_efficiency": round(efficiency, 4) if math.isfinite(efficiency) else None,
            "collision_count": self._collision_count,
            "min_clearance_m": round(self._min_clearance, 3) if math.isfinite(self._min_clearance) else None,
            "control_variation": round(ctrl_var, 4),
        }

        os.makedirs(self._output_dir, exist_ok=True)
        out_path = os.path.join(self._output_dir, f'trial_{self._trial_id:04d}.json')
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.get_logger().info(
            f'Trial {self._trial_id} complete — reached={reached} '
            f'time={elapsed:.1f}s collisions={self._collision_count} '
            f'saved to {out_path}')

        self._done_pub.publish(Bool(data=True))

    def destroy_node(self):
        if not self._done and self._start_time is not None:
            self._finish(reached=False)
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', default='ranger_mini')
    parser.add_argument('--trial-id', type=int, default=0)
    parser.add_argument('--output-dir', default='/tmp/mppi_sweep_results')
    parser.add_argument('--goal-tolerance', type=float, default=0.5)
    parser.add_argument('--collision-threshold', type=float, default=0.25,
                        help='LiDAR range below this counts as near-collision (metres)')
    parser.add_argument('--timeout', type=float, default=120.0,
                        help='Max seconds per trial before declaring failure')
    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = MPPIMetricsCollector(
        namespace=parsed.namespace,
        trial_id=parsed.trial_id,
        output_dir=parsed.output_dir,
        goal_tolerance=parsed.goal_tolerance,
        collision_threshold=parsed.collision_threshold,
        timeout=parsed.timeout,
    )
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
