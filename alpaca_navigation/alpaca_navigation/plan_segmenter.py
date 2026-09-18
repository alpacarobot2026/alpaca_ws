#!/usr/bin/env python3
import math

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray


class PlanSegmenter(Node):
    def __init__(self):
        super().__init__('plan_segmenter')

        self.declare_parameter('segment_length', 8.0)
        self.declare_parameter('min_segment_length', 3.5)
        self.declare_parameter('curvature_window', 8.0)
        self.declare_parameter('curvature_gain', 2.0)
        self.declare_parameter('continuous_carrot', True)
        self.declare_parameter('auto_advance', False)
        self.declare_parameter('goal_tolerance', 0.6)

        self.segment_length = self.get_parameter('segment_length').value
        self.min_segment_length = self.get_parameter('min_segment_length').value
        self.curvature_window = self.get_parameter('curvature_window').value
        self.curvature_gain = self.get_parameter('curvature_gain').value
        self.continuous_carrot = self.get_parameter('continuous_carrot').value
        self.auto_advance = self.get_parameter('auto_advance').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value

        self.min_segment_length = max(0.1, min(self.min_segment_length, self.segment_length))
        self.curvature_window = max(0.1, self.curvature_window)

        self.path = None
        self.robot_pose = None
        self.discrete_goals = []
        self.current_idx = -1
        self.current_goal = None
        self.last_marker_count = 0

        self._plan_client = ActionClient(
            self,
            ComputePathToPose,
            'compute_path_to_pose',
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.final_goal_sub = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self._final_goal_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.amcl_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        self.goal_pub = self.create_publisher(PoseStamped, 'goal', 10)
        self.final_goal_pub = self.create_publisher(PoseStamped, 'final_destination', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'goal_markers', 10)

        self.next_goal_srv = self.create_service(
            Trigger,
            'next_goal',
            self.next_goal_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.reached_goal_srv = self.create_service(
            Trigger,
            'reached_goal',
            self.reached_goal_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.cancel_navigation_srv = self.create_service(
            Trigger,
            'cancel_navigation',
            self.cancel_navigation_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.get_logger().info(
            f'PlanSegmenter started. segment_length={self.segment_length:.2f}m '
            f'min_segment_length={self.min_segment_length:.2f}m '
            f'curvature_window={self.curvature_window:.2f}m '
            f'curvature_gain={self.curvature_gain:.2f} '
            f'continuous_carrot={self.continuous_carrot} '
            f'auto_advance={self.auto_advance}'
        )

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_pose = PoseStamped()
        self.robot_pose.header = msg.header
        self.robot_pose.pose = msg.pose.pose

        if not self.path:
            return

        if self.continuous_carrot:
            self.publish_carrot_from_robot()
        elif self.auto_advance:
            self.advance_if_close()

    def distance2d(self, p1: PoseStamped, p2: PoseStamped) -> float:
        dx = p1.pose.position.x - p2.pose.position.x
        dy = p1.pose.position.y - p2.pose.position.y
        return math.hypot(dx, dy)

    def heading(self, p1: PoseStamped, p2: PoseStamped) -> float:
        dx = p2.pose.position.x - p1.pose.position.x
        dy = p2.pose.position.y - p1.pose.position.y
        return math.atan2(dy, dx)

    def angle_diff(self, a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def interpolate_pose(self, start: PoseStamped, end: PoseStamped, ratio: float) -> PoseStamped:
        ratio = max(0.0, min(1.0, ratio))
        pose = PoseStamped()
        pose.header = end.header
        pose.pose.position.x = start.pose.position.x + ratio * (
            end.pose.position.x - start.pose.position.x
        )
        pose.pose.position.y = start.pose.position.y + ratio * (
            end.pose.position.y - start.pose.position.y
        )
        pose.pose.position.z = start.pose.position.z + ratio * (
            end.pose.position.z - start.pose.position.z
        )
        pose.pose.orientation = end.pose.orientation
        return pose

    def curvature_ahead(self, poses, start_idx: int) -> float:
        if start_idx + 1 >= len(poses):
            return 0.0

        start_heading = self.heading(poses[start_idx], poses[start_idx + 1])
        distance_ahead = 0.0
        max_weighted_curvature = 0.0

        for i in range(start_idx + 1, len(poses)):
            prev_pose = poses[i - 1]
            curr_pose = poses[i]
            seg_len = self.distance2d(prev_pose, curr_pose)
            distance_ahead += seg_len

            if seg_len > 1e-6:
                heading_change = abs(
                    self.angle_diff(self.heading(prev_pose, curr_pose), start_heading)
                )
                curvature = heading_change / max(distance_ahead, 1e-6)
                weight = max(0.0, 1.0 - distance_ahead / self.curvature_window)
                max_weighted_curvature = max(max_weighted_curvature, curvature * weight)

            if distance_ahead >= self.curvature_window:
                break

        return max_weighted_curvature

    def distance_to_turn(self, poses, start_idx: int) -> float:
        if start_idx + 1 >= len(poses):
            return math.inf

        turn_angle_threshold = 0.55
        start_heading = self.heading(poses[start_idx], poses[start_idx + 1])
        distance_ahead = 0.0

        for i in range(start_idx + 1, len(poses)):
            prev_pose = poses[i - 1]
            curr_pose = poses[i]
            seg_len = self.distance2d(prev_pose, curr_pose)
            distance_ahead += seg_len

            if seg_len > 1e-6:
                heading_change = abs(
                    self.angle_diff(self.heading(prev_pose, curr_pose), start_heading)
                )
                if heading_change >= turn_angle_threshold:
                    return distance_ahead

            if distance_ahead >= self.curvature_window:
                break

        return math.inf

    def lookahead_for_index(self, poses, start_idx: int) -> float:
        if len(poses) < 3:
            return self.segment_length

        curvature = self.curvature_ahead(poses, start_idx)
        lookahead = self.segment_length / (1.0 + self.curvature_gain * curvature)
        lookahead = max(self.min_segment_length, min(self.segment_length, lookahead))

        turn_distance = self.distance_to_turn(poses, start_idx)
        if turn_distance < lookahead:
            turn_buffer = min(1.0, self.min_segment_length * 0.25)
            min_turn_lookahead = max(1.0, self.min_segment_length * 0.35)
            turn_lookahead = max(min_turn_lookahead, turn_distance - turn_buffer)
            lookahead = min(lookahead, turn_lookahead)

        return lookahead

    def pose_at_distance(self, poses, start_idx: int, start_pose: PoseStamped, distance: float):
        remaining = max(0.0, distance)
        segment_start = start_pose

        for i in range(start_idx + 1, len(poses)):
            segment_end = poses[i]
            seg_len = self.distance2d(segment_start, segment_end)
            if seg_len < 1e-6:
                segment_start = segment_end
                continue

            if remaining <= seg_len:
                return self.interpolate_pose(segment_start, segment_end, remaining / seg_len), i - 1

            remaining -= seg_len
            segment_start = segment_end

        return poses[-1], len(poses) - 1

    def project_to_path(self, pose: PoseStamped, poses):
        if not poses:
            return None, -1
        if len(poses) == 1:
            return poses[0], 0

        px = pose.pose.position.x
        py = pose.pose.position.y
        best_pose = poses[0]
        best_idx = 0
        best_dist_sq = math.inf

        for i in range(len(poses) - 1):
            start = poses[i]
            end = poses[i + 1]
            sx = start.pose.position.x
            sy = start.pose.position.y
            vx = end.pose.position.x - sx
            vy = end.pose.position.y - sy
            seg_len_sq = vx * vx + vy * vy
            if seg_len_sq < 1e-12:
                continue

            ratio = ((px - sx) * vx + (py - sy) * vy) / seg_len_sq
            ratio = max(0.0, min(1.0, ratio))
            projected_x = sx + ratio * vx
            projected_y = sy + ratio * vy
            dist_sq = (px - projected_x) ** 2 + (py - projected_y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_pose = self.interpolate_pose(start, end, ratio)
                best_idx = i

        return best_pose, best_idx

    def carrot_from_pose(self, pose: PoseStamped):
        if not self.path or not self.path.poses:
            return None, -1

        poses = self.path.poses
        projected_pose, projected_idx = self.project_to_path(pose, poses)
        if projected_pose is None:
            return None, -1

        lookahead = self.lookahead_for_index(poses, projected_idx)
        return self.pose_at_distance(poses, projected_idx, projected_pose, lookahead)

    def segment_path(self, path: Path):
        poses = path.poses
        if not poses:
            return []

        goals = [poses[0]]
        start_pose = poses[0]
        start_idx = 0

        while start_idx < len(poses) - 1:
            lookahead = self.lookahead_for_index(poses, start_idx)
            goal, goal_idx = self.pose_at_distance(poses, start_idx, start_pose, lookahead)
            if self.distance2d(goals[-1], goal) < 1e-6:
                break
            goals.append(goal)
            start_pose = goal
            start_idx = goal_idx

        if self.distance2d(goals[-1], poses[-1]) > 1e-6:
            goals.append(poses[-1])

        return goals

    def publish_goal(self, goal: PoseStamped, marker_idx: int = None):
        self.current_goal = goal
        if marker_idx is not None:
            self.current_idx = marker_idx
        self.goal_pub.publish(goal)
        self.publish_markers()

    def publish_carrot_from_robot(self):
        if self.robot_pose is None:
            return

        goal, _ = self.carrot_from_pose(self.robot_pose)
        if goal is None:
            return

        self.publish_goal(goal, self.nearest_discrete_goal_idx(goal))

    def nearest_discrete_goal_idx(self, goal: PoseStamped) -> int:
        if not self.discrete_goals:
            return -1
        return min(
            range(len(self.discrete_goals)),
            key=lambda i: self.distance2d(goal, self.discrete_goals[i]),
        )

    def publish_discrete_goal(self):
        if 0 <= self.current_idx < len(self.discrete_goals):
            self.publish_goal(self.discrete_goals[self.current_idx], self.current_idx)
        else:
            self.get_logger().warn('No valid current goal to publish.')

    def advance_if_close(self):
        if self.robot_pose is None or not (0 <= self.current_idx < len(self.discrete_goals)):
            return
        if self.current_idx + 1 >= len(self.discrete_goals):
            return

        if self.distance2d(self.robot_pose, self.discrete_goals[self.current_idx]) <= self.goal_tolerance:
            self.current_idx += 1
            self.publish_discrete_goal()

    def publish_markers(self):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, goal in enumerate(self.discrete_goals):
            marker = self.make_marker(
                goal,
                i,
                scale=0.15,
                color=(0.0, 1.0, 0.0, 1.0) if i == self.current_idx else (0.0, 0.2, 1.0, 0.8),
                stamp=now,
            )
            ma.markers.append(marker)

        if self.current_goal is not None:
            marker = self.make_marker(
                self.current_goal,
                len(self.discrete_goals),
                scale=0.25,
                color=(1.0, 0.5, 0.0, 1.0),
                stamp=now,
            )
            ma.markers.append(marker)

        self.marker_pub.publish(ma)
        self.last_marker_count = len(ma.markers)

    def make_marker(self, pose: PoseStamped, marker_id: int, scale: float, color, stamp):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = pose.header.frame_id if pose.header.frame_id else 'map'
        marker.ns = 'segment_goals'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.lifetime = Duration()
        return marker

    def clear_markers(self):
        if self.last_marker_count <= 0:
            return

        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i in range(self.last_marker_count):
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = 'map'
            marker.ns = 'segment_goals'
            marker.id = i
            marker.action = Marker.DELETE
            ma.markers.append(marker)

        self.marker_pub.publish(ma)
        self.last_marker_count = 0

    def _final_goal_callback(self, msg: PoseStamped):
        self.get_logger().info(
            f'Final goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )
        if not self._plan_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('compute_path_to_pose action server not available')
            return

        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = msg
        goal_msg.planner_id = ''
        goal_msg.use_start = False

        future = self._plan_client.send_goal_async(goal_msg)
        future.add_done_callback(self._plan_response_callback)

    def _plan_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('compute_path_to_pose goal rejected')
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._plan_result_callback)

    def _plan_result_callback(self, future):
        path = future.result().result.path
        if not path.poses:
            self.get_logger().warn('compute_path_to_pose returned empty path')
            self.clear_plan()
            return

        self.path = path
        self.discrete_goals = self.segment_path(path)
        self.current_idx = 1 if len(self.discrete_goals) > 1 else 0
        self.current_goal = None

        self.get_logger().info(f'Segmented path into {len(self.discrete_goals)} goals.')
        self.final_goal_pub.publish(self.discrete_goals[-1])

        if self.continuous_carrot and self.robot_pose is not None:
            self.publish_carrot_from_robot()
        else:
            self.publish_discrete_goal()

    def _plan_topic_callback(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn('Received empty plan on plan topic')
            return

        self.path = msg
        self.discrete_goals = self.segment_path(msg)
        self.current_idx = 1 if len(self.discrete_goals) > 1 else 0
        self.current_goal = None

        self.get_logger().info(f'Plan topic: segmented into {len(self.discrete_goals)} goals.')
        self.final_goal_pub.publish(self.discrete_goals[-1])

        if self.continuous_carrot and self.robot_pose is not None:
            self.publish_carrot_from_robot()
        else:
            self.publish_discrete_goal()

    def next_goal_callback(self, request, response):
        if not self.discrete_goals:
            response.success = False
            response.message = 'No active plan/segment goals available.'
            return response

        if self.continuous_carrot:
            self.publish_carrot_from_robot()
            response.success = not self.final_goal_reached()
            response.message = (
                'Published updated carrot goal.'
                if response.success else
                'Final goal already reached. No more goals in this plan.'
            )
            return response

        if self.current_idx + 1 < len(self.discrete_goals):
            self.current_idx += 1
            self.publish_discrete_goal()
            response.success = True
            response.message = f'Published next goal {self.current_idx + 1}/{len(self.discrete_goals)}.'
        else:
            response.success = False
            response.message = 'Last goal already reached. No more goals in this plan.'

        return response

    def reached_goal_callback(self, request, response):
        if not self.discrete_goals:
            response.success = False
            response.message = 'No active plan/segment goals available.'
            return response
        if self.robot_pose is None:
            response.success = False
            response.message = 'Robot pose unavailable.'
            return response

        response.success = self.final_goal_reached()
        response.message = 'Final goal reached.' if response.success else 'Final goal not reached yet.'
        return response

    def cancel_navigation_callback(self, request, response):
        self.clear_plan()
        response.success = True
        response.message = 'Navigation plan canceled.'
        return response

    def final_goal_reached(self):
        if self.robot_pose is None or not self.discrete_goals:
            return False
        return self.distance2d(self.robot_pose, self.discrete_goals[-1]) <= self.goal_tolerance

    def clear_plan(self):
        self.path = None
        self.discrete_goals = []
        self.current_idx = -1
        self.current_goal = None
        self.clear_markers()


def main(args=None):
    rclpy.init(args=args)
    node = PlanSegmenter()
    executor = MultiThreadedExecutor()
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
