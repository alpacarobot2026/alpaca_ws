import math

from std_msgs.msg import Float32
import torch

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration

from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import alpaca_navigation.costs as costs


class VisualizationUtils:
    def __init__(self, node: Node, global_frame) -> None:
        self._node = node
        self.global_frame = global_frame

        self._rollouts_pub = self._node.create_publisher(
            MarkerArray, f"/{self._node.get_name()}/vis/rollouts", 1
        )

        self._path_pub = self._node.create_publisher(
            Path, f"/{self._node.get_name()}/vis/path", 1)

        self.cost_topics = {'/cost/terminal_goal_cost':None, '/cost/goal_progress_cost':None, '/cost/action_i_axis':None, '/cost/action_t_axis':None, '/cost/heading':None, '/cost/steering_i_axis':None, '/cost/steering_t_axis':None, '/cost/costmap':None}
        
        for topic_name in self.cost_topics.keys():
            self.cost_topics[topic_name] = self._node.create_publisher(Float32, topic_name, 1)

    
    def publish_costs (self, state: torch.Tensor, action:torch.Tensor, prev_action: torch.Tensor, goal:torch.Tensor):
        terminal_goal_cost = costs.terminal_goal_cost (state, goal)
        goal_progress_cost = costs.goal_progress_cost (state, goal)
        action_i_axis_cost = costs.action_i_axis_cost (action)
        action_t_axis_cost = costs.action_t_axis_cost (action)
        heading_cost = costs.heading_cost (state, goal)
        steering_i_axis_cost = costs.steering_i_axis_cost (action, prev_action)
        steering_t_axis_cost = costs.steering_t_axis_cost (action, prev_action)
        costmap_cost = costs.costmap_cost (state)

        for topic_name, cost in zip (self.cost_topics.keys(), [terminal_goal_cost, goal_progress_cost, action_i_axis_cost, action_t_axis_cost, heading_cost, steering_i_axis_cost, steering_t_axis_cost, costmap_cost]):
            msg = Float32()
            msg.data = cost.item()
            self.cost_topics[topic_name].publish (msg)


    def visualize_rollouts(self, rollouts: torch.Tensor, costs: torch.Tensor, is_rollout, clear=True) -> None:
        """
        Input:
        rollouts: (shape: 1 x NUM_SAMPLES x HORIZON x 3)
        costs: (shape: NUM_SAMPLES)
        """

        rollouts = rollouts.unsqueeze(0)

        assert rollouts.ndim == 4 and rollouts.shape[0] == 1 and rollouts.shape[-1] == 3
        min_cost = torch.min(costs).item()
        max_cost = torch.max(costs).item()

        marker_array = MarkerArray()

        stamp = self._node.get_clock().now().to_msg()

        if clear:
            clear_marker = Marker()
            clear_marker.header.frame_id = self.global_frame
            clear_marker.header.stamp = stamp
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
        
        # min cost trajectory 
        min_cost_id = torch.argmin (costs).item()
        max_cost_id = torch.argmax (costs).item()

        rollouts = rollouts.cpu().numpy()

        if is_rollout:
            min_marker = Marker()
            min_marker.header.frame_id = self.global_frame
            min_marker.header.stamp = stamp
            min_marker.ns = 'mppi_selected_rollout'
            min_marker.id = min_cost_id
            min_marker.type = Marker.LINE_STRIP
            min_marker.scale.x = 0.05
            min_marker.scale.y = 0.05
            min_marker.scale.z = 0.005
            min_marker.lifetime = Duration(seconds=0.07).to_msg()
            min_marker.pose.orientation.w = 1.0
            min_marker.color.g = min_marker.color.a = 1.0

            for state_idx in range(rollouts.shape[2]):
                state = rollouts[0, 0, state_idx, :]
                min_marker.points.append(Point(x=state[0].item(), y=state[1].item()))

            marker_array.markers.append(min_marker)

        else:
            num_sample_to_visualize = min(20, rollouts.shape[1], costs.numel())
            for sample_idx in range(num_sample_to_visualize):
                marker = Marker()
                marker.header.frame_id = self.global_frame
                marker.header.stamp = stamp
                marker.ns = 'mppi_candidate_rollouts'
                marker.id = sample_idx
                marker.type = Marker.LINE_STRIP
                marker.scale.x = 0.01
                marker.scale.y = 0.01
                marker.scale.z = 0.01
                marker.lifetime = Duration(seconds=0.07).to_msg()
                marker.pose.orientation.w = 1.0
                

                cost = costs[sample_idx].item()
                if cost == min_cost:
                    marker.color.r = marker.color.g = marker.color.a = 1.0
                else:
                    denom = max(max_cost - min_cost, 1e-9)
                    cost_prop = (cost - min_cost) / denom
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.a = 1.0

                for state_idx in range(rollouts.shape[2]):
                    state = rollouts[0, sample_idx, state_idx,:]
                    marker.points.append(Point(x=state[0].item(), y=state[1].item()))

                marker_array.markers.append (marker)
            
        self._rollouts_pub.publish(marker_array)

    def visualize_path(self, path: list[tuple[float, float, float]]) -> None:
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self._node.get_clock().now().to_msg()

        for state in path:
            pose = PoseStamped()
            pose.header.frame_id = self.global_frame
            pose.header.stamp = self._node.get_clock().now().to_msg()

            pose.pose.position.x = state[0]
            pose.pose.position.y = state[1]
            pose.pose.orientation.w = math.cos(state[2] / 2)
            pose.pose.orientation.z = math.sin(state[2] / 2)

            path_msg.poses.append(pose)

        self._path_pub.publish(path_msg)
