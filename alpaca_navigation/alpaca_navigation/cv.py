import torch
import rclpy
from rclpy.node import Node
from alpaca_navigation.mppi_config import *
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point



DT = 0.4
sigma_h = SIGMA_H
sigma_s = SIGMA_S
sigma_r = SIGMA_R
q_obs = Q_OBS



def construct_cv_prediction (state, agent_velocity:torch.tensor, device = 'cuda'):
    dist = agent_velocity * DT
    steps = torch.linspace(1, HORIZON_LENGTH + 1, HORIZON_LENGTH + 1)
    steps = steps.reshape((steps.shape[0], 1)).to(device)
    dist_tile = torch.tile(dist, (HORIZON_LENGTH + 1, 1, 1)).to(device)
    dist_mult = dist_tile * steps.view((-1, 1, 1))
    steps_dist = dist_mult + state[:2]
    steps_dist = steps_dist.unsqueeze(0)
    steps_dist = steps_dist.unsqueeze(0)
    steps_dist = steps_dist.to(device)
    logits = torch.tensor([1]).to(device)

    return steps_dist, logits

def compute_cv_cost (state_squeezed, cv_prediction):
    # basic cv_cost based on l2 norm between predicted trajectory and constant velocity traj

    cv_cost = 0
    cv_flattened = cv_prediction [0,0,1:,0,:].unsqueeze(0)

    dist = state_squeezed[:,:,:2] - cv_flattened  #(N, T, 2) - (N, T, 2)
    dist_norm = torch.norm (dist, dim = -1)  #(N, T)
    dist_norm = torch.where (dist_norm < 0.8, 10e3, 0)

    cv_cost = torch.sum (dist_norm, dim = -1)  # (N,)

    return cv_cost
    

# prediction cost 
def prediction_cost(state, predictions, logits, decay_rate = -0.15 , static=False):
        sigma_h = SIGMA_H
        sigma_s = SIGMA_S
        sigma_r = SIGMA_R
        q_obs = Q_OBS
        state_view = state.squeeze(0)  #(N, T, 3)

                                                                                            # (1, 5000, 12, 3) -> (1, 1, 5000, 1, 12, 1), (1, 1, 12, 13, 2) -> (1, 1, 12, 13, 1)
        dx = state_view[:, None, :, None, 0] - predictions[:,:,:state_view.shape[1],:,0] #(5000, 1, 7, 1, 2) - (1, 20, 7, 13, 2)
        dy = state_view[:, None, :, None, 1] - predictions[:,:,:state_view.shape[1],:,1]
       

        vx = (predictions[:,:,1:state_view.shape[1]+1,:,0] - predictions[:,:,:state_view.shape[1]-1,:,0]) / DT
        vy = (predictions[:,:,1:state_view.shape[1]+1,:,1] - predictions[:,:,:state_view.shape[1]-1,:,1]) / DT

        dx = dx[:, :, 1:, :]  # Align time steps N x S x T' x H
        dy = dy[:, :, 1:, :]  # Align time steps N x S


        # Heading of "other agent"
        obs_theta = torch.arctan2(vy, vx) # N x S x T' x H

       
        # Checking for static obstacles
        static_obs = (torch.norm(torch.stack((vx, vy), dim=-1), dim=-1) < 0.01)# N x S x T' x H
        alpha = (torch.arctan2(dy, dx) - obs_theta + torch.pi/2.0)
        alpha = (torch.remainder(alpha + torch.pi, 2 * torch.pi) - torch.pi) <= 0
        # Sigma values used to create 2D gaussian around obstacles for cost penalty
        sigma = torch.where(alpha, sigma_r, sigma_h)
        sigma = torch.multiply(~static_obs, sigma) + 0.1 # N x S x T' x H
        sigma_s = 1.0 * static_obs + sigma_s * (~static_obs) # N x S x T' x H
        # Variables used in cost_obs function based on sigma and obs_theta
        a = torch.cos(obs_theta) ** 2 / (2 * sigma ** 2) + torch.sin(obs_theta) ** 2 / (2 * sigma_s ** 2)
        b = torch.sin(2 * obs_theta) / (4 * sigma ** 2) - torch.sin(2 * obs_theta) / (4 * sigma_s ** 2)
        c = torch.sin(obs_theta) ** 2 / (2 * sigma ** 2) + torch.cos(obs_theta) ** 2 / (2 * sigma_s ** 2)

        cost = torch.exp(-((a * dx ** 2) + (2 * b * dx * dy) +  (c * dy ** 2))) # N x S x T' x H
        cost = torch.mean(cost, axis=3)
        logits = torch.softmax(logits, dim=-1)
        cost = cost * logits[None, :]

        t = torch.arange (cost.shape[2], device = cost.device, dtype = cost.dtype)
        decay_weight = torch.exp (decay_rate * t)

        cost *= decay_weight[None, None, :]


        cost = torch.mean(cost, axis=2)
        cost = torch.sum(cost, axis=-1)

        return q_obs * (cost) # (N, S)





class CVVizNode(Node):
    def __init__(self):
        super().__init__('cv_viz_node')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.map_frame = 'map'
        self.agents = {}
        self.agents_last_seen = {}
        self.agent_velocities = {}
        self.agent_time = 0.0
        self.prev_agent_time = 0.0
        self.zero_velocity = torch.zeros(2, dtype=torch.float32).to(self.device)
        self.path_publishers = {}
        self.path_topic_prefix = '/predicted_human_path'

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.detection_subscriber = self.create_subscription(
            Detection3DArray,
            '/SimpleTrack_results',
            self.human_detection_cb,
            10,
        )
        self.get_logger().info('CV Viz Node started')


    
    def human_detection_cb (self, msg):
        self.prev_agent_time = self.agent_time
        self.agent_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.agents_last_seen = self.agents.copy()

        if not msg.detections:
            return

        for det in msg.detections:
            id = int(det.results[0].hypothesis.class_id)
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y

            x_map, y_map = self.convert_to_map_frame (x, y, msg.header.frame_id)


            if x_map is None or y_map is None:
                self.get_logger().warn (f' No detections or failed to convert to map frame')
                continue

            self.agents[id] = torch.tensor ([x_map, y_map, 0.0], dtype=torch.float32).to(self.device)
            if id in self.agents_last_seen:
                dt = self.agent_time - self.prev_agent_time
                if dt > 0:
                    prev_pos = self.agents_last_seen[id]
                    curr_pos = self.agents[id]
                    velocity = (curr_pos[:2] - prev_pos[:2])/dt
                    self.agent_velocities[id] = velocity
            else:
                self.agent_velocities[id] = self.zero_velocity.clone()

        self.publish_agent_paths(msg.header.stamp)

    def publish_agent_paths(self, stamp):
        if not self.agents:
            return

        for agent_id, state in self.agents.items():
            velocity = self.agent_velocities.get(agent_id, self.zero_velocity)
            points = self.compute_prediction_points(state, velocity)

            if points is None:
                continue

            path_msg = Path()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = self.map_frame

            for x, y in points.tolist():
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = self.map_frame
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)

            publisher = self._get_path_publisher(agent_id)
            publisher.publish(path_msg)

    def compute_prediction_points(self, agent_state, agent_velocity):
        try:
            steps_dist, _ = construct_cv_prediction(agent_state, agent_velocity, device=self.device)
        except Exception as exc:
            self.get_logger().warn(f'Failed to compute CV prediction for agent {agent_state}: {exc}')
            return None

        points = steps_dist.squeeze().detach().cpu()
        if points.ndim != 2 or points.shape[1] < 2:
            self.get_logger().warn('Unexpected CV prediction shape; skipping path publish')
            return None
        return points

    def _get_path_publisher(self, agent_id):
        if agent_id not in self.path_publishers:
            topic = f'{self.path_topic_prefix}/human_number_{agent_id}'
            self.path_publishers[agent_id] = self.create_publisher(Path, topic, 10)
        return self.path_publishers[agent_id]

    def convert_to_map_frame (self, x, y, frame_id):
        point = PointStamped()
        point.header.frame_id = frame_id
        point.point.x = x
        point.point.y = y
        point.point.z = 0.0
        try:
            transform = self.tf_buffer.lookup_transform(self.map_frame, frame_id, rclpy.time.Time())
            transformed_point = do_transform_point(point, transform)
            return transformed_point.point.x, transformed_point.point.y

        except Exception as e:
            self.get_logger().warn (f'Failed to transform point from {frame_id} to map frame: {e}')
            return None, None

def main(args=None):
    rclpy.init(args=args)
    node = CVVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()