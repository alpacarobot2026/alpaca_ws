import torch
import math
import numpy as np
from alpaca_navigation.mppi_config import*
from nav_msgs.msg import OccupancyGrid
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F
class Costmap: 
    def __init__ (self, msg: OccupancyGrid = None, data:np.ndarray = None):
        if msg is not None:
            self.resolution = msg.info.resolution
            self.width = msg.info.width
            self.height = msg.info.height
            self.origin_x = msg.info.origin.position.x
            self.origin_y = msg.info.origin.position.y
            self.origin_theta = 2.0 * math.atan2(msg.info.origin.orientation.z, msg.info.origin.orientation.w)
            self.data = np.array(msg.data).reshape((self.height, self.width))
        else: 
            
            self.resolution = 0.05
            self.origin_x = - data.shape[0]*self.resolution / 2
            self.origin_y = - data.shape[1]*self.resolution / 2
            self.width = data.shape [0]
            self.height = data.shape [1]
            self.data = data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def steering_angle_single_action (action:torch.Tensor):
        x = action[0]
        z = action[2]

        k = 1.0 if x*z >= 0 else -1.0

        radius = torch.abs (x / (z + 1e-9))

        is_ackermann = True if abs(radius) >= MIN_TURN_RADIUS else False

        l = 0.494  # wheelbase ( front to back )
        w = 0.364  # track (left to right)

        phi = math.atan ( (l/2)/radius)*k if is_ackermann else torch.sign(z) *math.pi / 4.0
        return phi, is_ackermann
    
def calculate_steering_angle (action:torch.Tensor):
    x = action[:,:,0]
    z = action[:,:,2]

    k = torch.zeros (x.shape).to(device)
    k = torch.where (x*z >= 0, 1.0, -1.0)

    radius = torch.abs(x / (z + 1e-9)).to(device)


    is_ackermann = torch.abs(radius) >= MIN_TURN_RADIUS

    l = torch.tensor(0.494).to(device) # wheelbase ( front to back )
    w = torch.tensor(0.364).to(device) # track (left to right)
    x = torch.sqrt (radius**2 - (l/2)**2)
    phi = torch.atan ( (l/2)/radius )
    phi = phi * k
    phi = torch.where (~is_ackermann, torch.sign (z) * torch.tensor(np.pi / 4.0).to(device), phi)

    return phi, is_ackermann


def goal_progress_cost (state: torch.Tensor, goal : torch.Tensor) -> torch.Tensor:
        goal_expanded = goal [None,:]
        state_squeezed = state.squeeze()

        esp = 1e-9

        numerator = torch.norm (goal_expanded - state_squeezed [:,1:,:2], dim = 2)
        denominator = torch.norm (goal_expanded - state_squeezed [:,:-1,:2], dim = 2) 

        ratio = torch.log (numerator + esp) - torch.log (denominator + esp)
        cost = ratio * torch.exp(-torch.arange(numerator.shape[1], dtype=torch.float32, device=numerator.device) * 0.1)
        
        cost = torch.relu (cost.mean (dim = 1))

        return cost 

def goal_progress_cost_relu (state: torch.Tensor, goal : torch.Tensor) -> torch.Tensor:
    goal_expanded = goal [None,:]
    state_squeezed = state.squeeze()

    dist_start = torch.norm (goal_expanded - state_squeezed [:,0,:2], dim = 1)
    dists = torch.norm (goal_expanded - state_squeezed [:,:,:2], dim = -1)

    dist_normalized = dists / (dist_start [:,None] + 1e-9)

    dd = dist_normalized[:, 1:] - dist_normalized[:, :-1] 

    cost = torch.relu (dd).sum (dim = 1)

    return cost 

def goal_progress_cost_naive (state, goal, discount_factor = 6 / HORIZON_LENGTH) -> torch.Tensor:
    goal_expanded = goal [None,:]
    state_squeezed = state.squeeze()


    dist_start = torch.norm (goal_expanded - state_squeezed [:,0,:2], dim = 1)

    dist = torch.norm (goal_expanded - state_squeezed [:,:,:2], dim = 2) / (dist_start [:,None] + 1e-9) # normalize by initial distance to goal
    t = torch.arange (dist.shape[1], device = dist.device, dtype = dist.dtype)
    decay_weight = torch.exp (-discount_factor * t)

    dist = dist * decay_weight
    
    cost = torch.sum (dist, dim =1)

    return cost
    
def terminal_goal_cost (state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    goal_expanded = goal [None,:]
    state_squeezed = state.squeeze()
    

    dist = torch.norm(goal_expanded - state_squeezed[:,-1,:2], dim=1)  # (N,)
    cost = dist / (VMAX * DT * HORIZON_LENGTH)  # Normalize by maximum possible distance traveled in the horizon

    return cost

def steering_cost_i_axis (action: torch.Tensor, prev_u: torch.Tensor) -> torch.Tensor: 
    if prev_u is not None:
        action = action.squeeze()  # shape is (num_particles, horizon, 3)

        prev_steering_angle, prev_is_ackermann = steering_angle_single_action (prev_u[0,:])
        
        curr_steering_angle, curr_is_ackermann  = calculate_steering_angle (action)

        steering_angle_diff = curr_steering_angle - prev_steering_angle

        mode_switch = curr_is_ackermann[:,0].float() != prev_is_ackermann
        mode_switch_penalty = mode_switch * MS_COST

        action_cost = torch.abs (steering_angle_diff [:,0]) # + mode_switch_penalty
                                                                        
    else: 
        action_cost = torch.zeros (NUM_SAMPLES).to(device)
    return action_cost

def steering_cost_t_axis (action:torch.Tensor): 
        # action shape is ( num_particles, horizon, 3)

    steering_angles, _ = calculate_steering_angle (action)  # shape: (num_particles, horizon)
    steering_angles = steering_angles.squeeze()
    steering_angle_diff = steering_angles[:, 1:] - steering_angles[:, :-1]  # shape: (num_particles, horizon-1)

    steering_angle_cost = torch.sum(steering_angle_diff ** 2, dim=1)  # shape: (num_particles)

    return steering_angle_cost

def steering_cost_i_axis_naive (action: torch.Tensor, prev_u = torch.Tensor) -> torch.Tensor:
    if prev_u is not None: 
        action = action.squeeze()
        angular_acceleration = action[:,0,2] - prev_u [None,0,2] 
        angular_acceleration = angular_acceleration / AMAX_Z
        steering_cost =  angular_acceleration ** 2
    else:
         steering_cost = torch.zeros (NUM_SAMPLES, device = device)
    
    return steering_cost

def steering_cost_t_axis_naive (action: torch.Tensor) -> torch.Tensor:
    action = action.squeeze () 
    angular_acceleration = action[:,1:,2] - action[:,:-1,2]
    angular_acceleration = angular_acceleration / AMAX_Z
    steering_cost = torch.sum (angular_acceleration ** 2, dim = 1)
    return steering_cost




def action_cost_i_axis (action: torch.Tensor, prev_u: torch.Tensor) -> torch.Tensor:

    if prev_u is not None:
        action = action.squeeze()  # shape is (num_particles, horizon, 3)
        acceleration = action [:,0,0] - prev_u[None,0,0]
        acceleration = acceleration / AMAX_X
        action_cost = acceleration ** 2
    else: 
        action_cost = torch.zeros (NUM_SAMPLES).to(device)
    return action_cost

def action_cost_t_axis (action: torch.Tensor) -> torch.Tensor:
    action = action.squeeze()  # shape is (num_particles, horizon, 3)
    action_diff = action[:,1:,:] - action[:,:-1,:]
    action_diff = action_diff / AMAX_X
    action_cost = torch.sum (action_diff **2 ,  dim = (1,2))
    return action_cost


def control_cost_x (action: torch.Tensor) -> torch.Tensor: 
    action = action.squeeze()
    action = action / VMAX
    cost = torch.sum (action[:,:,0] ** 2, dim = 1)
    return cost 

def control_cost_z (action: torch.Tensor) -> torch.Tensor: 
    action = action.squeeze()
    action = action / WMAX
    cost = torch.sum (action[:,:,2] ** 2, dim = 1)
    return cost




def costmap_cost (state: torch.Tensor, local_costmap:Costmap, use_sim_cost = False) -> torch.Tensor:
    state_squeezed = state.squeeze()  #  (num_particles, horizon, 3)
    pos_in_costmap_frame = (state_squeezed[:,:,:2] - torch.tensor ([local_costmap.origin_x, local_costmap.origin_y], device=state.device)) 

    grid_x_original = (pos_in_costmap_frame[:,:,0] / local_costmap.resolution).long()
    grid_y_original = (pos_in_costmap_frame[:,:,1] / local_costmap.resolution).long()

    grid_x = torch.clamp((pos_in_costmap_frame[:,:,0] / local_costmap.resolution).long(), 0, local_costmap.width -1)
    grid_y = torch.clamp((pos_in_costmap_frame[:,:,1] / local_costmap.resolution).long(), 0, local_costmap.height -1)

    costmap_cost = local_costmap.data[grid_y.cpu().numpy(), grid_x.cpu().numpy()]  # Shape: (N, T') 

    costmap_cost_tensor = torch.tensor(costmap_cost, dtype=torch.float32, device=device)
    
    costmap_cost_sum = torch.sum(costmap_cost_tensor, dim=1)

    return costmap_cost_sum


def costmap_cost_edt(
    state: torch.Tensor,
    costmap,
    use_sim_cost: bool = False,         # OccupancyGrid 0/100 => False
    sigma_collision: float = 0.30,
    beta_collision: float = 1.5,
    flip_y: bool = False,               # keep for compatibility; usually False for old-style sampling
    # anti-snaking additions
    w_smooth: float = 0.0,
    w_yaw_smooth: float = 0.0,
) -> torch.Tensor:
    """
    EDT collision cost with integer grid indexing, plus optional
    anti-snaking (position curvature and yaw smoothing) terms.
    """

    state_squeezed = state.squeeze()  #  (num_particles, horizon, 3)
    pos_in_costmap_frame = (state_squeezed[:,:,:2] - torch.tensor ([costmap.origin_x, costmap.origin_y], device=state.device))

    grid_x = torch.clamp((pos_in_costmap_frame[:,:,0] / costmap.resolution).long(), 0, costmap.width -1)
    grid_y = torch.clamp((pos_in_costmap_frame[:,:,1] / costmap.resolution).long(), 0, costmap.height -1)

    dist_tensor = costmap.edt_gpu[grid_y, grid_x]  # (N, T') — pure GPU, no sync/transfer

    safety_distance = ROBOT_RADIUS + SAFE_DISTANCE
    edt_cost = torch.nn.functional.softplus((safety_distance - dist_tensor) / sigma_collision, beta=beta_collision)

    wall_cost = edt_cost.sum(dim=1)

    s = state_squeezed 
    N,T,D = state_squeezed.shape 

    # ---- anti-snaking: penalize second difference in position (curvature/weaving) ----
    smooth_cost = torch.zeros((N,), device=device)
    if w_smooth > 0 and T >= 3:
        # use meters for stable scale
        pos_m = torch.stack([s[:, :, 0] - origin_x, s[:, :, 1] - origin_y], dim=-1)
        dp = pos_m[:, 1:, :] - pos_m[:, :-1, :]
        ddp = dp[:, 1:, :] - dp[:, :-1, :]
        smooth_cost = w_smooth * ddp.pow(2).sum(dim=-1).mean(dim=1)

    # ---- yaw smoothing if yaw exists ----
    yaw_cost = torch.zeros((N,), device=device)
    if w_yaw_smooth > 0 and D >= 3 and T >= 2:
        dyaw = s[:, 1:, 2] - s[:, :-1, 2]
        dyaw = (dyaw + torch.pi) % (2 * torch.pi) - torch.pi
        yaw_cost = w_yaw_smooth * dyaw.pow(2).mean(dim=1)

    return wall_cost + smooth_cost + yaw_cost 


def cv_cost (self, state: torch.Tensor) -> torch.Tensor:
    state_squeezed = state.squeeze()  #  (num_particles, horizon, 3)
    cv_cost = torch.zeros(self.num_samples).to(device)
    curr_state = self.agent_states.copy()
    prev_state = self.previous_agent_states.copy()

    for agent_id in curr_state.keys():
        if agent_id in prev_state.keys():
            
            cv_pred, logits = construct_cv_prediction (curr_state[agent_id], self.agent_velocities[agent_id])
            agent_cost =  compute_cv_cost(state_squeezed, cv_pred)
            cv_cost += agent_cost
                
            print (f'mean cv cost for agent {agent_id} is {agent_cost.mean()}')

        else:
            cv_pred, logits = construct_cv_prediction (curr_state[agent_id], torch.tensor ([0.0, 0.0], dtype=torch.float32).to(device))
            cv_cost += compute_cv_cost (state_squeezed, cv_pred)
    return cv_cost

def heading_cost (state: torch.Tensor, goal: torch.Tensor, discount_factor = 3 / HORIZON_LENGTH) -> torch.Tensor:
    state_squeezed = state.squeeze()  #  (num_particles, horizon, 3)
    heading_to_goal = torch.atan2(goal[1] - state_squeezed[:,:,1], goal[0] - state_squeezed[:,:,0])  # Shape: (N, T')
    heading_error = heading_to_goal - state_squeezed[:,:,2]  # Shape: (N, T')


    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]

    heading_cost = 1 - torch.cos(heading_error)  # Shape: (N, T')

    heading_cost [ torch.abs(heading_error) < 3.1415/4]  = 0.0  # for third floor
    

    # exponential decay
    t = torch.arange (heading_cost.shape[1], device = heading_cost.device, dtype = heading_cost.dtype)
    decay_weight = torch.exp (-discount_factor * t)
    
    heading_cost = heading_cost * decay_weight 

    heading_cost = (heading_cost).sum (dim = 1)  # Shape: (N,)


    return heading_cost

def SocialCost(self, state: torch.Tensor,i,human_states, **kwargs) -> torch.Tensor:
        sm_cost = 0.0
        self.get_interacting_agents()
        if i in self.interacting_agents:
            state_squeezed = state.squeeze()
            r_c = (state_squeezed[:,:,:2] + human_states) / 2
            r_ac = state_squeezed[:,:,:2] - r_c
            r_bc = human_states - r_c
            r_ac_3d = torch.nn.functional.pad(r_ac, (0, 1), "constant", 0)  # [N, T', 3]
            r_bc_3d = torch.nn.functional.pad(r_bc, (0, 1), "constant", 0)  # [N, T', 3]
            robot_velocity_3d = torch.nn.functional.pad(self.robot_velocity, (0, 1), "constant", 0)  # Shape: [3]
            agent_velocities_3d = torch.nn.functional.pad(self.agent_velocities[i], (0, 1), "constant", 0) 
            l_ab = torch.cross(r_ac_3d, robot_velocity_3d[None,None,:], dim=2) + torch.cross(r_bc_3d, agent_velocities_3d[None,None,:], dim=2)
            l_ab = l_ab[:, :, 2]
            l_ab_dot_product = l_ab[:, :-1] * l_ab[:, 1:]    # Determine if dot product is positive or not
            condition = l_ab_dot_product > 0  # Shape: [N, T'-1]
            l_ab_conditional = torch.where(condition, -11*torch.abs(l_ab[:, :-1]), torch.tensor(10.0, device=l_ab.device))  # Shape: [N, T'-1]
            sm_cost += torch.sum(l_ab_conditional, dim=1)
        return sm_cost
        
def collision_avoidance_cost(self,state):
    xy_coords = state[..., :2].cpu().numpy()  # Shape: (N, T', 2)
    flattened_coords = xy_coords.reshape(-1, 2)
    x_min, y_min, x_max, y_max = self.bounds
    within_bounds = (
        (flattened_coords[:, 0] >= x_min) &
        (flattened_coords[:, 0] <= x_max) &
        (flattened_coords[:, 1] >= y_min) &
        (flattened_coords[:, 1] <= y_max)
    )

    # Use Shapely's vectorized `contains` for points within bounds
    collision_flags = contains(self.multi_polygon, flattened_coords[:, 0], flattened_coords[:, 1])
    collision_flags[~within_bounds] = False  # Points outside bounds are not collisions

    # Assign costs based on collision flags
    costs = torch.where(
        torch.tensor(collision_flags, dtype=torch.bool),
        torch.tensor(10.0),  
        torch.tensor(0.0) 
    )
    costs = costs.view(state.shape[0], state.shape[1]).sum(dim=1)

    return costs.to(state.device)

def workspace_bounds_cost(
    state: torch.Tensor,
    action: torch.Tensor = None,
    margin: float = 0.55,
    tau: float = 0.25,
    inside_weight: float = 2.0,
    outside_weight: float = 120.0,
    speed_weight: float = 4.0,
) -> torch.Tensor:
    """Soft workspace barrier with near-boundary speed shaping.

    Signed distance d to the (inset) workspace boundary:
    - d > 0 (inside): exponential barrier near edges.
    - d <= 0 (outside): stronger quadratic penalty.
    - Optional speed penalty near/outside edges to bias turn-in behavior.
    """

    state_squeezed = state.squeeze()
    if state_squeezed.ndim == 2:
        state_squeezed = state_squeezed.unsqueeze(1)

    x = state_squeezed[..., 0]
    y = state_squeezed[..., 1]

    # Workspace bounds from measured corners in map frame.
    x_min = -1.07
    x_max = 1.597
    y_min =  -1.975
    y_max =  1.798

    # Inset bounds for a safety corridor.
    x_min_eff = x_min + margin
    x_max_eff = x_max - margin
    y_min_eff = y_min + margin
    y_max_eff = y_max - margin

    d_left = x - x_min_eff
    d_right = x_max_eff - x
    d_bottom = y - y_min_eff
    d_top = y_max_eff - y
    d = torch.minimum(torch.minimum(d_left, d_right), torch.minimum(d_bottom, d_top))

    inside_mask = d > 0.0
    denom = max(tau, 1e-6)
    inside_barrier = inside_weight * torch.exp(-torch.clamp(d, min=0.0) / denom)
    outside_penalty = outside_weight * torch.relu(-d) ** 2
    cost_per_timestep = torch.where(inside_mask, inside_barrier, outside_penalty)

    if action is not None:
        action_squeezed = action.squeeze()
        if action_squeezed.ndim == 2:
            action_squeezed = action_squeezed.unsqueeze(1)
        v = action_squeezed[..., 0]
        proximity = torch.exp(-torch.clamp(d, min=0.0) / denom)
        cost_per_timestep = cost_per_timestep + speed_weight * (v ** 2) * proximity

    return cost_per_timestep.sum(dim=1)
