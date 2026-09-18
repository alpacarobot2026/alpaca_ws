import time
import traceback
import torch
from pytorch_mppi import MPPI, KMPPI, mppi
from alpaca_navigation.mppi_config import *
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.vectorized import contains
from nav_msgs.msg import OccupancyGrid
import math
from alpaca_navigation.cv import *
from alpaca_navigation.cv import prediction_cost
import alpaca_navigation.costs as costs
from alpaca_navigation.costs import Costmap
from alpaca_navigation.prediction.prediction_config import TIMESTEP as HST_DT
from scipy.ndimage import distance_transform_edt

class ControlFilter :
    def __init__ (self, alpha, x = False):
          self.x = x
          self.alpha = alpha 
          self.control = 0
    
    def get_control (self, new_command ):
         new_control = self.alpha * new_command + (1 - self.alpha) * self.control
        
         self.control =  new_control
         
         return self.control
    
    def reset(self):
        self.control = 0 


class SMMPPIController:
    def __init__(self,static_obs, device, params = None, verbose = True, costmap = None, use_sim_costs = False):
        if params is None: 
            weights = {

                "cov_x" : 1.0,
                "cov_z" : 0.01,
                "lambda_": 10,
                "goal_weight": 13000,
                "action_weight": 100,
                "heading_weight": 0,
                "steering_weight": 0,
                "steering_t_weight": 0,
                "sm_weight": 10,
                "costmap_weight": 1,
                "cv_weight": 100,
                "terminal_goal_weight": 100,
                "action_t_weight": 3000,
                "control_z_weight": 0,
                "control_x_weight": 0,
                "use_hst": False 

                
                }
        else:
            weights = params
        
        self.cov_x = weights.get("cov_x", 1.0)
        self.cov_z = weights.get("cov_z", 0.01)
        self.lambda_ = weights.get("lambda_", 10)
        self.goal_weight = weights["goal_weight"]
        self.action_weight = weights["action_weight"]
        self.heading_weight = weights["heading_weight"]
        self.steering_weight = weights["steering_weight"]
        self.steering_t_weight = weights ["steering_t_weight"]
        self.sm_weight = weights["sm_weight"]
        self.costmap_weight = weights["costmap_weight"]
        self.cv_weight = weights["cv_weight"]
        self.terminal_goal_weight = weights["terminal_goal_weight"]
        self.action_t_weight = weights["action_t_weight"]
        self.control_x_weight = weights ["control_x_weight"]
        self.control_z_weight = weights ["control_z_weight"]
        self.alpha_x = weights.get("alpha_x", 0.5)
        self.alpha_z = weights.get("alpha_z", 0.5)
        self.DT = weights.get ("DT", DT)
        self.horizon_length = weights.get ("horizon_length", HORIZON_LENGTH)
        self.num_samples = weights.get ("num_sample", NUM_SAMPLES)
        self.sigma_collision = weights.get ("sigma_collision", SIGMA_COLISION)
        self.beta_collision = weights.get ("beta_collision", 0.08)
        self.use_hst = weights.get("use_hst", False) # Option to use HST prediction
        self.decay_rate = weights.get ("decay_rate", -0.15)

        self.verbose = verbose 


        self.device = device
        if self.device != torch.device("cuda"):
            raise RuntimeError ("cuda not available, MPPI requires GPU acceleration")


        self.angular_alignment_threshold = ANGULAR_THRESHOLD   # Angular error threshold in radians
        self.goals = torch.tensor(GOALS, dtype=torch.float32).to(self.device)
        self.rollouts = torch.zeros((7, NUM_SAMPLES, 2)).to(self.device)
        self.costs = torch.zeros((7, NUM_SAMPLES, 2)).to(self.device)
        self.max_cycles = NUM_CYCLES
        self.s2_ego = torch.zeros((self.num_samples, 3)).to(self.device)
        self.sigma_h = SIGMA_H
        self.sigma_s = SIGMA_S
        self.sigma_r = SIGMA_R
        self.q_obs = Q_OBS


        self.current_goal_index = 0
        self.cycle_count = 0
        self.counter = 0
        self.goal = torch.tensor ([0,0]).to(self.device)
        self.agent_weights = {i: torch.tensor([0.1, 0.1, 0.0], dtype=torch.float32).to(self.device) for i in range(ACTIVE_AGENTS)}

        self.interacting_agents = []
        self.local_costmap = costmap

        # Initialize MPPI with the dynamics and cost functions
        cov = torch.eye(3, dtype=torch.float32).to(self.device)
        cov[0, 0] = self.cov_x
        cov[1, 1] = 10e-8 #0.001
        cov[2, 2] = self.cov_z

        self.horizon = self.horizon_length

        # Initialize Predictor
        if self.use_hst:
            self.latest_hst_predictions = {}
            print("HST Predictor Initialized in MPPI")

        self.mppi = mppi.KMPPI(
            self.dynamics,
            self.cost,
            3,  # State dimension
            cov,
            num_samples=self.num_samples,
            horizon=self.horizon_length,
            device=self.device,
            terminal_state_cost=self.terminal_cost,
            step_dependent_dynamics=True,

            u_min=torch.tensor([0.0, 0.0, -WMAX], dtype=torch.float32).to(self.device),
            u_max=torch.tensor([VMAX, 0.0, WMAX], dtype=torch.float32).to(self.device),
            
            lambda_ = self.lambda_, #1e-2,
            kernel = mppi.RBFKernel (sigma = 3.0),
            num_support_pts=self.horizon // 4, 

            noise_abs_cost = False, 
            u_per_command = HORIZON_LENGTH,
        )

        self.candidate_states = None
        self.candidate_costs = None
        self.last_cost_breakdown = {}
        self.last_cost_breakdown_tensors = {}
        self.publish_cost_breakdown = False

        self.prev_u = None

        self.x_filter = ControlFilter (self.alpha_x)
        self.z_filter = ControlFilter (self.alpha_z)

        self.use_sim_costs = use_sim_costs
        self.current_state = torch.tensor ([0,0,0]).to(self.device)

        
        

    def reset(self):
        print ("resetting mppi controller")
        self.mppi.reset()
    
    def set_goal (self, goal):
        prev_goal = self.goal
        self.goal = goal.to(self.device)

        
        heading_to_goal = torch.atan2(goal[1] - self.current_state[1], goal[0] - self.current_state[0])  # Shape: (N, T')
        heading_error = heading_to_goal - self.current_state[2]  # Shape
        
    
    def set_local_costmap(self, msg):
        cm = Costmap(msg)
        data = cm.data.copy()
        data[data != 100] = 0
        data[data == 100] = 1
        obstacles = data.astype(bool)
        edt = distance_transform_edt(~obstacles) * cm.resolution
        cm.edt_gpu = torch.tensor(edt, dtype=torch.float32, device=self.device)
        self.local_costmap = cm  # atomic: edt_gpu already set before assignment



    def compute_control(self, current_state, previous_robot_state, robot_velocity, agent_states, previous_agent_states,agent_velocities): 
        self.current_state = current_state
        self.previous_robot_state = previous_robot_state
        self.robot_velocity = robot_velocity
        
        self.agent_states = agent_states
        self.previous_agent_states = previous_agent_states
        self.agent_velocities = agent_velocities

        action = self.mppi.command(current_state, shift_nominal_trajectory= True)
        
        action [:,0]= torch.clamp (action[:,0], 0.0, VMAX)
        action [:,2]= torch.clamp (action[:,2], -WMAX, WMAX)


        action [0,0] = self.x_filter.get_control (action[0,0])
        action [0,2] = self.z_filter.get_control (action[0,2])
        self.mppi.u_init = action if action.dim() ==1 else action[0]

        rollouts = self.mppi.get_rollouts (current_state, num_rollouts = 1)
        costs = self.mppi.cost_total.squeeze(0)

        termination =  torch.linalg.norm(self.current_state[:2] - self.goal) < TERMINATION_TOLERANCE
        

        self.prev_u = action 
        return action, rollouts, costs, termination

    def update_hst_prediction(self, msg):
        """
        Updates the latest HST prediction from ROS message.
        Expected msg type: Float32MultiArray
        Layout: [Time, Agents, 2]
        """
        if not self.use_hst:
            return
        if hasattr(self, 'latest_hst_tensor') and self.latest_hst_tensor is not None:
                del self.latest_hst_tensor
                torch.cuda.empty_cache()
        try:
            if len(msg.layout.dim) != 3:
                return
                
            T = msg.layout.dim[0].size
            A = msg.layout.dim[1].size
            D = msg.layout.dim[2].size # Should be 2
            
            data = np.array(msg.data, dtype=np.float32)
            data = data.reshape((T, A, D))

            if np.allclose(data, np.zeros((1, 1, 2), dtype=np.float32)):
                self.latest_hst_tensor = None
                return

            # Convert to torch
            # hst_cost expects (1, 1, T, A, 2)
            tensor = torch.from_numpy(data).to(self.device)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            agent_idx_not_following = []

            for i in range (tensor.shape [3]):
                heading_to_agent = torch.atan2 (tensor[0,0,0,i,1] - self.current_state [1] ,   tensor[0,0,0,i,0] - self.current_state[0] )
                heading_to_agent = (heading_to_agent - self.current_state[2] + math.pi) % (2*math.pi) - math.pi

                if abs(heading_to_agent) <= np.deg2rad(120):
                    agent_idx_not_following.append (i)

            if agent_idx_not_following:
                human_tensor = tensor[:,:, :,agent_idx_not_following,:]
            else:
                human_tensor = None
            hst_ema_alpha = 0.4
            if (
                human_tensor is not None
                and getattr(self, 'latest_hst_tensor', None) is not None
                and human_tensor.shape == self.latest_hst_tensor.shape
            ):
                self.latest_hst_tensor = hst_ema_alpha * human_tensor + (1 - hst_ema_alpha) * self.latest_hst_tensor
            else:
                self.latest_hst_tensor = human_tensor
            
        except Exception as e:
            self.latest_hst_tensor = torch.zeros ((1,1,1,1,2)).to(self.device)
            print(f"Failed to update HST tensor: {traceback.format_exc()}")

    def cost(self, state: torch.Tensor, action: torch.Tensor, t) -> torch.Tensor:
        """
        Cost function for MPPI optimization.
        Args:
            state: (num_samples, 3) - States over the horizon.
            action: (num_samples, 3) - Actions over the horizon.

        Returns:
            cost: (num_samples) - Total cost for each sample.
        """

        return 0

    def dynamics(self, s: torch.Tensor, a: torch.Tensor, t=None) -> torch.Tensor:
        """
        Input:
        s: robot global state  (shape: BS x 3)
        a: robot action   (shape: BS x 3)  — [v (m/s), 0, omega (rad/s)]

        Output:
        next robot global state after executing action (shape: BS x 3)
        """
        dt = self.DT

        L = 0.494
        W = 0.364
        R_MIN = 0.47644

        v = a[:, 0]
        omega_cmd = a[:, 2]
        theta = s[:, 2]

        eps = 1e-6

        R = torch.abs(v) / (torch.abs(omega_cmd) + eps)
        spin_mask = R < R_MIN

        # This is the steering command calculated by ranger_ros2
        phi_inner = torch.sign(v * omega_cmd) * torch.atan(
            (L / 2.0) / (R + eps)
        )

        # Equivalent to ConvertInnerAngleToCentral()
        phi_abs = torch.abs(phi_inner)

        phi_central_abs = torch.atan(
            L * torch.sin(phi_abs)
            /
            (
                L * torch.cos(phi_abs)
                + W * torch.sin(phi_abs)
                + eps
            )
        )

        phi_central = torch.sign(phi_inner) * phi_central_abs

        dx = torch.zeros_like(s)

        # Dual Ackermann
        dx_ack_x = v * torch.cos(phi_central) * torch.cos(theta)
        dx_ack_y = v * torch.cos(phi_central) * torch.sin(theta)
        dx_ack_yaw = 2.0 * v * torch.sin(phi_central) / L

        # Spinning
        dx[:, 0] = torch.where(spin_mask, 0.0, dx_ack_x)
        dx[:, 1] = torch.where(spin_mask, 0.0, dx_ack_y)
        dx[:, 2] = torch.where(spin_mask, omega_cmd, dx_ack_yaw)

        return s + dx * dt


    def terminal_cost(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        
        goal_expanded = self.goal[None, :] # (num_particles, horizon, 3) 
        state_squeezed = state.squeeze() #  (num_particles, horizon, 3)

        terminal_goal_cost = costs.terminal_goal_cost(state, self.goal)

        goal_cost = costs.goal_progress_cost_naive (state, self.goal)

        action_cost = costs.action_cost_i_axis(action, self.prev_u) 

        action_cost_t = costs.action_cost_t_axis(action)


        costmap_cost = costs.costmap_cost_edt(state, self.local_costmap, sigma_collision = self.sigma_collision, beta_collision = self.beta_collision) if self.local_costmap is not None else torch.zeros (self.num_samples).to(self.device)

        if self.use_hst:
            cv_cost = self.hst_cost(state, decay_rate = self.decay_rate)

            if cv_cost is np.nan:
                cv_cost = torch.zeros(self.num_samples).to(self.device)
        else:
                cv_cost = torch.zeros(self.num_samples).to(self.device)

        steering_cost = costs.steering_cost_i_axis_naive (action, self.prev_u)

        steering_cost_t_axis = costs.steering_cost_t_axis_naive (action)

        heading_cost = costs.heading_cost(state, self.goal)

        control_z_cost = costs.control_cost_z (action)
        control_x_cost = costs.control_cost_x (action)

        goal_weight =  self.goal_weight #13000 
        action_weight = self.action_weight#100
        heading_weight = self.heading_weight #0 
        steering_weight = self.steering_weight #1000
        steering_t_weight = self.steering_t_weight
        control_z_weight = self.control_z_weight 
        control_x_weight = self.control_x_weight 

        sm_weight = self.sm_weight #10
        costmap_weight = self.costmap_weight # 1 
        cv_weight = self.cv_weight #100
        terminal_goal_weight = self.terminal_goal_weight # 100
        action_t_weight = self.action_t_weight #3000
        workspace_bounds_cost = costs.workspace_bounds_cost(
            state,
            action=action,
            margin=0.55,
            tau=0.25,
            inside_weight=2,
            outside_weight=120,
            speed_weight=4,
        )

        cost = goal_weight*(goal_cost)  + terminal_goal_cost * terminal_goal_weight \
            + action_weight*action_cost   + action_t_weight * action_cost_t \
                + steering_weight * steering_cost \
                    + steering_cost_t_axis * steering_t_weight \
                        + heading_weight * heading_cost \
                                + cv_weight * cv_cost \
                                    + costmap_weight * costmap_cost \
                                        + self.control_z_weight * control_z_cost + self.control_x_weight * control_x_cost  #+ workspace_bounds_cost * 0
        
        ###### for visualization ######
        self.candidate_states = state_squeezed
        self.candidate_costs = cost
        ##############################

        if self.publish_cost_breakdown:
            self.last_cost_breakdown_tensors = {
                "goal":          (torch.mean(goal_cost) * goal_weight).detach(),
                "terminal_goal": (torch.mean(terminal_goal_cost) * terminal_goal_weight).detach(),
                "action":        (torch.mean(action_cost) * action_weight).detach(),
                "action_t":      (torch.mean(action_cost_t) * action_t_weight).detach(),
                "steering":      (torch.mean(steering_cost) * steering_weight).detach(),
                "steering_t":    (torch.mean(steering_cost_t_axis) * steering_t_weight).detach(),
                "heading":       (torch.mean(heading_cost) * heading_weight).detach(),
                "cv":            (torch.mean(cv_cost) * cv_weight).detach(),
                "costmap":       (torch.mean(costmap_cost) * costmap_weight).detach(),
                "control_z":     (torch.mean(control_z_cost) * control_z_weight).detach(),
                "control_x":     (torch.mean(control_x_cost) * control_x_weight).detach(),
                "total":         torch.mean(cost).detach(),
            }

        return  cost

    def hst_cost(self, state: torch.Tensor, decay_rate = -0.15 ) -> torch.Tensor:
        """
        Computes cost based on HST predictions using prediction_cost from cv.py.
        """
        if getattr(self, 'latest_hst_tensor', None) is None:
                return torch.zeros(self.num_samples).to(self.device)

        # Slice state to match prediction length and timestep stride
        # MPPI DT = self.DT (0.1), HST DT = HST_DT (0.4)
        # stride = HST_DT / DT = 4
        # offset = stride - 1 (index 3 corresponds to t=0.4s)

        stride = int(round(HST_DT / self.DT))
        offset = stride - 1
        # Slice state with stride
        state_strided = state[:, :, offset::stride, :]
        T_state = state_strided.shape[2]
        T_pred = self.latest_hst_tensor.shape[2]
        T_common = min(T_state, T_pred)
        if T_common == 0:
                return torch.zeros(self.num_samples).to(self.device)

        # Slice state length
        state_sliced = state_strided[:, :,:T_common, :]
        
        # Slice prediction length to match
        # Prediction tensor is (1, 1, T_pred, N_agents, 2)
        pred_sliced = self.latest_hst_tensor[:, :, :T_common, :, :]

        # Prepare logits (1 mode with probability 1 -> log(1)=0)
        logits = torch.zeros((1, 1)).to(self.device)
        
        # input state: (N, T_common, 3)
        # input prediction: (1, 1, T_common, N_agents, 2)
        cost = prediction_cost (state_sliced, pred_sliced, logits, decay_rate = self.decay_rate)
        return cost

    def get_candidate_states_and_costs(self):
        return self.candidate_states, self.candidate_costs
        

    def get_interacting_agents(self):
        self.interacting_agents = []
        robot_state = self.current_state
        for idx, agent_state in self.agent_states.items():
            direction_to_agent = torch.arctan2(agent_state[1] - self.current_state[1], agent_state[0] - self.current_state[0])
            distance_to_agent = torch.norm(agent_state[:2] - self.current_state[:2])

            if self.current_goal_index == 0:
                robot_direction = np.pi/2
            elif self.current_goal_index == 1:
                robot_direction = -np.pi/2

            relative_angle = torch.rad2deg(direction_to_agent -robot_direction)
            relative_angle = (relative_angle + 180) % 360 - 180  
            
            if -90 <= relative_angle <= 90 and distance_to_agent < 2:  
                self.interacting_agents.append(idx)
                self.agent_weights[idx] = (1/distance_to_agent)


    def safety_check (self, action:torch.Tensor):
        d_stop = ROBOT_RADIUS + 0.05
        d_resume = d_stop + SAFE_DISTANCE


        pos_in_costmap_frame = (self.current_state [:2] - torch.tensor ([self.local_costmap.origin_x, self.local_costmap.origin_y], device = self.current_state.device) )
        grid_x = torch.clamp ((pos_in_costmap_frame [0] / self.local_costmap.resolution).long(), 0 , self.local_costmap.width - 1)
        grid_y = torch.clamp ((pos_in_costmap_frame[1] / self.local_costmap.resolution).long(),  0, self.local_costmap.height -1 ) 

        obstacles = self.local_costmap.data.copy()
        obstacles [obstacles != 100] = 0 
        obstacles [obstacles == 100] = 1
        obstacles = obstacles.astype (bool)

        r,c = grid_y, grid_x

        dist, (ri,ci) = distance_transform_edt (~obstacles, return_indices = True)
        
        nearest = (int(ri[r, c])*self.local_costmap.resolution + self.local_costmap.origin_x, int(ci[r, c])*self.local_costmap.resolution + self.local_costmap.origin_y)

        d = float (dist[r,c]) * self.local_costmap.resolution


        safety_factor = np.clip ( (d - d_stop) / (d_resume - d_stop), 0, 1)


        softplus = torch.nn.Softplus (beta = self.beta_collision) 
        costmap_cost = self.costmap_weight * softplus ( torch.tensor(SAFE_DISTANCE - d) / self.sigma_collision).item()

        
        print (f'nearest obstacle is {d:.2f} away, costmap at this location is {costmap_cost:.2f}, top speed recduced to {0.6 * safety_factor:.2f} m/s ')


        return action
