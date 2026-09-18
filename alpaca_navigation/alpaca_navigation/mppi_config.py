import numpy as np
import torch
from datetime import datetime

NOW = datetime.now()

VMAX = 0.7 #0.7
WMAX = 1.0

AMAX_X = 0.5
AMAX_Z = 1.5



DT = 0.2  # 0.1 #0.4
HORIZON_LENGTH =  30 #30 #16
MAX_AGENT_NUM = 14
ACTIVE_AGENTS = 1
AGENT_GOALS = np.array([[0.0,-2.35], [0.0, 2.75]]) #goal of the agents, only required for evaluation metrics
HZ = 0.05 #frequency at which the controller should run
ANCA_CONTROL_POINTS = 5
USE_TERMINAL_COST = True
ANGULAR_THRESHOLD = 0.3

STATIC_OBSTACLES = []

ROBOT_RADIUS = 0.45
# SAFE_DISTANCE = 0.2 #third floor 
SAFE_DISTANCE = 0.2 
SIGMA_COLISION = 15  ### how sharp costmap cost rises near safety distance ### 


NUM_SAMPLES = 5000 
NEED_ODOM = False 
NEED_LASER = False
HUMAN_FRAME = "human" # for the TF transform from the motion capture
RADIUS = 0.2
NUM_CYCLES = 10 # Number of time to cycle between the goals
# GOALS = np.array([[ 44.97074351139287,-35.43138448289912], [0.6466922193980356, -27.258705145313982]]) # waypoints on third floor (x,y)
GOALS = np.array([[ -2.1434060883598445, 15.083954186331338, 0.7106529884135117, 0.7035426995278576], [11,-21   , -0.5128752473184045, 0.858463150454395], [-0.30585165375508616,-16.591245073794813, -0.9999892887441735, 0.004628433527884085]]) # waypoints on first floor, (x,y, quat_z, quat_w)
# GOALS = np.array([[1.0,0.0],[-2.0, 0.0]])
REPEAT_GOALS = True
TERMINATION_TOLERANCE = 0.5

ACKNOWLEDGE_RADIUS = 2.0
SIGMA_H = 0.8
SIGMA_S = 0.2 #0.3
SIGMA_R = 0.25

MIN_TURN_RADIUS = 0.4764

MS_COST = 1000 # cost to switch from spot turn to dual ackermann

Q_OBS = 10e3 #Use with terminal cost