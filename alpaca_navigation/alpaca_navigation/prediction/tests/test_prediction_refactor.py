import sys
import os
import numpy as np
import torch
import unittest
from collections import deque

# 1. Setup path to allow importing alpaca_navigation package
# Assumes script is at: .../alpaca_navigation/prediction/tests/test_prediction_refactor.py
# Repo root is at:      .../ (3 levels up)
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Mock configuration if not available in python path
# We try to mock the location that prediction.py will try to import
try:
    from alpaca_navigation.prediction.prediction_config import *
except ImportError:
    # Fallback mocks for standalone testing without full env
    HISTORY_LENGTH = 7
    NODE_SKIP_INDEX = 8
    MAX_AGENT_NUM = 14
    WINDOW_LENGTH = 20
    ACTIVE_AGENT_NUM = 2
    ROBOT_AS_HUMAN = True
    HST_CKPT_PATH = "mock_path"

# Mock dependencies
# Mock dependencies
# We forcefully mock tensorflow to avoid loading the heavy real library and to allow using our MockModel
class MockTF:
    class config:
        @staticmethod
        def list_physical_devices(type): return []
        class experimental:
            @staticmethod
            def set_memory_growth(dev, val): pass
    class train:
        class Checkpoint:
            def __init__(self, **kwargs): pass
            def restore(self, path): return self
            def assert_existing_objects_matched(self): pass
    class MockTensor:
        def __init__(self, val):
            self.val = val
        def numpy(self):
            return self.val
        @property
        def shape(self):
            return self.val.shape

    @staticmethod
    def convert_to_tensor(val, dtype=None): 
        return MockTF.MockTensor(val)
    @staticmethod
    def device(dev): 
        class Context:
            def __enter__(self): pass
            def __exit__(self, a,b,c): pass
        return Context()
    float32 = "float32"

sys.modules['tensorflow'] = MockTF()
import tensorflow as tf # This will now import the mock

# Mock existing module imports for the unit under test
sys.modules['alpaca_navigation.prediction.prediction_config'] = type('config', (), {
    'HISTORY_LENGTH': HISTORY_LENGTH,
    'NODE_SKIP_INDEX': NODE_SKIP_INDEX,
    'MAX_AGENT_NUM': MAX_AGENT_NUM,
    'WINDOW_LENGTH': WINDOW_LENGTH,
    'HST_CKPT_PATH': type('Path', (), {'as_posix': lambda self: 'mock'})(),
    'ROBOT_AS_HUMAN': True,
    'ACTIVE_AGENT_NUM': ACTIVE_AGENT_NUM
})

sys.modules['alpaca_navigation.prediction.human_scene_transformer.model'] = type('m', (), {'model': type('M', (), {'HumanTrajectorySceneTransformer': None})})
sys.modules['alpaca_navigation.prediction.human_scene_transformer.config'] = type('c', (), {'hst_config': None})

class MockModel:
    def __call__(self, inputs, training=False):
        # inputs is dict of tensors
        # Return mock outputs
        # agent_position_pred shape: (Batch, Agents, Modes, Time, 2)
        # agent_position_logits shape: (Batch, Modes)
        
        # Batch=1, Agents=MAX_AGENT_NUM, Time=WINDOW_LENGTH, Modes=5, XY=2
        agents = inputs['agents/position'].shape[1]
        modes = 5
        time = inputs['agents/position'].shape[2]
        
        # Consistent with prediction.py comment: (Batch, Max_Agents, Window_Length, Modes, XY)
        pred = np.zeros((1, agents, time, modes, 2), dtype=np.float32)
        logits = np.ones((1, modes), dtype=np.float32)
        
        full_pred = {
            'agents/position': tf.convert_to_tensor(pred),
            'mixture_logits': tf.convert_to_tensor(logits)
        }
        return full_pred, None

sys.modules['alpaca_navigation.prediction.human_scene_transformer.infer'] = type('inf', (), {'init_model': lambda: MockModel()})

# Import Class Under Test
from alpaca_navigation.prediction.prediction import HSTPredictor

class TestHSTPredictor(unittest.TestCase):
    def test_variable_agents(self):
        predictor = HSTPredictor()
        
        # 1. Warmup with robot only
        traj, _ = predictor.update_and_predict(human_poses={}, robot_pose=[0,0])
        self.assertIsNone(traj)
        
        # 2. Add humans
        poses = {1: [1,1], 2: [2,2]}
        traj, _ = predictor.update_and_predict(human_poses=poses, robot_pose=[1,0])
        print("Step 2 (2 Humans) Output Shape:", traj.shape)
        
        # 3. Dropout one human
        poses = {1: [2,2]} # 2 disappeared
        traj, _ = predictor.update_and_predict(human_poses=poses, robot_pose=[2,0])
        print("Step 3 (Human 2 lost) Output Shape:", traj.shape)
        
        # 4. Reappear
        poses = {1: [3,3], 2: [4,4]} # 2 back
        traj, _ = predictor.update_and_predict(human_poses=poses, robot_pose=[3,0])
        print("Step 4 (Human 2 back) Output Shape:", traj.shape)
        
        # Check tensor internal reconstruction (requires accessing private members or mocks)
        # Here we just verify it runs and returns tensor
        self.assertTrue(isinstance(traj, torch.Tensor))
        
        # Verify shape.
        # Current implementation returns only selected human agents, not a
        # fully padded MAX_AGENT_NUM axis.
        expected_time = WINDOW_LENGTH - (HISTORY_LENGTH + 1)
        self.assertEqual(traj.shape[2], expected_time)
        self.assertEqual(traj.shape[3], len(poses))

if __name__ == '__main__':
    unittest.main()
