#! /usr/bin/env python

import os
from pathlib import Path
import numpy as np
import torch
import time

from collections import deque
import collections

# inference pipeline
from alpaca_navigation.prediction.prediction_config import *
from alpaca_navigation.prediction.backends import load_predictor_class


def _find_workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").is_dir() and (parent / "install").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]

WORKSPACE_ROOT = _find_workspace_root()


def _default_eqmotion_bundle() -> Path:
    models_dir = WORKSPACE_ROOT / "models"
    if (models_dir / "best_models_ade_eqmotion5.pth").exists() and (
        (models_dir / "config_eqmotion5.yaml").exists()
        or (models_dir / "config.json").exists()
        or (models_dir / "config.yaml").exists()
    ):
        return models_dir

    root_bundle = WORKSPACE_ROOT
    if (root_bundle / "best_models_ade_eqmotion5.pth").exists() and (
        (root_bundle / "config_eqmotion5.yaml").exists()
        or (root_bundle / "config.json").exists()
        or (root_bundle / "config.yaml").exists()
    ):
        return root_bundle

    models_zip = models_dir / "eqmotion_joint_K20_BS100_LR0.001_zara1_s0.zip"
    if models_zip.exists():
        return models_zip

    return WORKSPACE_ROOT / "eqmotion_joint_K20_BS100_LR0.001_zara1_s0.zip"


DEFAULT_EQMOTION_BUNDLE = _default_eqmotion_bundle()


def _default_autobots_config() -> Path:
    models_dir = WORKSPACE_ROOT / "models"
    for name in ("config_autobots_da.yaml", "config_autobots.yaml", "config _autobots.yaml"):
        config_path = models_dir / name
        if config_path.exists():
            return config_path
    return models_dir / "config_autobots.yaml"


DEFAULT_AUTOBOTS_CHECKPOINT = WORKSPACE_ROOT / "models" / "best_models_ade_autobots_da.pth"
DEFAULT_AUTOBOTS_CONFIG = _default_autobots_config()
DEFAULT_MOFLOW_CHECKPOINT = WORKSPACE_ROOT / "models" / "best_models_ade_student20_da.pth"
DEFAULT_MOFLOW_CONFIG = WORKSPACE_ROOT / "models" / "config_student20_da.yaml"
DEFAULT_MOFLOW_EVAL_CONFIG = WORKSPACE_ROOT / "models" / "moflow_joint_student_eval.yaml"

class HSTPredictor:
    def __init__(self):
        # The HST attention stack uses rank-6 attention scores on CPU. With
        # oneDNN enabled, TensorFlow can route softmax to an MKL kernel that
        # rejects tensors with rank > 5, which breaks inference at runtime.
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import tensorflow as tf
        from alpaca_navigation.prediction.human_scene_transformer.model import model as hst_model
        from alpaca_navigation.prediction.human_scene_transformer import infer

        self.tf = tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"HSTPredictor: Found {len(gpus)} GPUs. TensorFlow will use them.")
            except RuntimeError as e:
                print(e)
        else:
            print("HSTPredictor: No GPUs found. Using CPU.")

        hst_params = getattr(infer, "hst_model_param", None)
        default_history_steps = globals().get("HISTORY_LENGTH", 7)
        default_num_steps = globals().get("WINDOW_LENGTH", 20)
        default_timestep = globals().get("TIMESTEP", 0.4)
        self.hst_num_history_steps = int(
            getattr(hst_params, "num_history_steps", default_history_steps)
        )
        self.hst_num_steps = int(getattr(hst_params, "num_steps", default_num_steps))
        self.hst_timestep = float(getattr(hst_params, "timestep", default_timestep))
        default_timer_hz = float(globals().get("PREDICTION_TIMER_HZ", 10.0))
        self.hst_history_stride = max(1, round(self.hst_timestep * default_timer_hz))

        # History buffers: list of snapshots
        # Snapshot: dict { agent_id: [x, y] }
        # Max length covers the oldest required history sample plus current.
        self.buffer_len = self.hst_num_history_steps * self.hst_history_stride + 1
        self.human_history_buffer = deque(maxlen=self.buffer_len)
        self.robot_history_buffer = deque(maxlen=self.buffer_len)
        self.human_history_buffer_hst = deque(maxlen=self.buffer_len) # For HST, we maintain a separate buffer with robot-relative positions
        self.robot_history_buffer_hst = deque(maxlen=self.buffer_len) # For HST, we maintain a separate buffer with robot-relative positions


        
        ### hst checkpoint
        self.model: hst_model.HumanTrajectorySceneTransformer = infer.init_model()
        _checkpoint_path = HST_CKPT_PATH.as_posix()
        checkpoint_mngr = tf.train.Checkpoint(model=self.model)
        restore_status = checkpoint_mngr.restore(_checkpoint_path)
        if hasattr(restore_status, "expect_partial"):
            restore_status.expect_partial()
        if hasattr(restore_status, "assert_existing_objects_matched"):
            restore_status.assert_existing_objects_matched()

        def inference_fn(batch):
            return self.model(batch, training=False)

        if hasattr(tf, "function") and hasattr(tf, "TensorSpec"):
            inference_fn = tf.function(
                input_signature=[
                    {
                        'agents/position': tf.TensorSpec(shape=[1, MAX_AGENT_NUM, self.hst_num_steps, 2], dtype=tf.float32),
                        'robot/position': tf.TensorSpec(shape=[1, self.hst_num_steps, 2], dtype=tf.float32),
                    }
                ]
            )(inference_fn)
        
        self.inference_fn = inference_fn
        print(f"HST Predictor Initialized. Checkpoint: {_checkpoint_path}")
    
    

    def update_and_predict(self, human_poses: dict, robot_pose: list) -> torch.Tensor:
        """
        Updates history buffers and returns the most likely prediction.
        
        Args:
            human_poses: Dictionary {agent_id: [x, y]} containing current human positions.
                         Can be lists, numpy arrays, or torch tensors.
            robot_pose: List or array [x, y] of the robot's current position.
            
        Returns:
            torch.Tensor: Predicted trajectories for the most likely mode.
            Shape: (1, Modes, Future_Time, Agents, 2) -> filtered to (1, Agents, Future_Time, 2) usually?
            Note: User original code returned (1, AGENT_NUM, WINDOW_LENGTH, 2) approx (before mode selection).
            We returns best mode prediction.
        """
        
        def to_cpu_numpy(val):
            if isinstance(val, torch.Tensor):
                return val.detach().cpu().numpy()
            if isinstance(val, list):
                return np.array(val)
            return val

        # 1. Update History Buffers (Snapshots)
        
        # Human Snapshot
        current_human_snapshot = {}
        for agent_id, pos in human_poses.items():
            pos_np = to_cpu_numpy(pos)[:2]
            current_human_snapshot[agent_id] = pos_np + 10.0
        self.human_history_buffer.append(current_human_snapshot)
        
        
        # Robot Snapshot
        robot_pose_arr = to_cpu_numpy(robot_pose)
        robot_xy = robot_pose_arr[:2]
        robot_yaw = float(robot_pose_arr[2]) if robot_pose_arr.shape[0] > 2 else 0.0
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)

        def world_to_robot_frame(world_xy: np.ndarray, robot_xy_: np.ndarray) -> np.ndarray:
            dx = float(world_xy[0] - robot_xy_[0])
            dy = float(world_xy[1] - robot_xy_[1])
            return np.array(
                [cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy],
                dtype=np.float32,
            )

        self.human_history_buffer_hst = deque(maxlen=self.buffer_len) # Clear and rebuild HST buffer with robot-relative positions
        for snapshot in self.human_history_buffer:
            new_snapshot = {}
            for agent_id in snapshot:
                # Keep +10 offset in robot frame while making relative coordinates yaw-aware.
                human_world_xy = snapshot[agent_id] - 10.0
                human_robot_xy = world_to_robot_frame(human_world_xy, robot_xy)
                new_snapshot[agent_id] = human_robot_xy + 10.0
            self.human_history_buffer_hst.append(new_snapshot)

        self.robot_history_buffer.append(robot_xy)
        self.robot_history_buffer_hst = deque(maxlen=self.buffer_len) # Clear and rebuild HST buffer with robot-relative positions
        for pos in self.robot_history_buffer:
            self.robot_history_buffer_hst.append(world_to_robot_frame(pos, robot_xy))
            

        # 2. Select Agents for Tensor
        # Robot is always at index 0.
        # Humans fill indices 1 to MAX_AGENT_NUM - 1.
        
        # Collect all agent IDs currently in the buffer window
        present_agents_ids = set()
        all_agent_ids = set()
        
        # Check presence in the latest snapshot (current)
        present_agents_ids.update(current_human_snapshot.keys())
        
        # Check all history for potential filling if slots are available
        for snapshot in self.human_history_buffer_hst:
            all_agent_ids.update(snapshot.keys())

        # Selection Strategy:
        # 1. Current agents (sorted for determinism)
        # 2. Recent agents (if space allows) 
        
        sorted_present = sorted(list(present_agents_ids))
        sorted_all = sorted(list(all_agent_ids))
        
        selected_agents = []
        
        # Add present agents first
        for aid in sorted_present:
            if len(selected_agents) < MAX_AGENT_NUM - 1: # Reserve 1 slot for robot
                selected_agents.append(aid)
        
        # Add historical agents if space remains
        # for aid in sorted_all:
        #      if len(selected_agents) < MAX_AGENT_NUM - 1:
        #          if aid not in selected_agents:
        #              selected_agents.append(aid)
        #      else:
        #          break
                 
        # 3. Construct Input Tensors
        # Shape: (Batch, Agents, Time, 2)
        # Note: Original code used (1, MAX_AGENT_NUM, WINDOW_LENGTH, 2)
        # Time dimension: 0..HISTORY_LENGTH corresponds to t_indices in HST
        # We construct history.
        
        agent_position_map_np = np.full((1, MAX_AGENT_NUM, self.hst_num_steps, 2), np.nan, dtype=np.float32)

        
        
        # Helper to get position from buffer at logical index t_hist (0..HISTORY_LENGTH)
        # t_hist=0 means oldest history point needed. t_hist=HISTORY_LENGTH means current.
        # We sample backwards from current.
        # Logical Time: t (0 to HISTORY_LENGTH)
        # Distance from end of deque: (HISTORY_LENGTH - t) * NODE_SKIP_INDEX
        
        # Helper: Fill Robot (Index 0)
        # Robot always present in our buffer logic
        robot_idx = 0
        current_buffer_idx = len(self.robot_history_buffer_hst) - 1
        
        for t in range(self.hst_num_history_steps + 1):
            offset = (self.hst_num_history_steps - t) * self.hst_history_stride
            buffer_idx = current_buffer_idx - offset
            
            if buffer_idx >= 0:
                pos = self.robot_history_buffer_hst[buffer_idx]
                agent_position_map_np[:, robot_idx, t, :] = pos
                
        # Helper: Fill Humans (Indices 1+)
        tensor_idx_list = []
        for i, agent_id in enumerate(selected_agents):
            tensor_idx = i + 1 # Start after robot
            tensor_idx_list.append(tensor_idx)
            
            current_buffer_idx = len(self.human_history_buffer_hst) - 1
            
            for t in range(self.hst_num_history_steps + 1):
                offset = (self.hst_num_history_steps - t) * self.hst_history_stride
                buffer_idx = current_buffer_idx - offset
                
                if buffer_idx >= 0:
                    snapshot = self.human_history_buffer_hst[buffer_idx]
                    if agent_id in snapshot:
                        pos = snapshot[agent_id]
                        agent_position_map_np[:, tensor_idx, t, :] = pos
                    # Else remain 0 (padding)
        if len(tensor_idx_list) == 0:
            return None, None
        
        # 4. Inference Preparation
        tf = self.tf
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        
        with tf.device(device):
            agent_position_map = tf.convert_to_tensor(agent_position_map_np, dtype=tf.float32)
            # print(f"Input agent_position_map shape: {agent_position_map.shape}")
            # Robot input map is dummy as per original code logic, robot is embedded in agent map
            robot_position_map_np = np.full((1, self.hst_num_steps, 2), -100.0)
            
            input_batch = {
                'agents/position': agent_position_map,
                'robot/position': tf.convert_to_tensor(robot_position_map_np, dtype=tf.float32),
            }

        # 5. Inference
        full_pred, output_batch = self.inference_fn(input_batch)
        
        agent_position_pred = full_pred['agents/position'] 
        agent_position_logits = full_pred['mixture_logits'] 
        # 6. Extract Most Likely Mode and Format Output
        logits = agent_position_logits.numpy()
        if logits.ndim == 2:
             best_mode_idx = np.argmax(logits[0])
        else:
             best_mode_idx = np.argmax(logits)
             
        pred_np = agent_position_pred.numpy()
        
        # Transpose: (Batch, Agents, Window, Modes, XY) -> (Batch, Modes, Window, Agents, XY)
        # Original code transpose: (0, 3, 2, 1, 4)
        # Input: (B, A, T, M, D)
        # Target: (B, M, T, A, D)
        pred_np = np.transpose(pred_np, (0, 3, 2, 1, 4))
        
        # Slice Future
        cutoff_index = self.hst_num_history_steps + 1

        pred_np = pred_np[:, :, cutoff_index:, :, :]
        
        # Select Best Mode
        best_pred = pred_np[:, best_mode_idx, None, :, :, :]
        best_pred = best_pred[:, :, :, torch.tensor(tensor_idx_list), :]
        if len(tensor_idx_list) == 1:
            best_pred = best_pred[:, :, :, None, :]
        # Remove Offset
        best_pred = best_pred - 10.0
        # Rotate back to world-axis relative vectors so downstream addition with robot x,y remains valid.
        best_pred_x = best_pred[..., 0].copy()
        best_pred_y = best_pred[..., 1].copy()
        best_pred[..., 0] = cos_yaw * best_pred_x - sin_yaw * best_pred_y
        best_pred[..., 1] = sin_yaw * best_pred_x + cos_yaw * best_pred_y
        
        # Convert to Torch
        out_tensor = torch.from_numpy(best_pred)
        if torch.cuda.is_available():
            out_tensor = out_tensor.cuda()
            
        return out_tensor, selected_agents


class EqMotionPredictorAdapter:
    """Adapter that makes EqMotion look like the existing HST predictor output."""

    def __init__(self, bundle_path=None):
        EqMotionPredictor = load_predictor_class('eqmotion')

        resolved_bundle = Path(bundle_path) if bundle_path else DEFAULT_EQMOTION_BUNDLE
        self.predictor = EqMotionPredictor(resolved_bundle)
        self.obs_len = int(self.predictor.obs_len)
        self.pred_len = int(self.predictor.pred_len)
        self.max_scene_agents = max(1, min(int(self.predictor.max_agents), MAX_AGENT_NUM - 1))
        self.history_stride = max(1, round(TIMESTEP * PREDICTION_TIMER_HZ))
        self.buffer_len = (self.obs_len - 1) * self.history_stride + 1
        self.human_history_buffer = deque(maxlen=self.buffer_len)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history_fill_strategy = str(getattr(self.predictor, "history_fill_strategy", "carry")).lower()
        self.min_target_history_points = 1

    @staticmethod
    def _to_xy_array(val):
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()[:2].astype(np.float32)
        if isinstance(val, np.ndarray):
            return val[:2].astype(np.float32)
        return np.asarray(val[:2], dtype=np.float32)

    @staticmethod
    def _select_mode(pred_modes: torch.Tensor, history_xy: np.ndarray) -> np.ndarray:
        """Choose the medoid EqMotion mode using only model output diversity."""
        del history_xy
        pred_np = pred_modes.detach().cpu().numpy()
        if pred_np.ndim != 3:
            raise ValueError(f"Expected target prediction shape (M,T,2), got {pred_np.shape}")

        pairwise_dist = np.linalg.norm(
            pred_np[:, None, :, :] - pred_np[None, :, :, :],
            axis=-1,
        ).mean(axis=-1)
        medoid_idx = int(np.argmin(pairwise_dist.mean(axis=1)))
        return pred_np[medoid_idx].astype(np.float32)

    def _sample_agent_history(self, agent_id: int) -> np.ndarray:
        """Sample one agent from the rolling snapshot buffer, HST-style."""
        history = np.full((self.obs_len, 2), np.nan, dtype=np.float32)
        current_buffer_idx = len(self.human_history_buffer) - 1

        for t in range(self.obs_len):
            offset = (self.obs_len - 1 - t) * self.history_stride
            buffer_idx = current_buffer_idx - offset
            if buffer_idx < 0:
                continue
            snapshot = self.human_history_buffer[buffer_idx]
            if agent_id in snapshot:
                history[t] = snapshot[agent_id]

        return history

    @staticmethod
    def _history_mask(history_xy: np.ndarray) -> np.ndarray:
        return (~np.isnan(history_xy[:, 0]) & ~np.isnan(history_xy[:, 1])).astype(bool)

    def _fill_history(self, history_xy: np.ndarray) -> np.ndarray:
        if self.history_fill_strategy == "zero":
            return np.nan_to_num(history_xy.copy(), nan=0.0).astype(np.float32)

        filled = history_xy.copy()
        valid_mask = self._history_mask(filled)
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            return np.zeros_like(filled, dtype=np.float32)

        carry = filled[int(valid_indices[0])].copy()
        for t in range(filled.shape[0]):
            if valid_mask[t]:
                carry = filled[t].copy()
            else:
                filled[t] = carry

        backfill = filled.copy()
        carry = backfill[int(valid_indices[-1])].copy()
        for t in range(backfill.shape[0] - 1, -1, -1):
            if valid_mask[t]:
                carry = backfill[t].copy()
            else:
                backfill[t] = carry
        return np.nan_to_num(backfill, nan=0.0).astype(np.float32)

    def update_and_predict(self, human_poses: dict, robot_pose: list):
        del robot_pose
        current_humans = {}

        for agent_id, pos in human_poses.items():
            pos_np = self._to_xy_array(pos)
            current_humans[int(agent_id)] = pos_np
        self.human_history_buffer.append(current_humans)

        # Build one joint scene with all valid agents (up to max_scene_agents)
        scene_entries = []
        for agent_id in sorted(current_humans.keys()):
            history_np = self._sample_agent_history(agent_id)
            valid_mask = self._history_mask(history_np)
            valid_count = int(np.count_nonzero(valid_mask))
            if valid_count < self.min_target_history_points:
                continue
            scene_entries.append((agent_id, history_np, valid_mask))
            if len(scene_entries) >= self.max_scene_agents:
                break

        if not scene_entries:
            return None, None

        history_xy = np.stack([hist for _, hist, _ in scene_entries], axis=0).astype(np.float32)
        history_mask = np.stack([mask for _, _, mask in scene_entries], axis=0).astype(bool)
        scene_tensor = torch.from_numpy(np.stack([self._fill_history(hist) for hist in history_xy], axis=0))
        pred = self.predictor.predict(
            scene_tensor,
            num_valid=len(scene_entries),
            history_mask=torch.from_numpy(history_mask),
        )
        # pred shape: (N_agents, num_modes, T_pred, 2)

        predicted_rel = []
        selected_agents = []
        for i, (agent_id, history_np, _history_mask) in enumerate(scene_entries):
            history_filled = self._fill_history(history_np)
            best_pred = self._select_mode(pred[i], history_filled)
            predicted_rel.append(best_pred)
            selected_agents.append(agent_id)

        pred_np = np.stack(predicted_rel, axis=1).astype(np.float32)
        out_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).to(self.device)
        return out_tensor, selected_agents


class AutoBotsPredictorAdapter:
    """Adapter that exposes standalone AutoBots inference through update_and_predict."""

    def __init__(self, checkpoint_path=None, config_path=None):
        AutoBotsPredictor = load_predictor_class('autobots')

        resolved_config = Path(config_path) if config_path else DEFAULT_AUTOBOTS_CONFIG
        resolved_checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_AUTOBOTS_CHECKPOINT

        self.predictor = AutoBotsPredictor(
            checkpoint_path=resolved_checkpoint,
            config_path=resolved_config,
        )
        self.obs_len = int(self.predictor.obs_len)
        self.pred_len = int(self.predictor.pred_len)
        self.num_agents = max(2, int(self.predictor.max_agents))
        self.num_other_agents = self.num_agents - 1
        self.max_scene_humans = min(self.num_other_agents, MAX_AGENT_NUM - 1)
        self.history_stride = max(1, round(TIMESTEP * PREDICTION_TIMER_HZ))
        self.buffer_len = (self.obs_len - 1) * self.history_stride + 1
        self.human_history_buffer = deque(maxlen=self.buffer_len)
        self.robot_history_buffer = deque(maxlen=self.buffer_len)
        self.device = self.predictor.device
        self.min_target_history_points = 2

    @staticmethod
    def _to_xy_array(val):
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()[:2].astype(np.float32)
        if isinstance(val, np.ndarray):
            return val[:2].astype(np.float32)
        return np.asarray(val[:2], dtype=np.float32)

    def _sample_agent_history(self, agent_id: int):
        history = np.zeros((self.obs_len, 2), dtype=np.float32)
        mask = np.zeros((self.obs_len,), dtype=bool)
        current_buffer_idx = len(self.human_history_buffer) - 1

        for t in range(self.obs_len):
            offset = (self.obs_len - 1 - t) * self.history_stride
            buffer_idx = current_buffer_idx - offset
            if buffer_idx < 0:
                continue
            snapshot = self.human_history_buffer[buffer_idx]
            if agent_id in snapshot:
                history[t] = snapshot[agent_id]
                mask[t] = True
        return history, mask

    def _sample_robot_history(self):
        history = np.zeros((self.obs_len, 2), dtype=np.float32)
        mask = np.zeros((self.obs_len,), dtype=bool)
        current_buffer_idx = len(self.robot_history_buffer) - 1

        for t in range(self.obs_len):
            offset = (self.obs_len - 1 - t) * self.history_stride
            buffer_idx = current_buffer_idx - offset
            if buffer_idx < 0:
                continue
            history[t] = self.robot_history_buffer[buffer_idx]
            mask[t] = True
        return history, mask

    @staticmethod
    def _carry_fill(history_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
        filled = history_xy.copy()
        valid_indices = np.flatnonzero(mask)
        if valid_indices.size == 0:
            return filled
        first_valid = int(valid_indices[0])
        filled[:first_valid] = filled[first_valid]
        for t in range(first_valid + 1, filled.shape[0]):
            if not mask[t]:
                filled[t] = filled[t - 1]
        return filled

    def update_and_predict(self, human_poses: dict, robot_pose: list):
        robot_pose_arr = np.asarray(robot_pose, dtype=np.float32)
        robot_xy = robot_pose_arr[:2]

        current_humans = {}
        for agent_id, pos in human_poses.items():
            current_humans[int(agent_id)] = self._to_xy_array(pos)
        self.human_history_buffer.append(current_humans)
        self.robot_history_buffer.append(robot_xy.astype(np.float32))

        scene_entries = []
        for agent_id in sorted(current_humans.keys()):
            history_np, history_mask = self._sample_agent_history(agent_id)
            valid_count = int(np.count_nonzero(history_mask))
            if valid_count < self.min_target_history_points:
                continue
            scene_entries.append((agent_id, self._carry_fill(history_np, history_mask), history_mask))
            if len(scene_entries) >= self.max_scene_humans:
                break

        if not scene_entries:
            return None, None

        robot_history, robot_mask = self._sample_robot_history()
        robot_history = self._carry_fill(robot_history, robot_mask)

        obs_xy = np.zeros((1, self.num_agents, self.obs_len, 2), dtype=np.float32)
        obs_mask = np.zeros((1, self.num_agents, self.obs_len), dtype=bool)
        obs_xy[0, 0] = robot_history
        obs_mask[0, 0] = robot_mask

        selected_agents = []
        for i, (agent_id, history_np, history_mask) in enumerate(scene_entries, start=1):
            obs_xy[0, i] = history_np
            obs_mask[0, i] = history_mask
            selected_agents.append(agent_id)

        # Center all coordinates on robot's last known position so model input
        # stays in the same small-range ego-centric frame it was trained on
        # (ETH/UCY Zara1 coords are ~[0,15]m; arbitrary map offsets break this).
        # Predictions are shifted back to world frame after inference.
        ego_center = robot_history[-1].copy()  # shape (2,)
        obs_xy = obs_xy - ego_center  # broadcast over (1, N, T, 2)

        trajectories, scores = self.predictor.predict(
            torch.from_numpy(obs_xy),
            torch.from_numpy(obs_mask),
        )
        best_mode_idx = int(torch.argmax(scores[0]).item()) if scores is not None else 0
        pred = trajectories[0, best_mode_idx, 1 : 1 + len(selected_agents), :, :].permute(1, 0, 2)

        # Shift predictions back to absolute world frame
        ego_center_t = torch.tensor(ego_center, dtype=torch.float32, device=trajectories.device)
        pred = pred + ego_center_t  # broadcast over (T, N, 2)

        out_tensor = pred.unsqueeze(0).unsqueeze(0).to(self.device).contiguous()
        return out_tensor, selected_agents


class MoFlowPredictorAdapter:
    """Adapter that exposes standalone MoFlow inference through update_and_predict."""

    def __init__(self, checkpoint_path=None, config_path=None, eval_config_path=None):
        MoFlowPredictor = load_predictor_class('moflow')

        resolved_checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_MOFLOW_CHECKPOINT
        resolved_config = Path(config_path) if config_path else DEFAULT_MOFLOW_CONFIG
        resolved_eval_config = Path(eval_config_path) if eval_config_path else DEFAULT_MOFLOW_EVAL_CONFIG

        self.predictor = MoFlowPredictor(
            checkpoint_path=resolved_checkpoint,
            config_path=resolved_config,
            eval_config_path=resolved_eval_config,
        )
        self.obs_len = int(self.predictor.obs_len)
        self.pred_len = int(self.predictor.pred_len)
        self.num_agents = max(2, int(self.predictor.max_agents))
        self.max_scene_humans = min(self.num_agents - 1, MAX_AGENT_NUM - 1)
        self.history_stride = max(1, round(TIMESTEP * PREDICTION_TIMER_HZ))
        self.buffer_len = (self.obs_len - 1) * self.history_stride + 1
        self.human_history_buffer = deque(maxlen=self.buffer_len)
        self.robot_history_buffer = deque(maxlen=self.buffer_len)
        self.device = self.predictor.device
        self.min_target_history_points = 1
        self.last_robot_pred = None  # (T, 2) numpy in world frame, updated each predict call
        self.last_best_mode_idx = 0

    @staticmethod
    def _to_xy_array(val):
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()[:2].astype(np.float32)
        if isinstance(val, np.ndarray):
            return val[:2].astype(np.float32)
        return np.asarray(val[:2], dtype=np.float32)

    @staticmethod
    def _scene_medoid_idx(all_modes_np: np.ndarray, forecast_mask: np.ndarray) -> int:
        """Scene-level medoid: pick the mode closest to all others across forecast agents.

        all_modes_np: (modes, agents, T, 2)
        forecast_mask: (agents,) bool
        """
        traj = all_modes_np[:, forecast_mask]          # (modes, N, T, 2)
        traj_flat = traj.reshape(traj.shape[0], -1)    # (modes, N*T*2)
        diff = traj_flat[:, None, :] - traj_flat[None, :, :]  # (modes, modes, N*T*2)
        dists = np.linalg.norm(diff, axis=-1)          # (modes, modes)
        return int(np.argmin(dists.mean(axis=1)))

    def _sample_agent_history(self, agent_id: int):
        history = np.zeros((self.obs_len, 2), dtype=np.float32)
        mask = np.zeros((self.obs_len,), dtype=bool)
        current_buffer_idx = len(self.human_history_buffer) - 1

        for t in range(self.obs_len):
            offset = (self.obs_len - 1 - t) * self.history_stride
            buffer_idx = current_buffer_idx - offset
            if buffer_idx < 0:
                continue
            snapshot = self.human_history_buffer[buffer_idx]
            if agent_id in snapshot:
                history[t] = snapshot[agent_id]
                mask[t] = True
        return history, mask

    def _sample_robot_history(self):
        history = np.zeros((self.obs_len, 2), dtype=np.float32)
        mask = np.zeros((self.obs_len,), dtype=bool)
        current_buffer_idx = len(self.robot_history_buffer) - 1

        for t in range(self.obs_len):
            offset = (self.obs_len - 1 - t) * self.history_stride
            buffer_idx = current_buffer_idx - offset
            if buffer_idx < 0:
                continue
            history[t] = self.robot_history_buffer[buffer_idx]
            mask[t] = True
        return history, mask

    def update_and_predict(self, human_poses: dict, robot_pose: list):
        robot_pose_arr = np.asarray(robot_pose, dtype=np.float32)
        robot_xy = robot_pose_arr[:2]

        current_humans = {}
        for agent_id, pos in human_poses.items():
            current_humans[int(agent_id)] = self._to_xy_array(pos)
        self.human_history_buffer.append(current_humans)
        self.robot_history_buffer.append(robot_xy.astype(np.float32))

        scene_entries = []
        for agent_id in sorted(current_humans.keys()):
            history_np, history_mask = self._sample_agent_history(agent_id)
            if int(np.count_nonzero(history_mask)) < self.min_target_history_points:
                continue
            scene_entries.append((agent_id, history_np, history_mask))
            if len(scene_entries) >= self.max_scene_humans:
                break

        if not scene_entries:
            return None, None

        robot_history, robot_mask = self._sample_robot_history()
        obs_xy = np.zeros((1, self.num_agents, self.obs_len, 2), dtype=np.float32)
        obs_mask = np.zeros((1, self.num_agents, self.obs_len), dtype=bool)
        obs_xy[0, 0] = robot_history
        obs_mask[0, 0] = robot_mask

        selected_agents = []
        for i, (agent_id, history_np, history_mask) in enumerate(scene_entries, start=1):
            obs_xy[0, i] = history_np
            obs_mask[0, i] = history_mask
            selected_agents.append(agent_id)

        # Center coordinates on robot's last position so past_abs features stay
        # within the ETH/UCY normalization range the model was trained on.
        ego_center = robot_history[-1].copy()
        obs_xy = obs_xy - ego_center

        print("=" * 60)
        print(f"[MoFlow] ego_center (robot last pos): {ego_center}")
        print(f"[MoFlow] selected_agents: {selected_agents}")
        print(f"[MoFlow] obs_xy shape: {obs_xy.shape}  obs_mask shape: {obs_mask.shape}")
        for agent_slot in range(obs_xy.shape[1]):
            valid_steps = np.where(obs_mask[0, agent_slot])[0]
            if valid_steps.size == 0:
                continue
            label = "robot" if agent_slot == 0 else f"agent_id={selected_agents[agent_slot - 1] if agent_slot - 1 < len(selected_agents) else '?'}"
            print(f"  slot {agent_slot} ({label}): {valid_steps.size}/{obs_mask.shape[2]} valid steps")
            for t in range(obs_xy.shape[2]):
                valid = obs_mask[0, agent_slot, t]
                xy = obs_xy[0, agent_slot, t]
                print(f"    t={t:2d} {'[v]' if valid else '[ ]'} xy={xy}")
        print("=" * 60)

        trajectories, scores = self.predictor.predict(
            torch.from_numpy(obs_xy),
            torch.from_numpy(obs_mask),
        )
        # Shift predictions back to absolute world frame.
        ego_center_t = torch.tensor(ego_center, dtype=torch.float32, device=trajectories.device)

        # Store all modes for all agents (world frame) for multi-mode visualization.
        # Shape: (modes, num_agents_including_robot, T, 2)
        all_modes_world = (trajectories[0] + ego_center_t).detach().cpu().numpy()
        self.last_all_modes_pred = all_modes_world          # (modes, agents, T, 2)
        self.last_selected_agents = list(selected_agents)  # human agent IDs at slots 1+

        # Weighted expectation over all modes.
        # Use softmax of classifier scores as weights; fall back to uniform if no scores.
        if scores is not None:
            weights = torch.softmax(scores[0], dim=0).detach().cpu().numpy()  # (modes,)
        else:
            weights = np.ones(all_modes_world.shape[0]) / all_modes_world.shape[0]
        expected_world = (weights[:, None, None, None] * all_modes_world).sum(axis=0)  # (agents, T, 2)
        self.last_expected_pred = expected_world  # (agents, T, 2) world frame

        # Robot expected prediction for visualization.
        self.last_robot_pred = expected_world[0]  # (T, 2)

        # Human expected predictions as output tensor.
        pred = torch.tensor(
            expected_world[1 : 1 + len(selected_agents)],  # (N, T, 2)
            dtype=torch.float32,
            device=self.device,
        ).permute(1, 0, 2)  # (T, N, 2)

        print("[MoFlow] predictions (robot/centered frame, expected over modes):")
        robot_pred_centered = expected_world[0] - ego_center  # (T, 2)
        print(f"  slot 0 (robot):")
        for t in range(robot_pred_centered.shape[0]):
            print(f"    t={t:2d} xy={robot_pred_centered[t]}")
        for i, agent_id in enumerate(selected_agents):
            agent_centered = expected_world[i + 1] - ego_center  # (T, 2)
            print(f"  slot {i+1} (agent_id={agent_id}):")
            for t in range(agent_centered.shape[0]):
                print(f"    t={t:2d} xy={agent_centered[t]}")

        # print("[MoFlow] predictions (world frame):")
        # robot_pred_world = self.last_robot_pred
        # print(f"  slot 0 (robot):")
        # for t in range(robot_pred_world.shape[0]):
        #     print(f"    t={t:2d} xy={robot_pred_world[t]}")
        # pred_np = pred.detach().cpu().numpy()
        # for i, agent_id in enumerate(selected_agents):
        #     print(f"  slot {i+1} (agent_id={agent_id}):")
        #     for t in range(pred_np.shape[0]):
        #         print(f"    t={t:2d} xy={pred_np[t, i]}")
        # print("=" * 60)

        out_tensor = pred.unsqueeze(0).unsqueeze(0).to(self.device).contiguous()
        return out_tensor, selected_agents


def create_predictor(backend="hst", **kwargs):
    backend_name = str(backend).strip().lower()
    if backend_name == "hst":
        return HSTPredictor()
    if backend_name == "eqmotion":
        return EqMotionPredictorAdapter(bundle_path=kwargs.get("bundle_path"))
    if backend_name == "autobots":
        return AutoBotsPredictorAdapter(
            checkpoint_path=kwargs.get("checkpoint_path"),
            config_path=kwargs.get("config_path"),
        )
    if backend_name == "moflow":
        return MoFlowPredictorAdapter(
            checkpoint_path=kwargs.get("checkpoint_path"),
            config_path=kwargs.get("config_path"),
            eval_config_path=kwargs.get("eval_config_path"),
        )
    raise ValueError(f"Unsupported prediction backend: {backend}")
