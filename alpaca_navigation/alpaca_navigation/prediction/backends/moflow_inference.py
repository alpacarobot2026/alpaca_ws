#!/usr/bin/env python3
"""Standalone MoFlow joint inference helper.

This mirrors the current complexity_prediction MoFlow inference behavior for
ETH/UCY-style joint prediction, including teacher and student variants.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


_YAML_INT_RE = re.compile(r"^[+-]?\d+$")
_YAML_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


def _nested(data: Dict[str, object], *keys: str, default):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _parse_yaml_scalar(value: str):
    stripped = value.strip()
    if stripped == "":
        return ""
    if stripped in {"null", "Null", "NULL", "~"}:
        return None
    if stripped in {"true", "True", "TRUE"}:
        return True
    if stripped in {"false", "False", "FALSE"}:
        return False
    if stripped == "[]":
        return []
    if stripped == "{}":
        return {}
    if (stripped.startswith("'") and stripped.endswith("'")) or (
        stripped.startswith('"') and stripped.endswith('"')
    ):
        return stripped[1:-1]
    if _YAML_INT_RE.match(stripped):
        try:
            return int(stripped)
        except ValueError:
            pass
    if _YAML_FLOAT_RE.match(stripped):
        try:
            return float(stripped)
        except ValueError:
            pass
    return stripped


def _load_simple_yaml_mapping(path: Path) -> Dict[str, object]:
    root: Dict[str, object] = {}
    stack = [(-1, root)]
    lines = path.read_text(encoding="utf-8").splitlines()

    def _next_content(start_idx: int):
        for next_idx in range(start_idx + 1, len(lines)):
            candidate = lines[next_idx].split("#", 1)[0].rstrip()
            if candidate.strip():
                return next_idx + 1, candidate
        return None, None

    for idx, raw_line in enumerate(lines):
        lineno = idx + 1
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Unsupported YAML indentation at {path}:{lineno}")

        content = line.strip()
        if content.startswith("- "):
            item_value = _parse_yaml_scalar(content[2:].strip())
            if not isinstance(stack[-1][1], list):
                raise ValueError(f"Unexpected list item at {path}:{lineno}: {raw_line}")
            stack[-1][1].append(item_value)
            continue
        if ":" not in content:
            raise ValueError(f"Unsupported YAML line at {path}:{lineno}: {raw_line}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not isinstance(parent, dict):
            raise ValueError(f"Unsupported YAML parent structure at {path}:{lineno}")

        if value == "":
            _next_lineno, next_line = _next_content(idx)
            if next_line is not None:
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_indent >= indent and next_line.strip().startswith("- "):
                    child = []
                else:
                    child = {}
            else:
                child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_yaml_scalar(value)

    return root


def _load_yaml(path: Path) -> Dict[str, object]:
    try:
        import yaml
    except ModuleNotFoundError:
        data = _load_simple_yaml_mapping(path)
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"MoFlow config must be a mapping, got {type(data)}")
    return data


class DotDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _to_dotdict(value):
    if isinstance(value, dict):
        return DotDict({key: _to_dotdict(val) for key, val in value.items()})
    return value


def normalize_min_max(x: torch.Tensor, x_min: float, x_max: float, y_min: float, y_max: float) -> torch.Tensor:
    denom = max(float(x_max) - float(x_min), 1e-6)
    return (x - float(x_min)) * (float(y_max) - float(y_min)) / denom + float(y_min)


def unnormalize_min_max(x: torch.Tensor, x_min: float, x_max: float, y_min: float, y_max: float) -> torch.Tensor:
    denom = max(float(y_max) - float(y_min), 1e-6)
    return (x - float(y_min)) * (float(x_max) - float(x_min)) / denom + float(x_min)


def fill_history_positions(obs_xy: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    filled = obs_xy.clone()
    carry = torch.zeros_like(filled[:, :, 0])
    have_carry = torch.zeros_like(obs_mask[:, :, 0], dtype=torch.bool)
    for t in range(obs_xy.shape[2]):
        valid = obs_mask[:, :, t]
        current = filled[:, :, t]
        carry = torch.where(valid.unsqueeze(-1), current, carry)
        filled[:, :, t] = torch.where(valid.unsqueeze(-1), current, carry)
        have_carry = have_carry | valid

    backfill = filled.clone()
    carry = torch.zeros_like(backfill[:, :, -1])
    have_carry = torch.zeros_like(obs_mask[:, :, -1], dtype=torch.bool)
    for t in range(obs_xy.shape[2] - 1, -1, -1):
        valid = obs_mask[:, :, t]
        current = backfill[:, :, t]
        carry = torch.where(valid.unsqueeze(-1), current, carry)
        backfill[:, :, t] = torch.where((valid | have_carry).unsqueeze(-1), current, carry)
        have_carry = have_carry | valid

    no_obs = ~obs_mask.any(dim=-1)
    backfill[no_obs] = 0.0
    return backfill


def context_agent_mask(obs_mask: torch.Tensor) -> torch.Tensor:
    return obs_mask.any(dim=-1)


def forecast_agent_mask(obs_mask: torch.Tensor) -> torch.Tensor:
    return obs_mask[:, :, -1]


def compute_scene_scores_from_agent_logits(agent_logits: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = agent_logits * agent_mask[:, None, :].float()
    denom = agent_mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
    return masked_logits.sum(dim=-1) / denom


class MoFlowPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path,
        eval_config_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.eval_config_path = Path(eval_config_path) if eval_config_path else None
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"MoFlow checkpoint not found: {self.checkpoint_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"MoFlow config not found: {self.config_path}")
        if self.eval_config_path is not None and not self.eval_config_path.exists():
            raise FileNotFoundError(f"MoFlow eval config not found: {self.eval_config_path}")

        self.config = _load_yaml(self.config_path)
        family = str(_nested(self.config, "model", "family", default="moflow")).lower()
        prediction_mode = str(_nested(self.config, "model", "prediction_mode", default="joint")).lower()
        if family != "moflow":
            raise ValueError(f"Expected MoFlow family config, got {family!r}")
        if prediction_mode != "joint":
            raise ValueError(f"Only MoFlow joint prediction is supported here, got {prediction_mode!r}")

        self.eval_config = _load_yaml(self.eval_config_path) if self.eval_config_path is not None else {}
        self._apply_eval_overrides()

        self.dataset_name = str(_nested(self.config, "dataset", "name", default="eth_ucy")).lower()
        self.use_nba_backend = self.dataset_name == "nba"
        dataset_key = "nba" if self.use_nba_backend else "eth_ucy"
        self.obs_len = int(_nested(self.config, "dataset", dataset_key, "obs_len", default=8))
        self.pred_len = int(_nested(self.config, "dataset", dataset_key, "pred_len", default=12))
        self.max_agents = int(_nested(self.config, "dataset", dataset_key, "max_agents", default=20))
        self.num_modes = int(_nested(self.config, "model", "num_modes", default=20))
        self.variant = str(_nested(self.config, "model", "moflow", "variant", default="teacher")).lower()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        try:
            from .moflow.models.backbone import IMLETransformer, MotionTransformer
            from .moflow.models.backbone_eth_ucy import ETHIMLETransformer, ETHMotionTransformer
            from .moflow.models.flow_matching import FlowMatcher
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"MoFlow backend is missing the python dependency {getattr(exc, 'name', None)!r}. "
                "Install the backend requirements: easydict, einops, tensorboardX, tqdm, matplotlib."
            ) from exc

        class _NullLogger:
            def info(self, *_args, **_kwargs):
                return None

        self._null_logger = _NullLogger()
        self.flow_matcher_cls = FlowMatcher
        self.imle_transformer_cls = IMLETransformer
        self.motion_transformer_cls = MotionTransformer
        self.eth_imle_transformer_cls = ETHIMLETransformer
        self.eth_motion_transformer_cls = ETHMotionTransformer
        self.flow_cfg = self._build_flow_config()

        if self.variant == "student":
            self.model = (
                self.imle_transformer_cls(self.flow_cfg.MODEL, logger=self._null_logger, config=self.flow_cfg)
                if self.use_nba_backend
                else self.eth_imle_transformer_cls(self.flow_cfg.MODEL, logger=self._null_logger, config=self.flow_cfg)
            ).to(self.device)
        else:
            denoiser_backbone = (
                self.motion_transformer_cls(self.flow_cfg.MODEL, logger=self._null_logger, config=self.flow_cfg)
                if self.use_nba_backend
                else self.eth_motion_transformer_cls(self.flow_cfg.MODEL, logger=self._null_logger, config=self.flow_cfg)
            )
            self.model = self.flow_matcher_cls(
                cfg=self.flow_cfg,
                model=denoiser_backbone,
                logger=self._null_logger,
            ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = self._checkpoint_state_for_eval(checkpoint)
        if not isinstance(state_dict, dict):
            raise KeyError("MoFlow checkpoint does not contain a model_state_dict.")
        self.model.load_state_dict(state_dict, strict=True)

        stats = checkpoint.get("moflow_stats", {})
        self.flow_cfg.past_traj_min = float(stats.get("past_traj_min", -1.0))
        self.flow_cfg.past_traj_max = float(stats.get("past_traj_max", 1.0))
        self.flow_cfg.fut_traj_min = float(stats.get("fut_traj_min", -1.0))
        self.flow_cfg.fut_traj_max = float(stats.get("fut_traj_max", 1.0))
        self._disable_nested_tensor_transformers()
        self.model.eval()

    def _checkpoint_state_for_eval(self, checkpoint: Dict[str, object]) -> Dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            return checkpoint
        use_ema = bool(
            _nested(
                self.config,
                "model",
                "moflow",
                "distillation",
                "use_ema_for_eval",
                default=False,
            )
        )
        if self.variant == "student" and use_ema:
            ema_state = checkpoint.get("ema_state_dict")
            if isinstance(ema_state, dict):
                return ema_state.get("model_state_dict", ema_state)
        return checkpoint.get("model_state_dict", checkpoint)

    def _disable_nested_tensor_transformers(self) -> None:
        modules = []
        model_root = getattr(self.model, "model", self.model)
        context_encoder = getattr(model_root, "context_encoder", None)
        if context_encoder is not None:
            agent_social_encoder = getattr(context_encoder, "agent_social_encoder", None)
            if agent_social_encoder is not None:
                modules.append(getattr(agent_social_encoder, "transformer_encoder", None))
            modules.append(getattr(context_encoder, "transformer_encoder", None))

        for module in modules:
            if module is None:
                continue
            if hasattr(module, "enable_nested_tensor"):
                module.enable_nested_tensor = False
            if hasattr(module, "use_nested_tensor"):
                module.use_nested_tensor = False

    def _apply_eval_overrides(self) -> None:
        eval_moflow = self.eval_config.get("moflow", {}) if isinstance(self.eval_config, dict) else {}
        if not isinstance(eval_moflow, dict):
            return
        target = self.config.setdefault("model", {}).setdefault("moflow", {})
        for key in ("sampling_steps", "solver", "lin_poly_p", "lin_poly_long_step"):
            value = eval_moflow.get(key)
            if value is not None:
                target[key] = value

    def _build_flow_config(self):
        hidden_dim = int(_nested(self.config, "model", "moflow", "hidden_dim", default=128))
        variant = str(_nested(self.config, "model", "moflow", "variant", default="teacher")).lower()
        drop_method = _nested(self.config, "model", "moflow", "drop_method", default=None)
        if drop_method is not None:
            drop_method = str(drop_method).strip().lower()
            if drop_method in {"", "none"}:
                drop_method = None

        common = {
            "device": str(self.device),
            "agents": self.max_agents,
            "future_frames": self.pred_len,
            "denoising_head_preds": self.num_modes,
            "tied_noise": bool(_nested(self.config, "model", "moflow", "tied_noise", default=True)),
            "t_schedule": str(_nested(self.config, "model", "moflow", "t_schedule", default="logit_normal")),
            "logit_norm_mean": float(_nested(self.config, "model", "moflow", "logit_norm_mean", default=-0.5)),
            "logit_norm_std": float(_nested(self.config, "model", "moflow", "logit_norm_std", default=1.5)),
            "objective": "pred_data",
            "denoising_method": "fm",
            "sigma_data": float(_nested(self.config, "model", "moflow", "sigma_data", default=0.13)),
            "fm_wrapper": "direct",
            "fm_rew_sqrt": False,
            "fm_in_scaling": bool(_nested(self.config, "model", "moflow", "use_input_scaling", default=True)),
            "drop_method": drop_method,
            "drop_logi_k": float(_nested(self.config, "model", "moflow", "drop_logi_k", default=20.0)),
            "drop_logi_m": float(_nested(self.config, "model", "moflow", "drop_logi_m", default=0.5)),
            "LOSS_NN_MODE": str(_nested(self.config, "model", "moflow", "loss_nn_mode", default="scene")),
            "data_norm": "min_max",
            "sampling_steps": int(_nested(self.config, "model", "moflow", "sampling_steps", default=10)),
            "solver": str(_nested(self.config, "model", "moflow", "solver", default="euler")),
            "lin_poly_p": int(_nested(self.config, "model", "moflow", "lin_poly_p", default=2)),
            "lin_poly_long_step": int(_nested(self.config, "model", "moflow", "lin_poly_long_step", default=1000)),
            "variant": variant,
            "OPTIMIZATION": {
                "LOSS_WEIGHTS": {
                    "reg": float(_nested(self.config, "model", "moflow", "weight_reg", default=1.0)),
                    "cls": float(_nested(self.config, "model", "moflow", "weight_cls", default=1.0)),
                    "vel": float(_nested(self.config, "model", "moflow", "weight_vel", default=0.2)),
                }
            },
        }
        if variant == "student":
            distill_cfg = _nested(self.config, "model", "moflow", "distillation", default={})
            if not isinstance(distill_cfg, dict):
                distill_cfg = {}
            common.update(
                {
                    "objective": "set",
                    "num_to_gen": int(distill_cfg.get("num_to_gen", 20)),
                    "loss_reg_chamfer_weight": float(distill_cfg.get("loss_reg_chamfer_weight", 1.0)),
                    "loss_reg_gt_weight": float(distill_cfg.get("loss_reg_gt_weight", 0.0)),
                    "loss_reg_reduction": str(distill_cfg.get("loss_reg_reduction", "sum")),
                }
            )

        if self.use_nba_backend:
            common.update(
                {
                    "traj_mean": [14.0, 7.5],
                    "traj_scale_total": 94.0 / 28.0,
                    "MODEL": {
                        "NUM_PROPOSED_QUERY": self.num_modes,
                        "MODEL_OUT_DIM": self.pred_len * 2,
                        "REGRESSION_MLPS": [hidden_dim, hidden_dim * 2, self.pred_len * 2],
                        "CLASSIFICATION_MLPS": [hidden_dim, hidden_dim, 1],
                        "USE_PE_QUERY": True,
                        "USE_PE_AGENT": True,
                        "CONTEXT_ENCODER": {
                            "NAME": "MTREncoder",
                            "NUM_OF_ATTN_NEIGHBORS": self.max_agents,
                            "NUM_INPUT_CONTEXT": 6,
                            "NUM_CHANNEL_IN_MLP_AGENT": hidden_dim * 2,
                            "NUM_LAYER_IN_MLP_AGENT": 3,
                            "D_MODEL": hidden_dim,
                            "NUM_ATTN_LAYERS": int(_nested(self.config, "model", "moflow", "num_layers", default=4)),
                            "NUM_ATTN_HEAD": int(_nested(self.config, "model", "moflow", "num_heads", default=8)),
                            "DROPOUT_OF_ATTN": float(_nested(self.config, "model", "moflow", "dropout", default=0.1)),
                        },
                        "MOTION_DECODER": {
                            "NAME": "MTRDecoder",
                            "D_MODEL": hidden_dim,
                            "NUM_DECODER_BLOCKS": int(_nested(self.config, "model", "moflow", "num_layers", default=4)),
                            "NUM_ATTN_HEAD": int(_nested(self.config, "model", "moflow", "num_heads", default=8)),
                            "DROPOUT_OF_ATTN": float(_nested(self.config, "model", "moflow", "dropout", default=0.1)),
                        },
                    },
                }
            )
        else:
            common.update(
                {
                    "MODEL": {
                        "NUM_PROPOSED_QUERY": self.num_modes,
                        "MODEL_OUT_DIM": self.pred_len * 2,
                        "REGRESSION_MLPS": [hidden_dim, hidden_dim * 2, self.pred_len * 2],
                        "CLASSIFICATION_MLPS": [hidden_dim, hidden_dim, 1],
                        "CONTEXT_ENCODER": {
                            "NAME": "ETHEncoder",
                            "NUM_INPUT_CONTEXT": 6,
                            "PAST_FRAMES": self.obs_len,
                            "AGENTS": self.max_agents,
                            "NUM_CHANNEL_IN_MLP_AGENT": hidden_dim * 2,
                            "NUM_LAYER_IN_MLP_AGENT": 3,
                            "D_MODEL": hidden_dim,
                            "NUM_ATTN_LAYERS": int(_nested(self.config, "model", "moflow", "num_layers", default=4)),
                            "NUM_ATTN_HEAD": int(_nested(self.config, "model", "moflow", "num_heads", default=8)),
                            "DROPOUT_OF_ATTN": float(_nested(self.config, "model", "moflow", "dropout", default=0.1)),
                        },
                        "MOTION_DECODER": {
                            "NAME": "MTRDecoder",
                            "D_MODEL": hidden_dim,
                            "NUM_DECODER_BLOCKS": int(_nested(self.config, "model", "moflow", "num_layers", default=4)),
                            "NUM_ATTN_HEAD": int(_nested(self.config, "model", "moflow", "num_heads", default=8)),
                            "DROPOUT_OF_ATTN": float(_nested(self.config, "model", "moflow", "dropout", default=0.1)),
                        },
                    },
                }
            )
        return _to_dotdict(common)

    @staticmethod
    def _history_velocity(rel: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        vel = torch.zeros_like(rel)
        if rel.shape[2] > 1:
            transition_mask = obs_mask[:, :, 1:] & obs_mask[:, :, :-1]
            vel[:, :, :-1] = (rel[:, :, 1:] - rel[:, :, :-1]) * transition_mask.unsqueeze(-1)
        return vel

    @staticmethod
    def _identity_rotation(reference: torch.Tensor, num_agents: int) -> torch.Tensor:
        eye = torch.eye(2, device=reference.device, dtype=reference.dtype)
        return eye.view(1, 1, 2, 2).expand(reference.shape[0], num_agents, 2, 2).clone()

    @staticmethod
    def _rotation_from_vectors(vectors: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        theta = torch.atan2(vectors[..., 1], vectors[..., 0] + 1e-5)
        valid = torch.linalg.norm(vectors, dim=-1) > 1e-8
        if valid_mask is not None:
            valid = valid & valid_mask.bool()
        theta = torch.where(valid, theta, torch.zeros_like(theta))

        rotation = vectors.new_zeros(*vectors.shape[:-1], 2, 2)
        rotation[..., 0, 0] = torch.cos(theta)
        rotation[..., 0, 1] = torch.sin(theta)
        rotation[..., 1, 0] = -torch.sin(theta)
        rotation[..., 1, 1] = torch.cos(theta)
        return rotation

    @staticmethod
    def _apply_rotation(points: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        return torch.matmul(rotation.unsqueeze(2), points.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _apply_sample_rotation(points: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        return torch.matmul(rotation[:, None, :, None], points.unsqueeze(-1)).squeeze(-1)

    def _use_rotation(self) -> bool:
        return bool(_nested(self.config, "model", "moflow", "rotate", default=False))

    def _eth_rotation(self, past_rel: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        if not self._use_rotation():
            return self._identity_rotation(past_rel, past_rel.shape[1])
        rotate_time_frame = int(_nested(self.config, "model", "moflow", "rotate_time_frame", default=0))
        if rotate_time_frame < 0 or rotate_time_frame >= past_rel.shape[2]:
            raise ValueError(
                "model.moflow.rotate_time_frame must be in [0, obs_len), "
                f"got {rotate_time_frame} for obs_len {past_rel.shape[2]}."
            )
        valid = obs_mask[:, :, rotate_time_frame] & obs_mask[:, :, -1]
        return self._rotation_from_vectors(past_rel[:, :, rotate_time_frame], valid)

    def _build_eth_model_batch(self, obs_xy: torch.Tensor, obs_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        obs_filled = fill_history_positions(obs_xy, obs_mask)
        context_mask = context_agent_mask(obs_mask)
        pred_agent_mask = forecast_agent_mask(obs_mask)
        last_obs = obs_filled[:, :, -1:]
        past_abs = obs_filled
        past_rel = obs_filled - last_obs
        rotation = self._eth_rotation(past_rel, obs_mask)

        if self._use_rotation():
            past_abs = self._apply_rotation(past_abs, rotation)
            past_rel = self._apply_rotation(past_rel, rotation)

        past_vel = self._history_velocity(past_rel, obs_mask)
        past_features = torch.cat((past_abs, past_rel, past_vel), dim=-1)
        past_norm = normalize_min_max(
            past_features,
            self.flow_cfg.past_traj_min,
            self.flow_cfg.past_traj_max,
            -1,
            1,
        )
        return {
            "batch_size": int(obs_xy.shape[0]),
            "past_traj": past_norm,
            "fut_traj": torch.zeros(
                obs_xy.shape[0],
                obs_xy.shape[1],
                self.pred_len,
                2,
                device=obs_xy.device,
                dtype=obs_xy.dtype,
            ),
            "past_traj_original_scale": past_features,
            "context_agent_mask": context_mask,
            "agent_mask": context_mask,
            "forecast_agent_mask": pred_agent_mask,
            "rotation_matrix": rotation,
            "prediction_mode": "joint",
        }

    @torch.no_grad()
    def predict(self, obs_xy: torch.Tensor, obs_mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        obs_xy = obs_xy.to(self.device, dtype=torch.float32)
        obs_mask = obs_mask.to(self.device, dtype=torch.bool)
        if obs_xy.ndim != 4 or obs_xy.shape[-1] != 2:
            raise ValueError(f"Expected obs_xy shape (B,N,T,2), got {tuple(obs_xy.shape)}")
        if obs_mask.shape != obs_xy.shape[:-1]:
            raise ValueError(f"Expected obs_mask shape {tuple(obs_xy.shape[:-1])}, got {tuple(obs_mask.shape)}")
        if obs_xy.shape[1] != self.max_agents:
            raise ValueError(f"Expected {self.max_agents} agents, got {obs_xy.shape[1]}")

        if self.use_nba_backend:
            raise NotImplementedError("This standalone MoFlow inference helper currently supports ETH/UCY joint models only.")

        model_batch = self._build_eth_model_batch(obs_xy, obs_mask)
        pred_score = None
        if self.variant == "student":
            generated = self.model(model_batch, num_to_gen=1).float().squeeze(1)
            trajectories = generated.view(
                generated.shape[0],
                generated.shape[1],
                generated.shape[2],
                self.pred_len,
                2,
            )
            trajectories = unnormalize_min_max(
                trajectories,
                self.flow_cfg.fut_traj_min,
                self.flow_cfg.fut_traj_max,
                -1,
                1,
            )
        else:
            final_sample, _, _, _, pred_score = self.model.sample(model_batch, num_trajs=self.num_modes)
            final_sample = final_sample.float()
            if pred_score is not None:
                pred_score = pred_score.float()
            trajectories = final_sample.view(
                final_sample.shape[0],
                final_sample.shape[1],
                final_sample.shape[2],
                self.pred_len,
                2,
            )
            trajectories = unnormalize_min_max(
                trajectories,
                self.flow_cfg.fut_traj_min,
                self.flow_cfg.fut_traj_max,
                -1,
                1,
            )
        last_obs = fill_history_positions(obs_xy, obs_mask)[:, :, -1:]
        trajectories = self._apply_sample_rotation(
            trajectories,
            model_batch["rotation_matrix"].transpose(-1, -2),
        )
        trajectories = trajectories + last_obs[:, None]
        scene_scores = (
            compute_scene_scores_from_agent_logits(pred_score, model_batch["forecast_agent_mask"])
            if pred_score is not None
            else None
        )
        return trajectories, scene_scores
