#!/usr/bin/env python3
"""Minimal standalone AutoBots joint inference helper.

It carries only the model pieces needed to run the trained joint AutoBots
checkpoint used by the ROS prediction node.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_YAML_INT_RE = re.compile(r"^[+-]?\d+$")
_YAML_FLOAT_RE = re.compile(
    r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$"
)


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
        raise ValueError(f"AutoBots config must be a mapping, got {type(data)}")
    return data


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return self.dropout(x + self.pe[: x.size(0), :])


class OutputModel(nn.Module):
    def __init__(self, d_k=64, predict_yaw=False):
        super().__init__()
        self.d_k = d_k
        self.predict_yaw = predict_yaw
        out_len = 6 if predict_yaw else 5
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), math.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)),
            nn.ReLU(),
            init_(nn.Linear(self.d_k, self.d_k)),
            nn.ReLU(),
            init_(nn.Linear(self.d_k, out_len)),
        )
        self.min_stdev = 0.01

    def forward(self, agent_latent_state):
        t_steps = agent_latent_state.shape[0]
        batch_modes = agent_latent_state.shape[1]
        pred_obs = self.observation_model(agent_latent_state.reshape(-1, self.d_k)).reshape(t_steps, batch_modes, -1)
        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 4]) * 0.9
        if self.predict_yaw:
            return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho, pred_obs[:, :, 5]], dim=2)
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)


class AutoBotJoint(nn.Module):
    def __init__(
        self,
        d_k=128,
        num_other_agents=19,
        num_modes=5,
        pred_len=12,
        num_encoder_layers=1,
        dropout=0.0,
        k_attr=2,
        num_heads=16,
        num_decoder_layers=1,
        tx_hidden_size=384,
        num_agent_types=1,
    ):
        super().__init__()
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), math.sqrt(2))

        self.k_attr = k_attr
        self.d_k = d_k
        self._M = num_other_agents
        self.c = num_modes
        self.T = pred_len
        self.L_enc = num_encoder_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.L_dec = num_decoder_layers
        self.tx_hidden_size = tx_hidden_size

        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))

        self.social_attn_layers = nn.ModuleList()
        self.temporal_attn_layers = nn.ModuleList()
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_k,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
            )
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=2))

            tx_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_k,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
            )
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.emb_agent_types = nn.Sequential(init_(nn.Linear(num_agent_types, self.d_k)))
        self.dec_agenttypes_encoder = nn.Sequential(
            init_(nn.Linear(2 * self.d_k, self.d_k)),
            nn.ReLU(),
            init_(nn.Linear(self.d_k, self.d_k)),
        )

        self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, 1, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)

        self.social_attn_decoder_layers = nn.ModuleList()
        self.temporal_attn_decoder_layers = nn.ModuleList()
        for _ in range(self.L_dec):
            tx_decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_k,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
            )
            self.temporal_attn_decoder_layers.append(nn.TransformerDecoder(tx_decoder_layer, num_layers=2))
            tx_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_k,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
            )
            self.social_attn_decoder_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0)
        self.output_model = OutputModel(d_k=self.d_k, predict_yaw=False)
        self.P = nn.Parameter(torch.Tensor(self.c, 1, 1, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.P)
        self.prob_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        self.prob_predictor = init_(nn.Linear(self.d_k, 1))

    def generate_decoder_mask(self, seq_len, device):
        return torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()

    def process_observations(self, ego, agents):
        ego_tensor = ego[:, :, : self.k_attr]
        env_masks = ego[:, :, -1]
        temp_masks = torch.cat((env_masks.unsqueeze(-1), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).type(torch.bool).to(agents.device)
        opps_tensor = agents[:, :, :, : self.k_attr]
        return ego_tensor, opps_tensor, opps_masks

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        t_obs = agents_emb.size(0)
        batch = agent_masks.size(0)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, t_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == t_obs] = False
        agents_temp_emb = layer(
            self.pos_encoder(agents_emb.reshape(t_obs, batch * (self._M + 1), -1)),
            src_key_padding_mask=agent_masks,
        )
        return agents_temp_emb.view(t_obs, batch, self._M + 1, -1)

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        t_obs = agents_emb.size(0)
        batch = agent_masks.size(0)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self._M + 1, batch * t_obs, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, self._M + 1))
        return agents_soc_emb.view(self._M + 1, batch, t_obs, -1).permute(2, 1, 0, 3)

    def temporal_attn_decoder_fn(self, agents_emb, context, agent_masks, layer):
        t_obs = context.size(0)
        batch_modes = agent_masks.size(0)
        time_masks = self.generate_decoder_mask(seq_len=self.T, device=agents_emb.device)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, t_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == t_obs] = False
        agents_emb = agents_emb.reshape(self.T, -1, self.d_k)
        context = context.view(-1, batch_modes * (self._M + 1), self.d_k)
        agents_temp_emb = layer(agents_emb, context, tgt_mask=time_masks, memory_key_padding_mask=agent_masks)
        return agents_temp_emb.view(self.T, batch_modes, self._M + 1, -1)

    def social_attn_decoder_fn(self, agents_emb, agent_masks, layer):
        batch = agent_masks.size(0)
        agent_masks = agent_masks[:, -1:].repeat(1, self.T, 1).view(-1, self._M + 1)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self._M + 1, batch * self.T, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks)
        return agents_soc_emb.view(self._M + 1, batch, self.T, -1).permute(2, 1, 0, 3)

    def forward(self, ego_in, agents_in, agent_types):
        batch = ego_in.size(0)
        ego_tensor, agents_tensor, opps_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), agents_tensor), dim=2)
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)

        for i in range(self.L_enc):
            agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
            agents_emb = self.social_attn_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])

        opps_masks_modes = opps_masks.unsqueeze(1).repeat(1, self.c, 1, 1).view(batch * self.c, ego_in.shape[1], -1)
        context = agents_emb.unsqueeze(2).repeat(1, 1, self.c, 1, 1)
        context = context.view(ego_in.shape[1], batch * self.c, self._M + 1, self.d_k)

        agent_types_features = self.emb_agent_types(agent_types).unsqueeze(1)
        agent_types_features = agent_types_features.repeat(1, self.c, 1, 1).view(-1, self._M + 1, self.d_k)
        agent_types_features = agent_types_features.unsqueeze(0).repeat(self.T, 1, 1, 1)

        dec_parameters = self.Q.repeat(1, batch, 1, self._M + 1, 1).view(self.T, batch * self.c, self._M + 1, -1)
        agents_dec_emb = self.dec_agenttypes_encoder(torch.cat((dec_parameters, agent_types_features), dim=-1))

        for i in range(self.L_dec):
            agents_dec_emb = self.temporal_attn_decoder_fn(
                agents_dec_emb,
                context,
                opps_masks_modes,
                layer=self.temporal_attn_decoder_layers[i],
            )
            agents_dec_emb = self.social_attn_decoder_fn(
                agents_dec_emb,
                opps_masks_modes,
                layer=self.social_attn_decoder_layers[i],
            )

        out_dists = self.output_model(agents_dec_emb.reshape(self.T, -1, self.d_k))
        out_dists = out_dists.reshape(self.T, batch, self.c, self._M + 1, -1).permute(2, 0, 1, 3, 4)

        mode_params_emb = self.P.repeat(1, batch, self._M + 1, 1).view(self.c, -1, self.d_k)
        flat_agents = agents_emb.reshape(-1, batch * (self._M + 1), self.d_k)
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=flat_agents, value=flat_agents)[0]
        mode_probs = self.prob_predictor(mode_params_emb).squeeze(-1).view(self.c, batch, self._M + 1)
        return out_dists, F.softmax(mode_probs.sum(2).transpose(0, 1), dim=1)


class AutoBotsPredictor:
    def __init__(self, checkpoint_path: Path, config_path: Path, device: Optional[str] = None) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"AutoBots config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"AutoBots checkpoint not found: {self.checkpoint_path}")

        self.config = _load_yaml(self.config_path)
        family = str(_nested(self.config, "model", "family", default="autobots")).lower()
        mode = str(_nested(self.config, "model", "prediction_mode", default="joint")).lower()
        if family != "autobots" or mode != "joint":
            raise ValueError(f"Expected joint AutoBots config, got family={family!r}, prediction_mode={mode!r}")

        self.obs_len = int(_nested(self.config, "dataset", "eth_ucy", "obs_len", default=8))
        self.pred_len = int(_nested(self.config, "dataset", "eth_ucy", "pred_len", default=12))
        self.max_agents = int(_nested(self.config, "dataset", "eth_ucy", "max_agents", default=20))
        self.num_modes = int(_nested(self.config, "model", "num_modes", default=5))
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = AutoBotJoint(
            d_k=int(_nested(self.config, "model", "hidden_size", default=128)),
            num_other_agents=self.max_agents - 1,
            num_modes=self.num_modes,
            pred_len=self.pred_len,
            num_encoder_layers=int(_nested(self.config, "model", "num_encoder_layers", default=2)),
            dropout=float(_nested(self.config, "model", "dropout", default=0.1)),
            k_attr=2,
            num_heads=int(_nested(self.config, "model", "tx_num_heads", default=16)),
            num_decoder_layers=int(_nested(self.config, "model", "num_decoder_layers", default=2)),
            tx_hidden_size=int(_nested(self.config, "model", "tx_hidden_size", default=384)),
            num_agent_types=1,
        ).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("AutoBot")) if isinstance(checkpoint, dict) else checkpoint
        if state_dict is None:
            raise KeyError("AutoBots checkpoint does not contain model_state_dict or AutoBot.")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def predict(self, obs_xy: torch.Tensor, obs_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict joint futures.

        Parameters
        ----------
        obs_xy:
            Tensor shaped `(B, N, T_obs, 2)`.
        obs_mask:
            Boolean tensor shaped `(B, N, T_obs)`.

        Returns
        -------
        trajectories:
            Tensor shaped `(B, K, N, T_pred, 2)`.
        mode_probs:
            Tensor shaped `(B, K)`.
        """
        obs_xy = obs_xy.to(self.device, dtype=torch.float32)
        obs_mask = obs_mask.to(self.device, dtype=torch.bool)
        if obs_xy.ndim != 4 or obs_xy.shape[-1] != 2:
            raise ValueError(f"Expected obs_xy shape (B,N,T,2), got {tuple(obs_xy.shape)}")
        if obs_mask.shape != obs_xy.shape[:-1]:
            raise ValueError(f"Expected obs_mask shape {tuple(obs_xy.shape[:-1])}, got {tuple(obs_mask.shape)}")
        if obs_xy.shape[1] != self.max_agents:
            raise ValueError(f"Expected {self.max_agents} agents, got {obs_xy.shape[1]}")

        obs_xymask = torch.cat((obs_xy, obs_mask.unsqueeze(-1).float()), dim=-1)
        ego_in = obs_xymask[:, 0]
        agents_in = obs_xymask[:, 1:].permute(0, 2, 1, 3)
        agent_types = torch.ones((obs_xy.shape[0], self.max_agents, 1), dtype=torch.float32, device=self.device)
        pred_obs, mode_probs = self.model(ego_in, agents_in, agent_types)
        trajectories = pred_obs[:, :, :, :, :2].permute(2, 0, 3, 1, 4).contiguous()
        return trajectories, mode_probs
