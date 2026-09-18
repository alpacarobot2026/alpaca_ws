#!/usr/bin/env python3
"""Minimal EqMotion pedestrian inference helper.

This ports only the pieces needed for inference from the public
`MediaBrain-SJTU/EqMotion` repository so we can evaluate the model against the
in-lab replay data without modifying any existing code paths.
"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


_YAML_INT_RE = re.compile(r"^[+-]?\d+$")
_YAML_FLOAT_RE = re.compile(
    r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$"
)


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

    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Unsupported YAML indentation at {path}:{lineno}")

        content = line.strip()
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
            child: Dict[str, object] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_yaml_scalar(value)

    return root


class FeatureLearningLayer(nn.Module):
    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        input_c: int,
        hidden_c: int,
        output_c: int,
        edges_in_d: int = 0,
        nodes_att_dim: int = 0,
        act_fn: nn.Module = nn.ReLU(),
        recurrent: bool = True,
        coords_weight: float = 1.0,
        attention: bool = False,
        norm_diff: bool = False,
        tanh: bool = False,
        apply_reasoning: bool = True,
        output_reasoning: bool = False,
        input_reasoning: bool = False,
        category_num: int = 2,
    ) -> None:
        super().__init__()
        del input_c, output_c, edges_in_d, attention, apply_reasoning, output_reasoning
        self.norm_diff = norm_diff
        self.coord_vel = nn.Linear(hidden_c, hidden_c, bias=False)
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.tanh = tanh
        self.hidden_c = hidden_c
        edge_coords_nf = hidden_c
        self.hidden_nf = hidden_nf

        layer = nn.Linear(hidden_nf, hidden_c, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1) * 3.0)
        self.coord_mlp = nn.Sequential(*coord_mlp)

        self.tao = 0.2
        self.category_num = category_num
        self.input_reasoning = input_reasoning

        if input_reasoning:
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_edge + edge_coords_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
            )
            self.category_mlp = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(input_edge + edge_coords_nf, hidden_nf),
                        act_fn,
                        nn.Linear(hidden_nf, hidden_c),
                        act_fn,
                    )
                    for _ in range(category_num)
                ]
            )
            self.factor_mlp = nn.Sequential(
                nn.Linear(hidden_c, hidden_c),
                act_fn,
                nn.Linear(hidden_c, hidden_c),
                act_fn,
            )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        self.layer_q = nn.Linear(hidden_c, hidden_c, bias=False)
        self.layer_k = nn.Linear(hidden_c, hidden_c, bias=False)
        self.mlp_q = nn.Sequential(nn.Linear(hidden_nf, int(hidden_c)), act_fn)

    def edge_model(self, h: torch.Tensor, coord: torch.Tensor, edge_attr=None):
        del edge_attr
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2]
        h1 = h[:, :, None, :].repeat(1, 1, agent_num, 1)
        h2 = h[:, None, :, :].repeat(1, agent_num, 1, 1)
        coord_diff = coord[:, :, None, :, :] - coord[:, None, :, :, :]
        coord_dist = torch.norm(coord_diff, dim=-1)
        edge_feat = torch.cat([h1, h2, coord_dist], dim=-1)
        edge_feat = self.edge_mlp(edge_feat)
        return edge_feat, coord_diff

    def aggregate_coord(self, coord: torch.Tensor, edge_feat: torch.Tensor, coord_diff: torch.Tensor):
        factors = self.coord_mlp(edge_feat).unsqueeze(-1)
        neighbor_effect = torch.sum(factors * coord_diff, dim=2)
        return coord + neighbor_effect

    def aggregate_coord_reasoning(
        self,
        coord: torch.Tensor,
        edge_feat: torch.Tensor,
        coord_diff: torch.Tensor,
        category: torch.Tensor,
        h: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        del edge_feat
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2]
        h1 = h[:, :, None, :].repeat(1, 1, agent_num, 1)
        h2 = h[:, None, :, :].repeat(1, agent_num, 1, 1)
        coord_dist = torch.norm(coord_diff, dim=-1)
        edge_h = torch.cat([h1, h2, coord_dist], dim=-1)
        factors = torch.zeros(batch_size, agent_num, agent_num, channels, device=coord.device, dtype=coord.dtype)
        for i in range(self.category_num - 2):
            factors = factors + category[:, :, :, i : i + 1] * self.category_mlp[i](edge_h)
        factors = self.factor_mlp(factors).unsqueeze(-1)
        neighbor_effect = torch.sum(valid_mask.unsqueeze(-1) * (factors * coord_diff), dim=2)
        return coord + neighbor_effect

    def node_model(self, x: torch.Tensor, edge_feat: torch.Tensor, valid_mask: torch.Tensor):
        batch_size, agent_num = edge_feat.shape[0], edge_feat.shape[1]
        mask = (torch.ones((agent_num, agent_num), device=edge_feat.device, dtype=edge_feat.dtype) - torch.eye(agent_num, device=edge_feat.device, dtype=edge_feat.dtype))
        mask = mask[None, :, :, None].repeat(batch_size, 1, 1, 1)
        aggregated_edge = torch.sum(valid_mask * mask * edge_feat, dim=2)
        out = self.node_mlp(torch.cat([x, aggregated_edge], dim=-1))
        if self.recurrent:
            out = x + out
        return out

    def inner_agent_attention(
        self,
        coord: torch.Tensor,
        h: torch.Tensor,
        valid_mask_agent: torch.Tensor,
        num_valid: torch.Tensor,
    ):
        agent_num = h.shape[1]
        att = self.mlp_q(h).unsqueeze(-1)
        centered = coord - torch.mean(coord * valid_mask_agent, dim=(1, 2), keepdim=True) * (
            agent_num / num_valid[:, None, None, None]
        )
        return att * centered + coord

    def non_linear(self, coord: torch.Tensor, valid_mask_agent: torch.Tensor, num_valid: torch.Tensor):
        agent_num = coord.shape[1]
        coord_mean = torch.mean(coord * valid_mask_agent, dim=(1, 2), keepdim=True) * (
            agent_num / num_valid[:, None, None, None]
        )
        coord = coord - coord_mean
        q = self.layer_q(coord.transpose(2, 3)).transpose(2, 3)
        k = self.layer_k(coord.transpose(2, 3)).transpose(2, 3)
        product = torch.matmul(q.unsqueeze(-2), k.unsqueeze(-1)).squeeze(-1)
        mask = (product >= 0).float()
        eps = 1e-4
        k_norm_sq = torch.sum(k * k, dim=-1, keepdim=True)
        coord = mask * q + (1.0 - mask) * (q - (product / (k_norm_sq + eps)) * k)
        return coord + coord_mean

    def forward(
        self,
        h: torch.Tensor,
        coord: torch.Tensor,
        vel: torch.Tensor,
        valid_mask: torch.Tensor,
        valid_mask_agent: torch.Tensor,
        num_valid: torch.Tensor,
        edge_attr=None,
        node_attr=None,
        category: Optional[torch.Tensor] = None,
    ):
        del node_attr
        edge_feat, coord_diff = self.edge_model(h, coord, edge_attr)
        coord = self.inner_agent_attention(coord, h, valid_mask_agent, num_valid)
        if self.input_reasoning:
            coord = self.aggregate_coord_reasoning(coord, edge_feat, coord_diff, category, h, valid_mask)
        else:
            coord = self.aggregate_coord(coord, edge_feat, coord_diff)
        coord = coord + self.coord_vel(vel.transpose(2, 3)).transpose(2, 3)
        coord = self.non_linear(coord, valid_mask_agent, num_valid)
        h = self.node_model(h, edge_feat, valid_mask)
        return h, coord, category


class EqMotion(nn.Module):
    def __init__(
        self,
        in_node_nf: int,
        in_edge_nf: int,
        hidden_nf: int,
        in_channel: int,
        hid_channel: int,
        out_channel: int,
        device: str = "cpu",
        act_fn: nn.Module = nn.SiLU(),
        n_layers: int = 4,
        coords_weight: float = 1.0,
        recurrent: bool = False,
        norm_diff: bool = False,
        tanh: bool = False,
        num_modes: int = 20,
    ) -> None:
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.num_modes = int(num_modes)

        self.embedding = nn.Linear(in_node_nf, int(self.hidden_nf / 2))
        self.embedding2 = nn.Linear(in_node_nf, int(self.hidden_nf / 2))
        self.coord_trans = nn.Linear(in_channel, int(hid_channel), bias=False)
        self.vel_trans = nn.Linear(in_channel, int(hid_channel), bias=False)

        self.apply_dct = True
        self.validate_reasoning = True
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.category_num = 4
        self.tao = 1.0
        self.given_category = False
        if not self.given_category:
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
            )
            self.coord_mlp = nn.Sequential(
                nn.Linear(hid_channel * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hid_channel * 2),
                act_fn,
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
            )
            self.category_mlp = nn.Sequential(
                nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, self.category_num),
                act_fn,
            )

        for i in range(0, n_layers - 1):
            self.add_module(
                f"gcl_{i}",
                FeatureLearningLayer(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    in_channel,
                    hid_channel,
                    out_channel,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    coords_weight=coords_weight,
                    recurrent=recurrent,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    apply_reasoning=False,
                    input_reasoning=True,
                    category_num=self.category_num,
                ),
            )

        self.predict_head = nn.ModuleList()
        for i in range(self.num_modes):
            self.add_module(
                f"head_{i}",
                FeatureLearningLayer(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    in_channel,
                    hid_channel,
                    out_channel,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    coords_weight=coords_weight,
                    recurrent=recurrent,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    apply_reasoning=False,
                    input_reasoning=True,
                    category_num=self.category_num,
                ),
            )
            self.predict_head.append(nn.Linear(hid_channel, out_channel, bias=False))

        self.to(self.device)

    @staticmethod
    def get_dct_matrix(n: int, x: torch.Tensor):
        import numpy as np

        dct_m = np.eye(n)
        for k in np.arange(n):
            for i in np.arange(n):
                w = (2.0 / n) ** 0.5
                if k == 0:
                    w = (1.0 / n) ** 0.5
                dct_m[k, i] = w * np.cos(np.pi * (i + 0.5) * k / n)
        idct_m = np.linalg.inv(dct_m)
        return torch.from_numpy(dct_m).type_as(x), torch.from_numpy(idct_m).type_as(x)

    def calc_category(self, h: torch.Tensor, coord: torch.Tensor, valid_mask: torch.Tensor):
        batch_size, agent_num = coord.shape[0], coord.shape[1]
        h1 = h[:, :, None, :].repeat(1, 1, agent_num, 1)
        h2 = h[:, None, :, :].repeat(1, agent_num, 1, 1)
        coord_diff = coord[:, :, None, :, :] - coord[:, None, :, :, :]
        coord_dist = torch.norm(coord_diff, dim=-1)
        coord_dist = self.coord_mlp(coord_dist)
        edge_feat_input = torch.cat([h1, h2, coord_dist], dim=-1)
        edge_feat = self.edge_mlp(edge_feat_input)
        mask = (torch.ones((agent_num, agent_num), device=edge_feat.device, dtype=edge_feat.dtype) - torch.eye(agent_num, device=edge_feat.device, dtype=edge_feat.dtype))
        mask = mask[None, :, :, None].repeat(batch_size, 1, 1, 1)
        node_new = self.node_mlp(torch.cat([h, torch.sum(valid_mask * mask * edge_feat, dim=2)], dim=-1))
        node_new1 = node_new[:, :, None, :].repeat(1, 1, agent_num, 1)
        node_new2 = node_new[:, None, :, :].repeat(1, agent_num, 1, 1)
        edge_feat_input_new = torch.cat([node_new1, node_new2, coord_dist], dim=-1)
        return F.softmax(self.category_mlp(edge_feat_input_new) / self.tao, dim=-1)

    @staticmethod
    def get_valid_mask(num_valid: torch.Tensor, agent_num: int):
        batch_size = num_valid.shape[0]
        valid_mask = torch.zeros((batch_size, agent_num, agent_num), device=num_valid.device, dtype=torch.float32)
        for i in range(batch_size):
            valid_mask[i, : num_valid[i], : num_valid[i]] = 1.0
        return valid_mask.unsqueeze(-1)

    @staticmethod
    def get_valid_mask2(num_valid: torch.Tensor, agent_num: int):
        batch_size = num_valid.shape[0]
        valid_mask = torch.zeros((batch_size, agent_num), device=num_valid.device, dtype=torch.float32)
        for i in range(batch_size):
            valid_mask[i, : num_valid[i]] = 1.0
        return valid_mask.unsqueeze(-1).unsqueeze(-1)

    def _history_step_mask(self, history_mask: Optional[torch.Tensor], valid_agent_mask: torch.Tensor, x: torch.Tensor):
        if history_mask is None:
            return None
        if history_mask.shape != x.shape[:3]:
            raise ValueError(
                f"history_mask shape {tuple(history_mask.shape)} must match x shape {tuple(x.shape[:3])}."
            )
        history_step_mask = history_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        return history_step_mask * valid_agent_mask.to(device=x.device, dtype=x.dtype)

    def _center_for_dct(
        self,
        x: torch.Tensor,
        vel: torch.Tensor,
        valid_agent_mask: torch.Tensor,
        num_valid: torch.Tensor,
        history_step_mask: Optional[torch.Tensor] = None,
    ):
        agent_num = x.shape[1]
        if history_step_mask is None:
            x_center = torch.mean(x * valid_agent_mask, dim=(1, 2), keepdim=True) * (
                agent_num / num_valid[:, None, None, None]
            )
            x = x - x_center
            return x, vel, x_center

        x = x * history_step_mask
        vel = vel * history_step_mask
        observed_count = history_step_mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        x_center = torch.sum(x, dim=(1, 2), keepdim=True) / observed_count
        x = (x - x_center) * history_step_mask
        vel = vel * history_step_mask
        return x, vel, x_center

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        vel: torch.Tensor,
        num_valid: torch.Tensor,
        edge_attr=None,
        history_mask: Optional[torch.Tensor] = None,
    ):
        vel_pre = torch.zeros_like(vel)
        vel_pre[:, :, 1:] = vel[:, :, :-1]
        vel_pre[:, :, 0] = vel[:, :, 0]
        eps = 1e-6
        vel_cosangle = torch.sum(vel_pre * vel, dim=-1) / (
            (torch.norm(vel_pre, dim=-1) + eps) * (torch.norm(vel, dim=-1) + eps)
        )
        vel_angle = torch.acos(torch.clamp(vel_cosangle, -1.0, 1.0))

        batch_size, agent_num = x.shape[0], x.shape[1]
        valid_agent_mask = self.get_valid_mask2(num_valid, agent_num).type_as(h)
        history_step_mask = self._history_step_mask(history_mask, valid_agent_mask, x)
        if history_step_mask is not None:
            h = h * history_step_mask.squeeze(-1).type_as(h)
            vel = vel * history_step_mask

        if self.apply_dct:
            x, vel, x_center = self._center_for_dct(x, vel, valid_agent_mask, num_valid, history_step_mask)
            dct_m, _ = self.get_dct_matrix(self.in_channel, x)
            _, idct_m = self.get_dct_matrix(self.out_channel, x)
            dct_m = dct_m[None, None, :, :].repeat(batch_size, agent_num, 1, 1)
            idct_m = idct_m[None, None, :, :].repeat(batch_size, agent_num, 1, 1)
            x = torch.matmul(dct_m, x)
            vel = torch.matmul(dct_m, vel)
        else:
            idct_m = None
            x_center = None

        h = self.embedding(h)
        vel_angle_embedding = self.embedding2(vel_angle)
        h = torch.cat([h, vel_angle_embedding], dim=-1)

        x_mean = torch.mean(torch.mean(x * valid_agent_mask, dim=-2, keepdim=True), dim=-3, keepdim=True) * (
            agent_num / num_valid[:, None, None, None]
        )
        x = self.coord_trans((x - x_mean).transpose(2, 3)).transpose(2, 3) + x_mean
        vel = self.vel_trans(vel.transpose(2, 3)).transpose(2, 3)
        x_cat = torch.cat([x, vel], dim=-2)

        valid_mask = self.get_valid_mask(num_valid, agent_num).type_as(h)
        category = self.calc_category(h, x_cat, valid_mask)

        for i in range(0, self.n_layers - 1):
            h, x, _ = self._modules[f"gcl_{i}"](h, x, vel, valid_mask, valid_agent_mask, num_valid, edge_attr=edge_attr, category=category)

        all_out = []
        for i in range(self.num_modes):
            _, out, _ = self._modules[f"head_{i}"](h, x, vel, valid_mask, valid_agent_mask, num_valid, edge_attr=edge_attr, category=category)
            out_mean = torch.mean(torch.mean(out * valid_agent_mask, dim=-2, keepdim=True), dim=-3, keepdim=True) * (
                agent_num / num_valid[:, None, None, None]  
            )
            out = self.predict_head[i]((out - out_mean).transpose(2, 3)).transpose(2, 3) + out_mean
            all_out.append(out[:, :, None, :, :])

        x = torch.cat(all_out, dim=2).view(batch_size, agent_num, self.num_modes, self.out_channel, -1)
        if self.apply_dct and idct_m is not None and x_center is not None:
            idct_m = idct_m[:, :, None, :, :]
            x = torch.matmul(idct_m, x)
            x = x + x_center.unsqueeze(2)
        return x, None


def _extract_checkpoint_bundle(zip_path: Path, extract_root: Optional[Path] = None) -> Path:
    if extract_root is None:
        extract_root = Path(tempfile.gettempdir()) / "eqmotion_bundle"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        top_levels = sorted({name.split("/", 1)[0] for name in zf.namelist() if "/" in name})
        if not top_levels:
            raise ValueError(f"No bundle directory found in {zip_path}")
        bundle_name = top_levels[0]
        out_dir = extract_root / bundle_name
        if not out_dir.exists():
            zf.extractall(extract_root)
        return out_dir


class EqMotionPredictor:
    def __init__(
        self,
        bundle_path: Path,
        checkpoint_name: str = "best_models_ade_eqmotion5.pth",
        device: Optional[str] = None,
    ) -> None:
        self.bundle_root = self._resolve_bundle_root(Path(bundle_path))
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        config = self._load_config(self.bundle_root)
        self.config = config

        if "model" in config and isinstance(config["model"], dict):
            model_cfg = config.get("model", {})
            dataset_cfg = config.get("dataset", {})
            eq_cfg = model_cfg.get("eqmotion", {})
            data_cfg = dataset_cfg.get("eth_ucy", {})
            subset = str(data_cfg.get("subset", "zara1")).lower()
            channels = eq_cfg.get("channels", None)
            hidden_size = int(channels) if channels is not None else (128 if subset == "zara1" else 64)
            self.obs_len = int(data_cfg.get("obs_len", 8))
            self.pred_len = int(data_cfg.get("pred_len", 12))
            self.num_modes = int(model_cfg.get("num_modes", 20))
            self.max_agents = int(data_cfg.get("max_agents", 30))
            self.history_fill_strategy = str(eq_cfg.get("history_fill_strategy", "carry")).lower()
        else:
            eq_cfg = {
                "nf": int(config.get("eqmotion_nf", 64) or 64),
                "n_layers": int(config.get("eqmotion_n_layers", 4) or 4),
                "norm_diff": bool(config.get("norm_diff", False)),
                "tanh": bool(config.get("tanh", False)),
            }
            hidden_size = int(config.get("hidden_size", 128))
            self.obs_len = int(config.get("eth_ucy_obs_len", 8))
            self.pred_len = int(config.get("eth_ucy_pred_len", 12))
            self.num_modes = int(config.get("num_modes", 20))
            self.max_agents = int(config.get("eth_ucy_max_agents", 30))
            self.history_fill_strategy = str(config.get("history_fill_strategy", "zero")).lower()

        self.model = EqMotion(
            in_node_nf=self.obs_len,
            in_edge_nf=2,
            hidden_nf=int(eq_cfg.get("nf", 64)),
            in_channel=self.obs_len,
            hid_channel=hidden_size,
            out_channel=self.pred_len,
            device=str(self.device),
            n_layers=int(eq_cfg.get("n_layers", 4)),
            recurrent=True,
            norm_diff=bool(eq_cfg.get("norm_diff", False)),
            tanh=bool(eq_cfg.get("tanh", False)),
            num_modes=self.num_modes,
        ).to(self.device)
        self.model.eval()

        checkpoint_path = self._find_bundle_file(checkpoint_name)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        self.model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _resolve_bundle_root(bundle_path: Path) -> Path:
        if bundle_path.is_dir():
            return bundle_path
        if bundle_path.suffix.lower() == ".zip":
            return _extract_checkpoint_bundle(bundle_path)
        raise ValueError(f"Expected EqMotion bundle directory or .zip, got: {bundle_path}")

    @staticmethod
    def _load_config(bundle_root: Path) -> Dict[str, object]:
        for cfg_yaml in (
            bundle_root / "config_eqmotion5.yaml",
            bundle_root / "models/config_eqmotion5.yaml",
            bundle_root / "config_eqmotion.yaml",
            bundle_root / "models/config_eqmotion.yaml",
            bundle_root / "config.yaml",
            bundle_root / "models/config.yaml",
        ):
            if cfg_yaml.exists():
                break
        else:
            cfg_yaml = bundle_root / "config_eqmotion.yaml"
        if cfg_yaml.exists():
            try:
                import yaml
            except ModuleNotFoundError:
                data = _load_simple_yaml_mapping(cfg_yaml)
            else:
                data = yaml.safe_load(cfg_yaml.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        cfg_json = bundle_root / "config.json"
        if not cfg_json.exists():
            cfg_json = bundle_root / "models/config.json"
        if cfg_json.exists():
            return json.loads(cfg_json.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"Missing config.json under {bundle_root}")

    def _find_bundle_file(self, filename: str) -> Path:
        direct_path = self.bundle_root / filename
        if direct_path.exists():
            return direct_path
        return self.bundle_root / "models" / filename

    @torch.no_grad()
    def predict(
        self,
        history_xy: torch.Tensor,
        num_valid: int,
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run EqMotion on one scene history.

        Parameters
        ----------
        history_xy:
            Shape `(N, T_obs, 2)` with the target agent at index 0.
        num_valid:
            Number of valid agents in `history_xy`.

        Returns
        -------
        torch.Tensor
            Shape `(N, num_modes, T_pred, 2)`.
        """
        if history_xy.ndim != 3 or history_xy.shape[-1] != 2:
            raise ValueError(f"Expected history shape (N,T,2), got {tuple(history_xy.shape)}")

        loc = history_xy.unsqueeze(0).to(self.device, dtype=torch.float32)
        history_mask_tensor = None
        if history_mask is not None:
            if history_mask.shape != history_xy.shape[:2]:
                raise ValueError(
                    f"Expected history_mask shape {tuple(history_xy.shape[:2])}, got {tuple(history_mask.shape)}"
                )
            history_mask_tensor = history_mask.unsqueeze(0).to(self.device, dtype=torch.bool)
        vel = torch.zeros_like(loc)
        vel[:, :, 1:] = loc[:, :, 1:] - loc[:, :, :-1]
        if history_mask_tensor is not None and loc.shape[2] > 1:
            transition_mask = history_mask_tensor[:, :, 1:] & history_mask_tensor[:, :, :-1]
            vel[:, :, 1:] = vel[:, :, 1:] * transition_mask.unsqueeze(-1)
            vel[:, :, 0] = vel[:, :, 1]
        elif loc.shape[2] > 1:
            vel[:, :, 0] = vel[:, :, 1]
        nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
        num_valid_tensor = torch.tensor([int(num_valid)], device=self.device, dtype=torch.int64)
        pred, _ = self.model(nodes, loc, vel, num_valid_tensor, history_mask=history_mask_tensor)
        return pred.squeeze(0).detach().cpu()
