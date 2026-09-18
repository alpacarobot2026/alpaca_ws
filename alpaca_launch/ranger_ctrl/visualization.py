#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from scipy.optimize import linear_sum_assignment


# ----------------------------
# Shared utilities
# ----------------------------

def _latest_run_dir(base_dir: Path) -> Path:
    candidates = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not candidates:
        raise FileNotFoundError(f"No run_*/ directories found in {base_dir}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def _run_dirs(base_dir: Path, runs=None):
    if runs:
        out = []
        for r in runs:
            p = base_dir / r
            if p.is_dir():
                out.append(p)
        return sorted(out, key=lambda p: p.name)
    return sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], key=lambda p: p.name)


def _load_pickle(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _entry_stamp(entry) -> float:
    if not isinstance(entry, dict):
        return float("nan")

    source = entry.get("stamp_source", float("nan"))
    receive = entry.get("stamp_receive", float("nan"))
    stamp = entry.get("stamp", float("nan"))

    if np.isfinite(source) and source > 0.0:
        return float(source)
    if np.isfinite(stamp) and stamp > 0.0:
        return float(stamp)
    if np.isfinite(receive) and receive > 0.0:
        return float(receive)
    return float("nan")


def _summarize_errors(errors: np.ndarray):
    if errors is None or len(errors) == 0:
        return {
            "n": 0,
            "rmse_m": float("nan"),
            "mae_m": float("nan"),
            "median_m": float("nan"),
            "p95_m": float("nan"),
            "max_m": float("nan"),
        }

    e = np.asarray(errors, dtype=float)
    return {
        "n": int(len(e)),
        "rmse_m": float(np.sqrt(np.mean(e ** 2))),
        "mae_m": float(np.mean(np.abs(e))),
        "median_m": float(np.median(e)),
        "p95_m": float(np.percentile(e, 95)),
        "max_m": float(np.max(e)),
    }


def _summarize_series(values: np.ndarray, unit_key_suffix: str = "m"):
    if values is None or len(values) == 0:
        return {
            "n": 0,
            f"mean_{unit_key_suffix}": float("nan"),
            f"median_{unit_key_suffix}": float("nan"),
            f"p95_{unit_key_suffix}": float("nan"),
            f"max_{unit_key_suffix}": float("nan"),
        }
    v = np.asarray(values, dtype=float)
    return {
        "n": int(len(v)),
        f"mean_{unit_key_suffix}": float(np.mean(v)),
        f"median_{unit_key_suffix}": float(np.median(v)),
        f"p95_{unit_key_suffix}": float(np.percentile(v, 95)),
        f"max_{unit_key_suffix}": float(np.max(v)),
    }


# ----------------------------
# Time-series helpers
# ----------------------------

def _clean_series(t, xy):
    t = np.asarray(t, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if len(t) == 0 or len(xy) == 0:
        return None

    mask = np.isfinite(t) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    t = t[mask]
    xy = xy[mask]
    if len(t) == 0:
        return None

    order = np.argsort(t)
    t = t[order]
    xy = xy[order]

    # Deduplicate timestamps (strictly increasing time required for interpolation).
    t_unique, idx = np.unique(t, return_index=True)
    t = t_unique
    xy = xy[idx]

    return {"t": t, "xy": xy}


def _interp_xy(series, tq):
    if series is None:
        return None

    t = series["t"]
    xy = series["xy"]
    if len(t) < 2:
        return None

    tq = np.asarray(tq, dtype=float)
    in_range = (tq >= t[0]) & (tq <= t[-1]) & np.isfinite(tq)
    if not np.any(in_range):
        return None

    tq_valid = tq[in_range]
    xq = np.interp(tq_valid, t, xy[:, 0])
    yq = np.interp(tq_valid, t, xy[:, 1])

    return {
        "mask": in_range,
        "xy": np.stack([xq, yq], axis=1),
    }


def _build_gt_series(gt_arr):
    # gt rows: [id, x, y, stamp]
    if gt_arr is None or len(gt_arr) == 0:
        return {}

    gt_arr = np.asarray(gt_arr, dtype=float)
    out = {}
    for obj_id in (0, 1, 2):
        mask = gt_arr[:, 0] == obj_id
        sub = gt_arr[mask]
        if len(sub) == 0:
            continue
        out[obj_id] = _clean_series(sub[:, 3], sub[:, 1:3])
    return out


def _build_amcl_series(amcl_arr):
    # amcl rows: [stamp, x, y, yaw]
    if amcl_arr is None or len(amcl_arr) == 0:
        return None

    arr = np.asarray(amcl_arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return None

    base = _clean_series(arr[:, 0], arr[:, 1:3])
    if base is None:
        return None

    # Keep yaw aligned with cleaned timestamps by nearest lookup on original array.
    t = base["t"]
    orig_t = arr[:, 0]
    orig_yaw = arr[:, 3]
    yaw = []
    for ti in t:
        idx = int(np.argmin(np.abs(orig_t - ti)))
        yaw.append(float(orig_yaw[idx]))
    base["yaw"] = np.asarray(yaw, dtype=float)
    return base


def _nearest_xy(series, t):
    if series is None:
        return None
    ts = series["t"]
    if ts is None or len(ts) == 0:
        return None
    idx = int(np.argmin(np.abs(ts - t)))
    return series["xy"][idx]


def _sample_xy_at_t(series, t, mode="interp", max_time_gap_s=None):
    if series is None:
        return None
    ts = series["t"]
    if ts is None or len(ts) == 0 or not np.isfinite(t):
        return None

    # Nearest-time guard for timestamp mismatch control.
    idx = int(np.argmin(np.abs(ts - t)))
    nearest_gap = float(abs(ts[idx] - t))
    if max_time_gap_s is not None and np.isfinite(max_time_gap_s) and nearest_gap > max_time_gap_s:
        return None

    if mode == "nearest":
        return series["xy"][idx]

    # Default: linear interpolation at detector timestamp.
    g = _interp_xy(series, np.asarray([t], dtype=float))
    if g is None or not np.any(g["mask"]):
        return None
    return g["xy"][0]


def _nearest_amcl(amcl_series, t):
    if amcl_series is None:
        return None
    ts = amcl_series["t"]
    if len(ts) == 0:
        return None
    idx = int(np.argmin(np.abs(ts - t)))
    return amcl_series["xy"][idx], float(amcl_series["yaw"][idx])


def _parse_simpletrack_frame(entry):
    # Keep IDs 1 and 2 only
    out = {}
    if not isinstance(entry, dict):
        return out

    for det in entry.get("detections", []):
        if not isinstance(det, dict):
            continue
        raw_id = str(det.get("id", "")).strip()
        if raw_id not in {"1", "2"}:
            continue
        cx = det.get("cx", None)
        cy = det.get("cy", None)
        if cx is None or cy is None:
            continue
        try:
            out[int(raw_id)] = (float(cx), float(cy))
        except Exception:
            continue
    return out


def _parse_simpletrack_all_detections(entry):
    # Return all valid detections as (track_id, xy), not only ids {1,2}.
    out = []
    if not isinstance(entry, dict):
        return out
    for det in entry.get("detections", []):
        if not isinstance(det, dict):
            continue
        cx = det.get("cx", None)
        cy = det.get("cy", None)
        if cx is None or cy is None:
            continue
        try:
            x = float(cx)
            y = float(cy)
        except Exception:
            continue

        raw_id = det.get("id", None)
        track_id = None
        if raw_id is not None:
            s = str(raw_id).strip()
            if s:
                try:
                    track_id = int(s)
                except Exception:
                    track_id = s
        out.append((track_id, np.asarray([x, y], dtype=float)))
    return out


def _prediction_tensor(entry):
    tensor = entry.get("tensor", None) if isinstance(entry, dict) else None
    if tensor is not None:
        arr = np.asarray(tensor, dtype=float)
        if arr.ndim == 3 and arr.shape[2] == 2:
            return arr

    if not isinstance(entry, dict):
        return None

    data = entry.get("data", None)
    shape = entry.get("shape", None)
    if data is None or shape is None:
        return None

    try:
        arr = np.asarray(data, dtype=float)
        shape = [int(v) for v in shape]
        if len(shape) == 3 and int(np.prod(shape)) == arr.size and shape[2] == 2:
            return arr.reshape(shape)
    except Exception:
        return None
    return None


def _nearest_prediction(pred_entries, pred_ts, t):
    if pred_ts is None or len(pred_ts) == 0:
        return None
    idx = int(np.argmin(np.abs(pred_ts - t)))
    return _prediction_tensor(pred_entries[idx])


def _prediction_anchor_times(pred_arr, simpletrack_arr=None):
    # Derive source timestamps for prediction entries.
    # Priority:
    # 1) prediction stamp_source (if available),
    # 2) nearest simpletrack stamp_source by receive-time,
    # 3) prediction stamp/receive as fallback.
    if pred_arr is None:
        return np.zeros((0,), dtype=float)

    pred_entries = list(pred_arr)
    t_anchor = np.full((len(pred_entries),), np.nan, dtype=float)
    pred_recv = np.full((len(pred_entries),), np.nan, dtype=float)

    for i, e in enumerate(pred_entries):
        if not isinstance(e, dict):
            continue
        src = e.get("stamp_source", float("nan"))
        recv = e.get("stamp_receive", float("nan"))
        pred_recv[i] = float(recv) if np.isfinite(recv) else float("nan")
        if np.isfinite(src):
            t_anchor[i] = float(src)

    # Fill missing source times from nearest simpletrack source stamp.
    if simpletrack_arr is not None and len(simpletrack_arr) > 0:
        st_recv = []
        st_src = []
        for e in simpletrack_arr:
            if not isinstance(e, dict):
                continue
            r = e.get("stamp_receive", float("nan"))
            s = e.get("stamp_source", float("nan"))
            if np.isfinite(r) and np.isfinite(s):
                st_recv.append(float(r))
                st_src.append(float(s))
        if st_recv:
            st_recv = np.asarray(st_recv, dtype=float)
            st_src = np.asarray(st_src, dtype=float)
            order = np.argsort(st_recv)
            st_recv = st_recv[order]
            st_src = st_src[order]

            missing = np.where(~np.isfinite(t_anchor))[0]
            for i in missing:
                r = pred_recv[i]
                if not np.isfinite(r):
                    continue
                j = int(np.searchsorted(st_recv, r))
                cand = []
                if j > 0:
                    cand.append(j - 1)
                if j < len(st_recv):
                    cand.append(j)
                if not cand:
                    continue
                jj = min(cand, key=lambda k: abs(st_recv[k] - r))
                t_anchor[i] = float(st_src[jj])

    # Final fallback to generic entry stamp.
    for i, e in enumerate(pred_entries):
        if np.isfinite(t_anchor[i]):
            continue
        t = _entry_stamp(e)
        if np.isfinite(t):
            t_anchor[i] = float(t)

    return t_anchor


def _compute_bounds(simpletrack_arr, gt_series, amcl_series):
    pts = []

    if simpletrack_arr is not None:
        for entry in simpletrack_arr:
            dets = _parse_simpletrack_frame(entry)
            for _, xy in dets.items():
                pts.append([xy[0], xy[1]])

    for series in gt_series.values():
        if series is not None and len(series["xy"]) > 0:
            pts.append(series["xy"])

    if amcl_series is not None and len(amcl_series["xy"]) > 0:
        pts.append(amcl_series["xy"])

    if not pts:
        return (-5.0, 5.0, -5.0, 5.0)

    pts = np.vstack([np.asarray(p, dtype=float).reshape(-1, 2) for p in pts])
    min_x = float(np.min(pts[:, 0]))
    max_x = float(np.max(pts[:, 0]))
    min_y = float(np.min(pts[:, 1]))
    max_y = float(np.max(pts[:, 1]))

    pad_x = max(0.5, 0.1 * (max_x - min_x + 1e-6))
    pad_y = max(0.5, 0.1 * (max_y - min_y + 1e-6))
    return (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)


# ----------------------------
# Metrics mode
# ----------------------------

def _localization_metrics(amcl_arr, gt_series):
    amcl = _build_amcl_series(amcl_arr)
    robot_gt = gt_series.get(0, None)
    if amcl is None or robot_gt is None:
        return _summarize_errors(np.asarray([]))

    interped = _interp_xy(robot_gt, amcl["t"])
    if interped is None:
        return _summarize_errors(np.asarray([]))

    mask = interped["mask"]
    amcl_xy = amcl["xy"][mask]
    gt_xy = interped["xy"]
    err = np.linalg.norm(amcl_xy - gt_xy, axis=1)
    out = _summarize_errors(err)
    out["time_overlap_start"] = float(max(amcl["t"][0], robot_gt["t"][0]))
    out["time_overlap_end"] = float(min(amcl["t"][-1], robot_gt["t"][-1]))

    # Relative trajectory error (translation) between consecutive aligned steps.
    if len(amcl_xy) >= 2 and len(gt_xy) >= 2:
        d_amcl = np.diff(amcl_xy, axis=0)
        d_gt = np.diff(gt_xy, axis=0)
        rpe = np.linalg.norm(d_amcl - d_gt, axis=1)
        out["rpe_trans_rmse_m"] = float(np.sqrt(np.mean(rpe ** 2)))
        out["rpe_trans_mae_m"] = float(np.mean(np.abs(rpe)))
        out["rpe_trans_p95_m"] = float(np.percentile(rpe, 95))
    else:
        out["rpe_trans_rmse_m"] = float("nan")
        out["rpe_trans_mae_m"] = float("nan")
        out["rpe_trans_p95_m"] = float("nan")
    return out


def _associate_detections_to_gt(
    simpletrack_arr,
    gt_series,
    match_thresh_m=0.5,
    include_all_detections=False,
    gt_time_mode="interp",
    gt_max_gap_s=None,
):
    # CLEAR-style frame association for GT ids {1,2} using distance threshold.
    # Returns aggregate counts and per-id matched localization errors.
    out = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "gt_count": 0,
        "id_switches": 0,
        "loc_err_tp": [],
        "loc_err_tp_by_id": {1: [], 2: []},
    }

    gt1 = gt_series.get(1, None)
    gt2 = gt_series.get(2, None)
    if simpletrack_arr is None or len(simpletrack_arr) == 0 or (gt1 is None and gt2 is None):
        return out

    entries = []
    for entry in simpletrack_arr:
        t = _entry_stamp(entry)
        if np.isfinite(t):
            entries.append((float(t), entry))
    if not entries:
        return out

    entries.sort(key=lambda x: x[0])
    prev_match = {1: None, 2: None}

    for t, entry in entries:
        gt_items = []
        for hid, gt in ((1, gt1), (2, gt2)):
            if gt is None:
                continue
            gxy = _sample_xy_at_t(gt, t, mode=gt_time_mode, max_time_gap_s=gt_max_gap_s)
            if gxy is None:
                continue
            gt_items.append((hid, gxy))
        out["gt_count"] += len(gt_items)

        if include_all_detections:
            det_items = _parse_simpletrack_all_detections(entry)
        else:
            # Targeted evaluation scope for this in-lab setup:
            # compare only tracker IDs mapped to GT humans {1,2},
            # with identity-locked correspondence (id_1->gt1, id_2->gt2).
            dets12 = _parse_simpletrack_frame(entry)
            for hid, gxy in gt_items:
                dxy = dets12.get(hid, None)
                if dxy is None:
                    out["fn"] += 1
                    continue
                dxy = np.asarray(dxy, dtype=float)
                dist = float(np.linalg.norm(dxy - gxy))
                if dist <= match_thresh_m:
                    out["tp"] += 1
                    out["loc_err_tp"].append(dist)
                    if hid in out["loc_err_tp_by_id"]:
                        out["loc_err_tp_by_id"][hid].append(dist)
                    # No ID switch when IDs are identity-locked by definition.
                    prev_match[hid] = hid
                else:
                    # Detection exists with correct ID but outside gate:
                    # contributes one FP and one FN under CLEAR counting.
                    out["fp"] += 1
                    out["fn"] += 1
            # Any extra detections among {1,2} that do not have corresponding GT
            # at this frame count as false positives.
            for did in dets12.keys():
                if did not in {hid for hid, _ in gt_items}:
                    out["fp"] += 1
            continue

        n_gt = len(gt_items)
        n_det = len(det_items)

        if n_gt == 0:
            out["fp"] += n_det
            continue
        if n_det == 0:
            out["fn"] += n_gt
            continue

        # Build cost matrix (n_gt x n_det).
        cost = np.full((n_gt, n_det), np.inf, dtype=float)
        for gi, (_, gxy) in enumerate(gt_items):
            for dj, (_, dxy) in enumerate(det_items):
                d = float(np.linalg.norm(dxy - gxy))
                if d <= match_thresh_m:
                    cost[gi, dj] = d

        # Small-cardinality exact assignment (n_gt <= 2 in this dataset).
        matched = []  # list of (gi, dj, dist)
        if n_gt == 1:
            gi = 0
            dj = int(np.argmin(cost[gi]))
            if np.isfinite(cost[gi, dj]):
                matched.append((gi, dj, float(cost[gi, dj])))
        else:
            best = None
            # 2-match candidates.
            for d0 in range(n_det):
                for d1 in range(n_det):
                    if d1 == d0:
                        continue
                    c0 = cost[0, d0]
                    c1 = cost[1, d1]
                    if np.isfinite(c0) and np.isfinite(c1):
                        c = float(c0 + c1)
                        if best is None or c < best[0]:
                            best = (c, [(0, d0, float(c0)), (1, d1, float(c1))])
            # 1-match fallbacks.
            for gi in (0, 1):
                for dj in range(n_det):
                    c = cost[gi, dj]
                    if np.isfinite(c):
                        if best is None or float(c) < best[0]:
                            best = (float(c), [(gi, dj, float(c))])
            if best is not None:
                matched = best[1]

        matched_g = {gi for gi, _, _ in matched}
        matched_d = {dj for _, dj, _ in matched}

        out["tp"] += len(matched)
        out["fn"] += (n_gt - len(matched))
        out["fp"] += (n_det - len(matched))

        # Localization quality for matched true positives and ID switch counting.
        for gi, dj, dist in matched:
            hid = int(gt_items[gi][0])
            det_id = det_items[dj][0]
            out["loc_err_tp"].append(dist)
            if hid in out["loc_err_tp_by_id"]:
                out["loc_err_tp_by_id"][hid].append(dist)

            # Count IDSW only for stable, comparable detector track IDs.
            if isinstance(det_id, (int, np.integer)):
                if prev_match[hid] is not None and prev_match[hid] != int(det_id):
                    out["id_switches"] += 1
                prev_match[hid] = int(det_id)

    return out


def _remap_detections_to_gt_ids(
    simpletrack_arr,
    gt_series,
    match_thresh_m=0.5,
    gt_time_mode="interp",
    gt_max_gap_s=None,
):
    """
    Run the same per-frame spatial cost-matrix matching as _associate_detections_to_gt
    (include_all_detections=True path), but return a remapped frame array where each
    matched detection's 'id' is replaced with the GT human ID (1 or 2).

    Unmatched detections are dropped. Suitable for feeding into
    _tracking_continuity_metrics when raw tracker IDs are not pre-mapped to GT IDs.
    """
    gt1 = gt_series.get(1, None)
    gt2 = gt_series.get(2, None)
    if simpletrack_arr is None or len(simpletrack_arr) == 0 or (gt1 is None and gt2 is None):
        return np.asarray([], dtype=object)

    entries = []
    for entry in simpletrack_arr:
        t = _entry_stamp(entry)
        if np.isfinite(t):
            entries.append((float(t), entry))
    entries.sort(key=lambda x: x[0])

    remapped = []
    for t, entry in entries:
        gt_items = []
        for hid, gt in ((1, gt1), (2, gt2)):
            if gt is None:
                continue
            gxy = _sample_xy_at_t(gt, t, mode=gt_time_mode, max_time_gap_s=gt_max_gap_s)
            if gxy is None:
                continue
            gt_items.append((hid, np.asarray(gxy, dtype=float)))

        det_items = _parse_simpletrack_all_detections(entry)
        n_gt  = len(gt_items)
        n_det = len(det_items)

        relabelled = []
        if n_gt > 0 and n_det > 0:
            cost = np.full((n_gt, n_det), np.inf, dtype=float)
            for gi, (_, gxy) in enumerate(gt_items):
                for dj, (_, dxy) in enumerate(det_items):
                    d = float(np.linalg.norm(dxy - gxy))
                    if d <= match_thresh_m:
                        cost[gi, dj] = d

            matched = []
            if n_gt == 1:
                gi = 0
                dj = int(np.argmin(cost[gi]))
                if np.isfinite(cost[gi, dj]):
                    matched.append((gi, dj))
            else:
                best = None
                for d0 in range(n_det):
                    for d1 in range(n_det):
                        if d1 == d0:
                            continue
                        c0, c1 = cost[0, d0], cost[1, d1]
                        if np.isfinite(c0) and np.isfinite(c1):
                            c = float(c0 + c1)
                            if best is None or c < best[0]:
                                best = (c, [(0, d0), (1, d1)])
                for gi in (0, 1):
                    for dj in range(n_det):
                        c = cost[gi, dj]
                        if np.isfinite(c):
                            if best is None or float(c) < best[0]:
                                best = (float(c), [(gi, dj)])
                if best is not None:
                    matched = best[1]

            for gi, dj in matched:
                hid = int(gt_items[gi][0])
                _, dxy = det_items[dj]
                relabelled.append({'id': str(hid), 'cx': float(dxy[0]), 'cy': float(dxy[1])})

        remapped.append({**entry, 'detections': relabelled})

    return np.asarray(remapped, dtype=object)


def _remap_detections_hungarian(
    tracker_arr,
    gt_series,
    match_thresh_m=0.5,
    switch_penalty=0.35,
    raw_id_memory_sec=3.0,
    gt_time_mode="interp",
    gt_max_gap_s=None,
):
    """
    Per-frame Hungarian assignment with switch penalty, mirroring
    MocapTrackerAssociationNode logic but applied offline to a batch
    of tracker frames.

    For each frame:
      1. Look up GT positions for human_1 and human_2 at the frame timestamp.
      2. Build cost matrix (n_gt x n_det) using Euclidean distance + switch
         penalty when a raw tracker ID previously matched a different GT label.
      3. Run linear_sum_assignment (Hungarian).
      4. Gate accepted matches by pure geometric distance (match_thresh_m).
      5. Update per-raw-ID label memory (expires after raw_id_memory_sec).

    Returns a remapped frame array where each matched detection's 'id' is
    replaced with the GT human label ('1' or '2'). Unmatched detections dropped.
    """
    gt1 = gt_series.get(1, None)
    gt2 = gt_series.get(2, None)
    if tracker_arr is None or len(tracker_arr) == 0 or (gt1 is None and gt2 is None):
        return np.asarray([], dtype=object)

    entries = []
    for entry in tracker_arr:
        t = _entry_stamp(entry)
        if np.isfinite(t):
            entries.append((float(t), entry))
    entries.sort(key=lambda x: x[0])

    # Stateful ID memory (mirrors node's _raw_id_to_label / _raw_id_last_seen_sec)
    raw_id_to_label: dict = {}
    raw_id_last_seen: dict = {}

    remapped = []
    for t, entry in entries:
        # Expire stale IDs
        stale = [rid for rid, ts in raw_id_last_seen.items() if t - ts > raw_id_memory_sec]
        for rid in stale:
            raw_id_to_label.pop(rid, None)
            raw_id_last_seen.pop(rid, None)

        # GT positions at this timestamp
        gt_items = []  # list of (label_str, xy_array)
        for hid, gt in ((1, gt1), (2, gt2)):
            if gt is None:
                continue
            gxy = _sample_xy_at_t(gt, t, mode=gt_time_mode, max_time_gap_s=gt_max_gap_s)
            if gxy is None:
                continue
            gt_items.append((str(hid), np.asarray(gxy, dtype=float)))

        # Tracker detections at this frame
        det_items = _parse_simpletrack_all_detections(entry)  # list of (raw_id, xy_array)

        n_gt  = len(gt_items)
        n_det = len(det_items)

        relabelled = []
        if n_gt > 0 and n_det > 0:
            mocap_labels = [label for label, _ in gt_items]
            mocap_xy     = [gxy   for _, gxy   in gt_items]
            tracker_ids  = [str(did) if did is not None else "" for did, _ in det_items]
            tracker_xy   = [dxy for _, dxy in det_items]

            # Build cost matrix with switch penalty
            cost = np.zeros((n_gt, n_det), dtype=np.float64)
            for i, gxy in enumerate(mocap_xy):
                for j, dxy in enumerate(tracker_xy):
                    c = float(math.hypot(gxy[0] - dxy[0], gxy[1] - dxy[1]))
                    rid = tracker_ids[j]
                    prev_label = raw_id_to_label.get(rid)
                    if rid and prev_label and prev_label != mocap_labels[i]:
                        c += switch_penalty
                    cost[i, j] = c

            row_ind, col_ind = linear_sum_assignment(cost)

            tracker_to_label: dict = {}
            for r, c in zip(row_ind, col_ind):
                gxy = mocap_xy[r]
                dxy = tracker_xy[c]
                dist = float(math.hypot(gxy[0] - dxy[0], gxy[1] - dxy[1]))
                if dist <= match_thresh_m:
                    tracker_to_label[c] = mocap_labels[r]

            # Update ID memory from accepted matches
            for j, label in tracker_to_label.items():
                rid = tracker_ids[j]
                if rid:
                    raw_id_to_label[rid]   = label
                    raw_id_last_seen[rid]  = t

            # Build relabelled detection list (drop unmatched)
            for j, (_, dxy) in enumerate(det_items):
                if j in tracker_to_label:
                    label = tracker_to_label[j]
                    relabelled.append({'id': label, 'cx': float(dxy[0]), 'cy': float(dxy[1])})

        remapped.append({**entry, 'detections': relabelled})


    return np.asarray(remapped, dtype=object)


def _detection_metrics(
    simpletrack_arr,
    gt_series,
    match_thresh_m=0.5,
    include_all_detections=False,
    gt_time_mode="interp",
    gt_max_gap_s=None,
):
    assoc = _associate_detections_to_gt(
        simpletrack_arr,
        gt_series,
        match_thresh_m=match_thresh_m,
        include_all_detections=include_all_detections,
        gt_time_mode=gt_time_mode,
        gt_max_gap_s=gt_max_gap_s,
    )
    return {
        "overall": _summarize_errors(np.asarray(assoc["loc_err_tp"], dtype=float)),
        "id_1": _summarize_errors(np.asarray(assoc["loc_err_tp_by_id"][1], dtype=float)),
        "id_2": _summarize_errors(np.asarray(assoc["loc_err_tp_by_id"][2], dtype=float)),
    }


def _tracking_detection_common_metrics(
    simpletrack_arr,
    gt_series,
    match_thresh_m=0.5,
    include_all_detections=False,
    gt_time_mode="interp",
    gt_max_gap_s=None,
):
    assoc = _associate_detections_to_gt(
        simpletrack_arr,
        gt_series,
        match_thresh_m=match_thresh_m,
        include_all_detections=include_all_detections,
        gt_time_mode=gt_time_mode,
        gt_max_gap_s=gt_max_gap_s,
    )
    tp = int(assoc["tp"])
    fp = int(assoc["fp"])
    fn = int(assoc["fn"])
    gt_count = int(assoc["gt_count"])
    id_switches = int(assoc["id_switches"])
    loc_err_tp = np.asarray(assoc["loc_err_tp"], dtype=float)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / gt_count) if gt_count > 0 else float("nan")
    f1 = float(2 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else float("nan")
    mota = float(1.0 - (fn + fp + id_switches) / gt_count) if gt_count > 0 else float("nan")
    motp = float(np.mean(loc_err_tp)) if len(loc_err_tp) > 0 else float("nan")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_count": gt_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mota": mota,
        "motp_m": motp,
        "id_switches": id_switches,
    }


def _prediction_metrics(pred_arr, gt_series, pred_step_s=0.4, simpletrack_arr=None):
    gt1 = gt_series.get(1, None)
    gt2 = gt_series.get(2, None)
    if pred_arr is None or len(pred_arr) == 0 or gt1 is None or gt2 is None:
        empty = {
            "n": 0,
            "ade_mean_m": float("nan"),
            "ade_median_m": float("nan"),
            "ade_p95_m": float("nan"),
            "fde_mean_m": float("nan"),
            "fde_median_m": float("nan"),
            "fde_p95_m": float("nan"),
        }
        return {"overall": dict(empty), "id_1": dict(empty), "id_2": dict(empty)}

    ade_all = []
    fde_all = []
    ade_by_id = {1: [], 2: []}
    fde_by_id = {1: [], 2: []}

    pred_t0 = _prediction_anchor_times(pred_arr, simpletrack_arr=simpletrack_arr)
    for i, entry in enumerate(pred_arr):
        t0 = pred_t0[i] if i < len(pred_t0) else _entry_stamp(entry)
        if not np.isfinite(t0):
            continue
        pred = _prediction_tensor(entry)
        if pred is None or pred.ndim != 3 or pred.shape[2] != 2:
            continue
        T, A, _ = pred.shape
        if T < 1 or A < 2:
            continue

        # HST output in this stack is future-only after history cutoff:
        # pred[k] corresponds to t0 + (k+1) * dt.
        future_times = t0 + pred_step_s * np.arange(1, T + 1, dtype=float)
        g1 = _interp_xy(gt1, future_times)
        g2 = _interp_xy(gt2, future_times)
        if g1 is None or g2 is None or not (np.all(g1["mask"]) and np.all(g2["mask"])):
            continue

        pred_eval = pred[:, :, :]
        gt_xy_by_id = {1: g1["xy"], 2: g2["xy"]}

        # Permutation-invariant channel-to-ID assignment:
        # choose the two distinct channels minimizing total ADE across horizon.
        best = None
        for a1 in range(A):
            for a2 in range(A):
                if a2 == a1:
                    continue
                d1 = np.linalg.norm(pred_eval[:, a1, :] - gt_xy_by_id[1], axis=1)
                d2 = np.linalg.norm(pred_eval[:, a2, :] - gt_xy_by_id[2], axis=1)
                cost = float(np.mean(d1) + np.mean(d2))
                if best is None or cost < best[0]:
                    best = (cost, a1, a2)
        if best is None:
            continue
        _, ch_id1, ch_id2 = best
        mapping = {ch_id1: 1, ch_id2: 2}

        for a, hid in mapping.items():
            pred_xy = pred_eval[:, a, :]
            gt_xy = gt_xy_by_id[hid]
            d = np.linalg.norm(pred_xy - gt_xy, axis=1)
            ade = float(np.mean(d))
            fde = float(d[-1])
            ade_all.append(ade)
            fde_all.append(fde)
            ade_by_id[hid].append(ade)
            fde_by_id[hid].append(fde)

    def _pack(ade_list, fde_list):
        ade_s = _summarize_series(np.asarray(ade_list, dtype=float), "m")
        fde_s = _summarize_series(np.asarray(fde_list, dtype=float), "m")
        return {
            "n": int(min(ade_s["n"], fde_s["n"])),
            "ade_mean_m": ade_s["mean_m"],
            "ade_median_m": ade_s["median_m"],
            "ade_p95_m": ade_s["p95_m"],
            "fde_mean_m": fde_s["mean_m"],
            "fde_median_m": fde_s["median_m"],
            "fde_p95_m": fde_s["p95_m"],
        }

    return {
        "overall": _pack(ade_all, fde_all),
        "id_1": _pack(ade_by_id[1], fde_by_id[1]),
        "id_2": _pack(ade_by_id[2], fde_by_id[2]),
    }


def _tracking_continuity_metrics(simpletrack_arr):
    # Quantify per-ID tracking continuity and untracked time over run duration.
    if simpletrack_arr is None or len(simpletrack_arr) < 2:
        empty = {
            "run_duration_s": float("nan"),
            "present_time_s": 0.0,
            "untracked_time_s": float("nan"),
            "present_ratio": float("nan"),
            "untracked_ratio": float("nan"),
            "segment_count": 0,
            "avg_segment_duration_s": float("nan"),
            "median_segment_duration_s": float("nan"),
            "max_segment_duration_s": float("nan"),
            "min_segment_duration_s": float("nan"),
        }
        return {"id_1": dict(empty), "id_2": dict(empty)}

    frame_ts = []
    frame_present = {1: [], 2: []}
    for entry in simpletrack_arr:
        t = _entry_stamp(entry)
        if not np.isfinite(t):
            continue
        dets = _parse_simpletrack_frame(entry)
        frame_ts.append(float(t))
        frame_present[1].append(1 in dets)
        frame_present[2].append(2 in dets)

    if len(frame_ts) < 2:
        empty = {
            "run_duration_s": float("nan"),
            "present_time_s": 0.0,
            "untracked_time_s": float("nan"),
            "present_ratio": float("nan"),
            "untracked_ratio": float("nan"),
            "segment_count": 0,
            "avg_segment_duration_s": float("nan"),
            "median_segment_duration_s": float("nan"),
            "max_segment_duration_s": float("nan"),
            "min_segment_duration_s": float("nan"),
        }
        return {"id_1": dict(empty), "id_2": dict(empty)}

    frame_ts = np.asarray(frame_ts, dtype=float)
    order = np.argsort(frame_ts)
    ts = frame_ts[order]
    run_duration = float(ts[-1] - ts[0])
    if run_duration <= 0.0:
        run_duration = float("nan")

    # Use piecewise-constant intervals [t_i, t_{i+1}) with state at t_i.
    dt = np.diff(ts)
    # Guard against negative/invalid intervals due to timestamp irregularities.
    dt = np.where(np.isfinite(dt) & (dt > 0.0), dt, 0.0)

    out = {}
    for hid in (1, 2):
        present = np.asarray(frame_present[hid], dtype=bool)[order]
        present_head = present[:-1]

        present_time = float(np.sum(dt[present_head]))
        total_time = float(np.sum(dt))
        untracked_time = float(max(0.0, total_time - present_time))

        # Contiguous tracked segments over interval samples.
        seg_durations = []
        in_seg = False
        seg_t0 = 0.0
        for i, is_present in enumerate(present_head):
            if is_present and not in_seg:
                in_seg = True
                seg_t0 = ts[i]
            if in_seg and (not is_present):
                seg_durations.append(float(ts[i] - seg_t0))
                in_seg = False
        if in_seg:
            seg_durations.append(float(ts[-1] - seg_t0))

        seg_arr = np.asarray(seg_durations, dtype=float) if seg_durations else np.asarray([], dtype=float)

        out[f"id_{hid}"] = {
            "run_duration_s": total_time if total_time > 0.0 else float("nan"),
            "present_time_s": present_time,
            "untracked_time_s": untracked_time,
            "present_ratio": float(present_time / total_time) if total_time > 0.0 else float("nan"),
            "untracked_ratio": float(untracked_time / total_time) if total_time > 0.0 else float("nan"),
            "segment_count": int(len(seg_arr)),
            "avg_segment_duration_s": float(np.mean(seg_arr)) if len(seg_arr) > 0 else float("nan"),
            "median_segment_duration_s": float(np.median(seg_arr)) if len(seg_arr) > 0 else float("nan"),
            "max_segment_duration_s": float(np.max(seg_arr)) if len(seg_arr) > 0 else float("nan"),
            "min_segment_duration_s": float(np.min(seg_arr)) if len(seg_arr) > 0 else float("nan"),
        }
    return out


def _compute_run_metrics(
    run_dir: Path,
    det_match_thresh_m=0.5,
    pred_step_s=0.4,
    include_all_detections=False,
    gt_time_mode="interp",
    gt_max_gap_s=None,
):
    amcl_arr = _load_pickle(run_dir / "amcl_history.pkl")
    gt_arr = _load_pickle(run_dir / "ground_truth_xy_map.pkl")
    st_arr = _load_pickle(run_dir / "simpletrack_history.pkl")
    pred_arr = _load_pickle(run_dir / "prediction_history.pkl")

    gt_series = _build_gt_series(gt_arr)

    loc = _localization_metrics(amcl_arr, gt_series)
    det = _detection_metrics(
        st_arr,
        gt_series,
        match_thresh_m=det_match_thresh_m,
        include_all_detections=include_all_detections,
        gt_time_mode=gt_time_mode,
        gt_max_gap_s=gt_max_gap_s,
    )
    det_common = _tracking_detection_common_metrics(
        st_arr,
        gt_series,
        match_thresh_m=det_match_thresh_m,
        include_all_detections=include_all_detections,
        gt_time_mode=gt_time_mode,
        gt_max_gap_s=gt_max_gap_s,
    )
    track = _tracking_continuity_metrics(st_arr)
    pred = _prediction_metrics(pred_arr, gt_series, pred_step_s=pred_step_s, simpletrack_arr=st_arr)

    return {
        "run": run_dir.name,
        "localization": loc,
        "detection": det,
        "detection_common": det_common,
        "tracking_continuity": track,
        "prediction": pred,
    }


def run_metrics_mode(args):
    base_dir = Path(args.data_dir)
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        run_dirs = _run_dirs(base_dir, runs=args.runs)

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {base_dir}")

    all_metrics = []
    for run_dir in run_dirs:
        m = _compute_run_metrics(
            run_dir,
            det_match_thresh_m=args.det_match_thresh_m,
            pred_step_s=args.pred_step_s,
            include_all_detections=args.include_all_detections,
            gt_time_mode=args.gt_time_mode,
            gt_max_gap_s=args.gt_max_gap_s,
        )
        all_metrics.append(m)

        out_json = run_dir / "accuracy_metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

    print("Per-run metrics (meters):")
    for m in all_metrics:
        loc = m["localization"]
        det = m["detection"]
        detc = m["detection_common"]
        track = m["tracking_continuity"]
        pred = m["prediction"]
        print(f"\n{m['run']}")
        print(
            "  localization  "
            f"n={loc['n']} rmse={loc['rmse_m']:.4f} mae={loc['mae_m']:.4f} "
            f"p95={loc['p95_m']:.4f} max={loc['max_m']:.4f}"
        )
        print(
            "  detection_all "
            f"n={det['overall']['n']} rmse={det['overall']['rmse_m']:.4f} "
            f"mae={det['overall']['mae_m']:.4f} p95={det['overall']['p95_m']:.4f} "
            f"max={det['overall']['max_m']:.4f}"
        )
        print(
            "  detection_id1 "
            f"n={det['id_1']['n']} rmse={det['id_1']['rmse_m']:.4f} "
            f"mae={det['id_1']['mae_m']:.4f}"
        )
        print(
            "  detection_id2 "
            f"n={det['id_2']['n']} rmse={det['id_2']['rmse_m']:.4f} "
            f"mae={det['id_2']['mae_m']:.4f}"
        )
        print(
            "  det_common    "
            f"P={detc['precision']:.3f} R={detc['recall']:.3f} F1={detc['f1']:.3f} "
            f"MOTA={detc['mota']:.3f} MOTP={detc['motp_m']:.3f} "
            f"IDSW={detc['id_switches']}"
        )
        print(
            "  tracking_id1  "
            f"avg_seg={track['id_1']['avg_segment_duration_s']:.2f}s "
            f"untracked={track['id_1']['untracked_time_s']:.2f}s "
            f"untracked_ratio={track['id_1']['untracked_ratio']:.3f}"
        )
        print(
            "  tracking_id2  "
            f"avg_seg={track['id_2']['avg_segment_duration_s']:.2f}s "
            f"untracked={track['id_2']['untracked_time_s']:.2f}s "
            f"untracked_ratio={track['id_2']['untracked_ratio']:.3f}"
        )
        print(
            "  prediction    "
            f"ADE={pred['overall']['ade_mean_m']:.3f} "
            f"FDE={pred['overall']['fde_mean_m']:.3f} "
            f"(n={pred['overall']['n']})"
        )

    summary_path = (Path(args.out) if args.out else (base_dir / "metrics_summary.json"))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")


# ----------------------------
# Video mode
# ----------------------------

def run_video_mode(args):
    base_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(base_dir)

    amcl_arr = _load_pickle(run_dir / "amcl_history.pkl")
    gt_arr = _load_pickle(run_dir / "ground_truth_xy_map.pkl")
    simpletrack_arr = _load_pickle(run_dir / "simpletrack_history.pkl")
    pred_arr = _load_pickle(run_dir / "prediction_history.pkl")

    if simpletrack_arr is None or len(simpletrack_arr) == 0:
        raise RuntimeError("simpletrack_history.pkl is missing or empty; cannot build per-frame video.")

    gt_series = _build_gt_series(gt_arr)
    amcl_series = _build_amcl_series(amcl_arr)

    simpletrack_entries = list(simpletrack_arr)
    simpletrack_ts = np.asarray([_entry_stamp(e) for e in simpletrack_entries], dtype=float)
    valid = np.isfinite(simpletrack_ts)
    if not np.any(valid):
        raise RuntimeError("No valid timestamps found in simpletrack_history.pkl")

    simpletrack_ts = simpletrack_ts[valid]
    simpletrack_entries = [e for e, ok in zip(simpletrack_entries, valid) if ok]

    order = np.argsort(simpletrack_ts)
    simpletrack_ts = simpletrack_ts[order]
    simpletrack_entries = [simpletrack_entries[i] for i in order]

    pred_entries = list(pred_arr) if pred_arr is not None else []
    pred_ts = np.asarray([_entry_stamp(e) for e in pred_entries], dtype=float) if pred_entries else np.zeros((0,), dtype=float)
    if len(pred_ts) > 0:
        valid_pred = np.isfinite(pred_ts)
        pred_ts = pred_ts[valid_pred]
        pred_entries = [e for e, ok in zip(pred_entries, valid_pred) if ok]
        order_pred = np.argsort(pred_ts)
        pred_ts = pred_ts[order_pred]
        pred_entries = [pred_entries[i] for i in order_pred]

    min_x, max_x, min_y, max_y = _compute_bounds(simpletrack_entries, gt_series, amcl_series)

    out_path = Path(args.out) if args.out else (run_dir / "tracking_video.mp4")

    fig, ax = plt.subplots(figsize=(8, 8))
    writer = FFMpegWriter(fps=args.fps)

    with writer.saving(fig, str(out_path), dpi=150):
        n_frames = len(simpletrack_entries)
        for i, (t, st_entry) in enumerate(zip(simpletrack_ts, simpletrack_entries)):
            ax.clear()
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            robot_gt = _nearest_xy(gt_series.get(0, None), t)
            if robot_gt is not None:
                ax.scatter(robot_gt[0], robot_gt[1], s=70, c="tab:green", marker="s", label="robot_gt")
                ax.text(robot_gt[0], robot_gt[1], " robot", color="tab:green", fontsize=9)

            amcl = _nearest_amcl(amcl_series, t)
            if amcl is not None:
                amcl_xy, amcl_yaw = amcl
                ax.scatter(amcl_xy[0], amcl_xy[1], s=60, c="tab:blue", marker="x", label="amcl")
                arrow_len = 0.35
                ax.arrow(
                    amcl_xy[0],
                    amcl_xy[1],
                    arrow_len * np.cos(amcl_yaw),
                    arrow_len * np.sin(amcl_yaw),
                    head_width=0.12,
                    head_length=0.15,
                    fc="tab:blue",
                    ec="tab:blue",
                    alpha=0.85,
                    length_includes_head=True,
                )

            for hid, color in ((1, "tab:orange"), (2, "tab:red")):
                hxy = _nearest_xy(gt_series.get(hid, None), t)
                if hxy is None:
                    continue
                ax.scatter(hxy[0], hxy[1], s=65, c=color, marker="o", label=f"human_{hid}_gt")
                ax.text(hxy[0], hxy[1], f" gt:{hid}", color=color, fontsize=9)

            dets = _parse_simpletrack_frame(st_entry)
            for hid, color in ((1, "gold"), (2, "magenta")):
                if hid not in dets:
                    continue
                x, y = dets[hid]
                ax.scatter(x, y, s=90, c=color, marker="D", edgecolors="k", linewidths=0.8, label=f"simpletrack:{hid}")
                ax.text(x, y, f" trk:{hid}", color="black", fontsize=9)

            pred = _nearest_prediction(pred_entries, pred_ts, t)
            if pred is not None and pred.ndim == 3 and pred.shape[2] == 2:
                n_agents = min(2, pred.shape[1])
                pred_colors = ["tab:cyan", "tab:purple"]
                for a in range(n_agents):
                    traj = pred[:, a, :]
                    if len(traj) == 0:
                        continue
                    ax.plot(traj[:, 0], traj[:, 1], color=pred_colors[a], alpha=0.8, linewidth=2.0, label=f"pred_h{a+1}")
                    ax.scatter(traj[0, 0], traj[0, 1], color=pred_colors[a], s=22)

            ax.set_title(f"Run {run_dir.name} | frame {i+1}/{n_frames} | t={t:.3f}s")

            handles, labels = ax.get_legend_handles_labels()
            uniq = {}
            for h, l in zip(handles, labels):
                if l not in uniq:
                    uniq[l] = h
            if uniq:
                ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

            writer.grab_frame()

    print(f"Saved video to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualization and accuracy tools for ranger data runs.")
    parser.add_argument("--mode", choices=["video", "metrics"], default="video")
    parser.add_argument("--data-dir", default="data", help="Base data directory containing run_* folders")
    parser.add_argument("--run-dir", default=None, help="Specific run directory (overrides --data-dir)")
    parser.add_argument("--runs", nargs="*", default=None, help="Run folder names for metrics mode, e.g. run_2 run_3")
    parser.add_argument("--out", default=None, help="Video path (video mode) or summary JSON path (metrics mode)")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS (video mode)")
    parser.add_argument("--det-match-thresh-m", type=float, default=0.5, help="Match threshold for detection/tracking metrics")
    parser.add_argument(
        "--include-all-detections",
        action="store_true",
        help="In metrics mode, count all tracker detections for FP/assignment. Default evaluates target IDs only (1,2).",
    )
    parser.add_argument(
        "--gt-time-mode",
        choices=["interp", "nearest"],
        default="interp",
        help="How GT is sampled at detector timestamps for detection/tracking metrics.",
    )
    parser.add_argument(
        "--gt-max-gap-s",
        type=float,
        default=None,
        help="Optional max allowed |t_det - t_gt_nearest| (s). Frames beyond this are excluded from GT association.",
    )
    parser.add_argument("--pred-step-s", type=float, default=0.4, help="Prediction horizon step in seconds for ADE/FDE")
    args = parser.parse_args()

    if args.mode == "metrics":
        run_metrics_mode(args)
    else:
        run_video_mode(args)


if __name__ == "__main__":
    main()
