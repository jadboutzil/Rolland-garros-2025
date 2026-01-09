import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Utilities
# -----------------------------

def _to_int_dict(ball_data_json):
    return {int(k): v for k, v in ball_data_json.items()}

def _bounded_interp(arr, max_gap=5):
    return pd.Series(arr.astype(float)).interpolate(limit=max_gap, limit_direction="both").to_numpy()

def _rolling_median(arr, w=5):
    return pd.Series(arr).rolling(w, center=True, min_periods=1).median().to_numpy()

def _compute_v(frames, series, fps):
    v = np.full_like(series, np.nan, dtype=float)
    dt = np.diff(frames.astype(float))
    ds = np.diff(series)
    valid = np.isfinite(ds) & (dt > 0)
    v[1:][valid] = (ds[valid] / dt[valid]) * fps
    return v

def _compute_a(frames, v):
    a = np.full_like(v, np.nan, dtype=float)
    dt = np.diff(frames.astype(float))
    dv = np.diff(v)
    valid = np.isfinite(dv) & (dt > 0)
    a[1:][valid] = dv[valid] / dt[valid]
    return a

def _consolidate_frames(frames, score_by_frame, merge_gap=10):
    """Group frames within merge_gap and keep the highest-score frame per group."""
    if not frames:
        return []
    frames = sorted(set(int(f) for f in frames))
    groups = []
    cur = [frames[0]]
    for f in frames[1:]:
        if f - cur[-1] <= merge_gap:
            cur.append(f)
        else:
            groups.append(cur)
            cur = [f]
    groups.append(cur)
    reps = [max(g, key=lambda ff: score_by_frame.get(ff, -1.0)) for g in groups]
    return sorted(set(reps))

def _remove_near(frames, anchors, exclusion=12):
    anchors = sorted(set(int(a) for a in anchors))
    out = []
    for f in frames:
        if all(abs(f - a) > exclusion for a in anchors):
            out.append(f)
    return sorted(set(out))

def _pick_one_hit_between_bounces(hits, bounces, score_by_frame=None):
    """
    Keep at most one hit in each interval between consecutive bounces.
    If score_by_frame is provided, pick the strongest; else pick earliest.
    """
    hits = sorted(set(int(h) for h in hits))
    bounces = sorted(set(int(b) for b in bounces))
    if len(bounces) < 2:
        return hits

    picked = []
    bounds = [-10**18] + bounces + [10**18]
    for L, R in zip(bounds[:-1], bounds[1:]):
        h_in = [h for h in hits if L < h < R]
        if not h_in:
            continue
        if score_by_frame:
            picked.append(max(h_in, key=lambda h: score_by_frame.get(h, -1.0)))
        else:
            picked.append(h_in[0])
    return sorted(set(picked))

# -----------------------------
# Segmentation
# -----------------------------

def split_into_segments_from_existing_frames(ball_data_int, max_invisible_run=137, min_segment_len=10):
    """
    Split based on consecutive frames present in JSON:
    - long runs of visible=False trigger a cut
    Returns list of segments: dict with keys {frames, start, end}
    """
    frames = sorted(ball_data_int.keys())
    if not frames:
        return []

    segments = []
    seg_frames = []
    inv_run = 0

    for f in frames:
        seg_frames.append(f)
        vis = bool(ball_data_int[f].get("visible", False))
        inv_run = inv_run + 1 if not vis else 0

        if inv_run >= max_invisible_run:
            # cut before the invisible run started
            cut_end_idx = len(seg_frames) - inv_run
            keep = seg_frames[:cut_end_idx]
            if len(keep) >= min_segment_len:
                segments.append({"frames": keep, "start": keep[0], "end": keep[-1]})
            # restart after the run (drop it)
            seg_frames = []
            inv_run = 0

    if len(seg_frames) >= min_segment_len:
        segments.append({"frames": seg_frames, "start": seg_frames[0], "end": seg_frames[-1]})

    return segments

# -----------------------------
# Feature extraction per segment
# -----------------------------

def build_features_segment(ball_data_int, seg_frames, fps=50, interp_max_gap=5, smooth_w=5):
    frames = np.array(seg_frames, dtype=int)

    x = np.array([
        ball_data_int[f].get("x", np.nan) if ball_data_int[f].get("visible", False) else np.nan
        for f in frames
    ], dtype=float)
    y = np.array([
        ball_data_int[f].get("y", np.nan) if ball_data_int[f].get("visible", False) else np.nan
        for f in frames
    ], dtype=float)

    # bounded interpolation
    x_i = _bounded_interp(x, interp_max_gap)
    y_i = _bounded_interp(y, interp_max_gap)

    # velocities / accelerations
    vx = _compute_v(frames, x_i, fps)
    vy = _compute_v(frames, y_i, fps)
    ax = _compute_a(frames, vx)
    ay = _compute_a(frames, vy)

    # smoothing
    vx_s = _rolling_median(vx, smooth_w)
    vy_s = _rolling_median(vy, smooth_w)
    ax_s = _rolling_median(ax, smooth_w)
    ay_s = _rolling_median(ay, smooth_w)

    speed = np.sqrt(vx_s**2 + vy_s**2)

    dv = np.full_like(vy_s, np.nan, dtype=float)
    dv[1:] = np.abs(np.diff(vy_s))

    # direction change flags
    dir_x = np.zeros_like(vx_s, dtype=float)
    dir_y = np.zeros_like(vy_s, dtype=float)
    okx = np.isfinite(vx_s[:-1]) & np.isfinite(vx_s[1:])
    oky = np.isfinite(vy_s[:-1]) & np.isfinite(vy_s[1:])
    dir_x[1:][okx] = (vx_s[:-1][okx] * vx_s[1:][okx] < 0).astype(float)
    dir_y[1:][oky] = (vy_s[:-1][oky] * vy_s[1:][oky] < 0).astype(float)

    # y normalized within segment
    yv = y_i[np.isfinite(y_i)]
    if len(yv) > 0:
        y_min, y_max = float(np.min(yv)), float(np.max(yv))
        y_norm = (y_i - y_min) / (y_max - y_min + 1e-9)
    else:
        y_norm = np.full_like(y_i, np.nan, dtype=float)

    # Feature matrix
    F = np.column_stack([vx_s, vy_s, ax_s, ay_s, speed, dv, dir_x, dir_y, y_norm])

    # rows valid
    valid_rows = np.isfinite(F).all(axis=1)
    return frames, F, valid_rows

# -----------------------------
# KMeans labeling (per point, pooled across segments)
# -----------------------------

def kmeans_predict_actions(ball_data_int, segments, fps=50,
                          interp_max_gap=5, smooth_w=5,
                          gate_quantile=0.85,
                          n_clusters=3):
    """
    Returns pred_action_by_frame: dict[int -> 'air'|'bounce'|'hit']
    """
    # default: air
    pred = {f: "air" for f in ball_data_int.keys()}

    all_frames = []
    all_feats = []

    # keep also raw feature slices to map cluster stats
    for seg in segments:
        frames, F, valid_rows = build_features_segment(
            ball_data_int, seg["frames"], fps=fps, interp_max_gap=interp_max_gap, smooth_w=smooth_w
        )
        frames_v = frames[valid_rows]
        F_v = F[valid_rows]

        if len(frames_v) < 10:
            continue

        # gating: keep event-like frames
        ay = F_v[:, 3]
        dv = F_v[:, 5]
        dirx = F_v[:, 6]
        diry = F_v[:, 7]

        ay_thr = np.quantile(np.abs(ay), gate_quantile)
        dv_thr = np.quantile(dv, gate_quantile)

        keep = (np.abs(ay) >= ay_thr) | (dv >= dv_thr) | (dirx > 0) | (diry > 0)
        frames_k = frames_v[keep]
        F_k = F_v[keep]

        if len(frames_k) == 0:
            continue

        all_frames.append(frames_k)
        all_feats.append(F_k)

    if not all_frames:
        return pred

    frames_all = np.concatenate(all_frames)
    feats_all = np.vstack(all_feats)

    # scale + kmeans
    X = StandardScaler().fit_transform(feats_all)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(X)

    # cluster mapping to actions (heuristics)
    y_norm = feats_all[:, 8]
    speed = feats_all[:, 4]
    ax_abs = np.abs(feats_all[:, 2])
    ay_abs = np.abs(feats_all[:, 3])
    acc = np.maximum(ax_abs, ay_abs)

    stats = {}
    for c in range(n_clusters):
        m = labels == c
        stats[c] = {
            "y_norm": float(np.median(y_norm[m])) if np.any(m) else -np.inf,
            "speed": float(np.median(speed[m])) if np.any(m) else -np.inf,
            "acc": float(np.median(acc[m])) if np.any(m) else -np.inf,
        }

    # bounce cluster: highest y_norm (near ground)
    bounce_c = max(stats, key=lambda c: stats[c]["y_norm"])

    # hit cluster: among remaining, highest speed + 0.5*acc
    remaining = [c for c in range(n_clusters) if c != bounce_c]
    hit_c = max(remaining, key=lambda c: stats[c]["speed"] + 0.5 * stats[c]["acc"])
    air_c = [c for c in range(n_clusters) if c not in (bounce_c, hit_c)][0]

    # assign predictions for gated frames
    for f, lab in zip(frames_all, labels):
        if lab == bounce_c:
            pred[int(f)] = "bounce"
        elif lab == hit_c:
            pred[int(f)] = "hit"
        else:
            pred[int(f)] = "air"

    return pred

# -----------------------------
# Post-processing to reduce flicker + apply tennis structure
# -----------------------------

def postprocess_pred_actions(ball_data_int, pred_action,
                             merge_gap_bounce=10, merge_gap_hit=10,
                             bounce_exclusion_for_hits=12):
    """
    Returns refined pred_action dict.
    """
    # build simple scores for consolidation: use y for bounce, speed for hit (approx)
    frames_sorted = sorted(ball_data_int.keys())
    # y score for bounce (higher y => more ground-like)
    y_score = {f: float(ball_data_int[f].get("y", -1e9)) if ball_data_int[f].get("visible", False) else -1e9
               for f in frames_sorted}

    # speed score for hit computed quickly from raw coords (coarse; enough for consolidation)
    # (If you want, you can reuse the full feature speed maps, but keep it simple here.)
    xs = np.array([ball_data_int[f].get("x", np.nan) if ball_data_int[f].get("visible", False) else np.nan for f in frames_sorted], float)
    ys = np.array([ball_data_int[f].get("y", np.nan) if ball_data_int[f].get("visible", False) else np.nan for f in frames_sorted], float)
    xs = _bounded_interp(xs, 5)
    ys = _bounded_interp(ys, 5)
    fr = np.array(frames_sorted, int)
    vx = _compute_v(fr, xs, fps=50)
    vy = _compute_v(fr, ys, fps=50)
    sp = np.sqrt(vx**2 + vy**2)
    sp = _rolling_median(sp, 5)
    speed_score = {int(f): float(sp[i]) if np.isfinite(sp[i]) else -1.0 for i, f in enumerate(fr)}

    # collect raw predicted frames
    pred_bounce_frames = [f for f in frames_sorted if pred_action.get(f) == "bounce"]
    pred_hit_frames = [f for f in frames_sorted if pred_action.get(f) == "hit"]

    # consolidate flicker
    bounce_events = _consolidate_frames(pred_bounce_frames, y_score, merge_gap=merge_gap_bounce)
    hit_events = _consolidate_frames(pred_hit_frames, speed_score, merge_gap=merge_gap_hit)

    # remove hits near bounces
    hit_events = _remove_near(hit_events, bounce_events, exclusion=bounce_exclusion_for_hits)

    # tennis structure: one hit between two bounces (pick strongest by speed_score)
    hit_events = _pick_one_hit_between_bounces(hit_events, bounce_events, score_by_frame=speed_score)

    # rebuild frame-level pred_action:
    refined = {f: "air" for f in frames_sorted}
    for b in bounce_events:
        refined[b] = "bounce"
    for h in hit_events:
        if refined.get(h) != "bounce":
            refined[h] = "hit"

    return refined

# -----------------------------
# Main function (unsupervised)
# -----------------------------

def unsupervised_hit_bounce_detection_kmeans(
    ball_data_json,
    fps=50,
    max_invisible_run=137,
    min_segment_len=10,
    interp_max_gap=5,
    smooth_w=5,
    gate_quantile=0.85,
    n_clusters=3,
    merge_gap_bounce=10,
    merge_gap_hit=10,
    bounce_exclusion_for_hits=12
):
    """
    Input: JSON dict with frame keys as strings.
    Output: same structure, plus "pred_action" for each frame.
    """
    ball_data_int = _to_int_dict(ball_data_json)

    # segments
    segments = split_into_segments_from_existing_frames(
        ball_data_int, max_invisible_run=max_invisible_run, min_segment_len=min_segment_len
    )

    # KMeans predictions (frame-level for gated frames, default air for others)
    pred_action = kmeans_predict_actions(
        ball_data_int, segments,
        fps=fps, interp_max_gap=interp_max_gap, smooth_w=smooth_w,
        gate_quantile=gate_quantile, n_clusters=n_clusters
    )

    # Post-process to events + tennis structure, then write back to frame-level labels (mostly air)
    refined_pred = postprocess_pred_actions(
        ball_data_int, pred_action,
        merge_gap_bounce=merge_gap_bounce,
        merge_gap_hit=merge_gap_hit,
        bounce_exclusion_for_hits=bounce_exclusion_for_hits
    )

    # Build enriched JSON (keys as strings like original)
    enriched = {str(f): dict(info) for f, info in ball_data_int.items()}
    for f in enriched:
        enriched[f]["pred_action"] = refined_pred.get(int(f), "air")

    return enriched

import json

with open(f"{folder_path_back}/ball_data_92.json", "r") as f:
    ball_data_json = json.load(f)

enriched = unsupervised_hit_bounce_detection_kmeans(ball_data_json)

with open("ball_data_92_enriched_kmeans.json", "w") as f:
    json.dump(enriched, f, indent=2)



# -----------------------------
# supervised function
# -----------------------------

import os, json, random
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ----------------------------
# Config
# ----------------------------
LABEL2ID = {"air": 0, "bounce": 1, "hit": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ----------------------------
# Segmentation + features
# ----------------------------

def split_into_segments_from_existing_frames(ball_data_int, max_invisible_run=137, min_segment_len=30):
    frames = sorted(ball_data_int.keys())
    if not frames:
        return []
    segments = []
    seg = []
    inv_run = 0
    for f in frames:
        seg.append(f)
        vis = bool(ball_data_int[f].get("visible", False))
        inv_run = inv_run + 1 if not vis else 0
        if inv_run >= max_invisible_run:
            cut_end_idx = len(seg) - inv_run
            keep = seg[:cut_end_idx]
            if len(keep) >= min_segment_len:
                segments.append(keep)
            seg = []
            inv_run = 0
    if len(seg) >= min_segment_len:
        segments.append(seg)
    return segments

def bounded_interp(arr: np.ndarray, max_gap=5) -> np.ndarray:
    return pd.Series(arr.astype(float)).interpolate(limit=max_gap, limit_direction="both").to_numpy()

def rolling_median(arr: np.ndarray, w=5) -> np.ndarray:
    return pd.Series(arr).rolling(w, center=True, min_periods=1).median().to_numpy()

def compute_v(frames: np.ndarray, series: np.ndarray, fps: float) -> np.ndarray:
    v = np.full_like(series, np.nan, dtype=float)
    dt = np.diff(frames.astype(float))
    ds = np.diff(series)
    valid = np.isfinite(ds) & (dt > 0)
    v[1:][valid] = (ds[valid] / dt[valid]) * fps
    return v

def compute_a(frames: np.ndarray, v: np.ndarray) -> np.ndarray:
    a = np.full_like(v, np.nan, dtype=float)
    dt = np.diff(frames.astype(float))
    dv = np.diff(v)
    valid = np.isfinite(dv) & (dt > 0)
    a[1:][valid] = dv[valid] / dt[valid]
    return a

def build_features_for_segment(ball_data_int, seg_frames,
                               fps=50, interp_max_gap=5, smooth_w=5,
                               drop_invisible=True):
    frames = np.array(seg_frames, dtype=int)

    visible = np.array([bool(ball_data_int[f].get("visible", False)) for f in frames], dtype=bool)
    x = np.array([ball_data_int[f].get("x", np.nan) for f in frames], dtype=float)
    y = np.array([ball_data_int[f].get("y", np.nan) for f in frames], dtype=float)
    actions = [ball_data_int[f].get("action", "air") for f in frames]

    if drop_invisible:
        x[~visible] = np.nan
        y[~visible] = np.nan

    x_i = bounded_interp(x, interp_max_gap)
    y_i = bounded_interp(y, interp_max_gap)

    vx = compute_v(frames, x_i, fps)
    vy = compute_v(frames, y_i, fps)
    ax = compute_a(frames, vx)
    ay = compute_a(frames, vy)

    vx = rolling_median(vx, smooth_w)
    vy = rolling_median(vy, smooth_w)
    ax = rolling_median(ax, smooth_w)
    ay = rolling_median(ay, smooth_w)

    speed = np.sqrt(vx**2 + vy**2)

    dv = np.full_like(vy, np.nan, dtype=float)
    dv[1:] = np.abs(np.diff(vy))

    dir_x = np.zeros_like(vx, dtype=float)
    dir_y = np.zeros_like(vy, dtype=float)
    okx = np.isfinite(vx[:-1]) & np.isfinite(vx[1:])
    oky = np.isfinite(vy[:-1]) & np.isfinite(vy[1:])
    dir_x[1:][okx] = (vx[:-1][okx] * vx[1:][okx] < 0).astype(float)
    dir_y[1:][oky] = (vy[:-1][oky] * vy[1:][oky] < 0).astype(float)

    # y_norm per segment
    yv = y_i[np.isfinite(y_i)]
    if len(yv) > 0:
        y_min, y_max = float(np.min(yv)), float(np.max(yv))
        y_norm = (y_i - y_min) / (y_max - y_min + 1e-9)
    else:
        y_norm = np.full_like(y_i, np.nan, dtype=float)

    X = np.column_stack([
        x_i, y_i,
        vx, vy,
        ax, ay,
        speed,
        dv,
        dir_x, dir_y,
        y_norm,
        visible.astype(float),
    ])

    y_lab = np.array([LABEL2ID.get(a, 0) for a in actions], dtype=int)

    # keep only finite rows
    valid_rows = np.isfinite(X).all(axis=1)
    X = X[valid_rows]
    y_lab = y_lab[valid_rows]

    return X, y_lab

def load_sequences_from_files(folder, files,
                              fps=50, max_invisible_run=137,
                              interp_max_gap=5, smooth_w=5,
                              min_segment_len=30):
    seqs = []
    for fn in files:
        path = os.path.join(folder, fn)
        with open(path, "r") as f:
            raw = json.load(f)
        ball_data_int = {int(k): v for k, v in raw.items()}
        segments = split_into_segments_from_existing_frames(
            ball_data_int, max_invisible_run=max_invisible_run, min_segment_len=min_segment_len
        )
        for seg_frames in segments:
            X, y = build_features_for_segment(
                ball_data_int, seg_frames, fps=fps, interp_max_gap=interp_max_gap, smooth_w=smooth_w
            )
            if len(y) >= (min_segment_len + 2):
                seqs.append((X, y))
    return seqs

# ----------------------------
# Dataset (Past → Next)
# ----------------------------

class PastToNextDataset(Dataset):
    def __init__(self, sequences, win_size=20, stride=1, scaler=None, fit_scaler=False):
        """
        Input: X[t-win+1 : t+1]  (length=win_size)
        Target: y[t+1]
        """
        self.win = win_size
        self.samples = []

        # optional scaling (fit only on train)
        if scaler is None:
            scaler = StandardScaler()
        self.scaler = scaler

        # flatten train features to fit scaler if requested
        if fit_scaler:
            allX = np.vstack([X for X, _ in sequences if len(X) > 0])
            self.scaler.fit(allX)

        for X, y in sequences:
            if len(y) <= win_size:
                continue
            Xs = self.scaler.transform(X)

            T = len(y)
            for t in range(win_size - 1, T - 1, stride):
                Xw = Xs[t - win_size + 1: t + 1]
                target = y[t + 1]
                self.samples.append((Xw.astype(np.float32), int(target)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        Xw, yt = self.samples[idx]
        return torch.tensor(Xw), torch.tensor(yt, dtype=torch.long)

# ----------------------------
# Model
# ----------------------------

class BiLSTMNextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)       # [B, T, 2H]
        last = out[:, -1, :]        # last timestep represents "now"
        return self.head(last)

# ----------------------------
# Training / Eval
# ----------------------------

def make_weighted_sampler(dataset, num_classes=3):
    # weight each sample by inverse class frequency
    ys = np.array([dataset.samples[i][1] for i in range(len(dataset))], dtype=int)
    counts = np.bincount(ys, minlength=num_classes)
    class_w = 1.0 / np.maximum(counts, 1)
    sample_w = class_w[ys]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True
    )
    return sampler, counts

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()) * len(y)
        n += len(y)
    return total / max(n, 1)

@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        p = torch.argmax(logits, dim=1).cpu().numpy()
        ps.append(p)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)

# ----------------------------
# End-to-end runner
# ----------------------------

def run_training(folder_path_back,
                 val_ratio=0.2,
                 seed=0,
                 fps=50,
                 max_invisible_run=137,
                 interp_max_gap=5,
                 smooth_w=5,
                 min_segment_len=30,
                 win_size=20,
                 stride=1,
                 batch_size=256,
                 epochs=10,
                 lr=1e-3,
                 hidden_dim=64):
    # list files
    files = sorted([f for f in os.listdir(folder_path_back) if f.endswith(".json")])
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_ratio)
    val_files = files[:n_val]
    train_files = files[n_val:]

    print(f"Files: train={len(train_files)}, val={len(val_files)}")
    with open("train_files.txt", "w") as f:
       f.write("\n".join(train_files))

    with open("val_files.txt", "w") as f:
       f.write("\n".join(val_files))


    # load sequences
    train_seqs = load_sequences_from_files(folder_path_back, train_files,
                                           fps=fps, max_invisible_run=max_invisible_run,
                                           interp_max_gap=interp_max_gap, smooth_w=smooth_w,
                                           min_segment_len=min_segment_len)
    val_seqs = load_sequences_from_files(folder_path_back, val_files,
                                         fps=fps, max_invisible_run=max_invisible_run,
                                         interp_max_gap=interp_max_gap, smooth_w=smooth_w,
                                         min_segment_len=min_segment_len)

    print(f"Sequences: train={len(train_seqs)}, val={len(val_seqs)}")

    # datasets + scaler fit on train
    scaler = StandardScaler()
    train_ds = PastToNextDataset(train_seqs, win_size=win_size, stride=stride, scaler=scaler, fit_scaler=True)
    val_ds = PastToNextDataset(val_seqs, win_size=win_size, stride=stride, scaler=scaler, fit_scaler=False)

    print(f"Windows: train={len(train_ds)}, val={len(val_ds)}")

    sampler, counts = make_weighted_sampler(train_ds, num_classes=3)
    print("Train class counts (by window target):", counts)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = train_ds.samples[0][0].shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiLSTMNextClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    # weighted CE (still useful even with sampler)
    class_counts = np.maximum(counts, 1)
    class_weights = (class_counts.sum() / class_counts).astype(np.float32)
    class_weights = class_weights / class_weights.sum() * 3
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        y_true, y_pred = predict_all(model, val_loader, device)

        print(f"\nEpoch {ep}/{epochs}  loss={loss:.4f}")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=["air", "bounce", "hit"], digits=3))

    return model, scaler


folder_path_back = "Data hit & bounce/per_point_v2"
model, scaler = run_training(folder_path_back, epochs=10, win_size=20, batch_size=256)




def predict_frame_labels_past_to_next(
    ball_data_json,
    model,
    scaler,
    win_size=20,
    fps=50,
    max_invisible_run=137,
    min_segment_len=30,
    interp_max_gap=5,
    smooth_w=5,
    device=None,
):
    """
    Returns: pred_label_by_frame: dict[int -> "air"/"bounce"/"hit"]
    Only frames that receive a prediction are included.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    ball_data_int = {int(k): v for k, v in ball_data_json.items()}

    # default (we'll fill only predicted frames)
    pred_label_by_frame = {}

    segments = split_into_segments_from_existing_frames(
        ball_data_int, max_invisible_run=max_invisible_run, min_segment_len=min_segment_len
    )

    for seg_frames in segments:
        # Build features for this segment
        # IMPORTANT: This drops invalid rows, so we must also compute frames_valid.
        frames = np.array(seg_frames, dtype=int)

        visible = np.array([bool(ball_data_int[f].get("visible", False)) for f in frames], dtype=bool)
        x = np.array([ball_data_int[f].get("x", np.nan) for f in frames], dtype=float)
        y = np.array([ball_data_int[f].get("y", np.nan) for f in frames], dtype=float)

        x[~visible] = np.nan
        y[~visible] = np.nan

        x_i = bounded_interp(x, interp_max_gap)
        y_i = bounded_interp(y, interp_max_gap)

        vx = rolling_median(compute_v(frames, x_i, fps), smooth_w)
        vy = rolling_median(compute_v(frames, y_i, fps), smooth_w)
        ax = rolling_median(compute_a(frames, vx), smooth_w)
        ay = rolling_median(compute_a(frames, vy), smooth_w)

        speed = np.sqrt(vx**2 + vy**2)

        dv = np.full_like(vy, np.nan, dtype=float)
        dv[1:] = np.abs(np.diff(vy))

        dir_x = np.zeros_like(vx, dtype=float)
        dir_y = np.zeros_like(vy, dtype=float)
        okx = np.isfinite(vx[:-1]) & np.isfinite(vx[1:])
        oky = np.isfinite(vy[:-1]) & np.isfinite(vy[1:])
        dir_x[1:][okx] = (vx[:-1][okx] * vx[1:][okx] < 0).astype(float)
        dir_y[1:][oky] = (vy[:-1][oky] * vy[1:][oky] < 0).astype(float)

        yv = y_i[np.isfinite(y_i)]
        if len(yv) > 0:
            y_min, y_max = float(np.min(yv)), float(np.max(yv))
            y_norm = (y_i - y_min) / (y_max - y_min + 1e-9)
        else:
            y_norm = np.full_like(y_i, np.nan, dtype=float)

        F = np.column_stack([
            x_i, y_i,
            vx, vy,
            ax, ay,
            speed,
            dv,
            dir_x, dir_y,
            y_norm,
            visible.astype(float),
        ])

        valid_rows = np.isfinite(F).all(axis=1)
        X_valid = F[valid_rows]
        frames_valid = frames[valid_rows]

        if len(frames_valid) <= win_size:
            continue

        # scale with train scaler
        Xs = scaler.transform(X_valid)

        # build windows
        T = len(frames_valid)
        windows = []
        target_frame_nums = []

        for t in range(win_size - 1, T - 1):
            Xw = Xs[t - win_size + 1: t + 1]  # length win
            windows.append(Xw.astype(np.float32))
            target_frame_nums.append(int(frames_valid[t + 1]))

        if not windows:
            continue

        X_tensor = torch.tensor(np.stack(windows)).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            pred_ids = torch.argmax(logits, dim=1).cpu().numpy()

        for fnum, pid in zip(target_frame_nums, pred_ids):
            pred_label_by_frame[fnum] = ID2LABEL[int(pid)]

    return pred_label_by_frame


def remove_near(frames, anchors, exclusion=12):
    anchors = sorted(set(int(a) for a in anchors))
    out = []
    for f in frames:
        if all(abs(f - a) > exclusion for a in anchors):
            out.append(f)
    return sorted(set(out))

def pick_one_hit_between_bounces(hit_events, bounce_events, score_map=None):
    hit_events = sorted(set(int(h) for h in hit_events))
    bounce_events = sorted(set(int(b) for b in bounce_events))
    if len(bounce_events) < 2:
        return hit_events

    picked = []
    bounds = [-10**18] + bounce_events + [10**18]
    for L, R in zip(bounds[:-1], bounds[1:]):
        hits_in = [h for h in hit_events if L < h < R]
        if not hits_in:
            continue
        if score_map:
            picked.append(max(hits_in, key=lambda h: score_map.get(h, -1.0)))
        else:
            picked.append(hits_in[0])
    return sorted(set(picked))



def build_point_motion_scores(ball_data_int, fps=50, interp_max_gap=5, smooth_w=5):
    frames = np.array(sorted(ball_data_int.keys()), dtype=int)

    visible = np.array([bool(ball_data_int[f].get("visible", False)) for f in frames], dtype=bool)
    x = np.array([ball_data_int[f].get("x", np.nan) for f in frames], dtype=float)
    y = np.array([ball_data_int[f].get("y", np.nan) for f in frames], dtype=float)

    # drop invisibles -> interpolate
    x[~visible] = np.nan
    y[~visible] = np.nan
    x = pd.Series(x).interpolate(limit=interp_max_gap, limit_direction="both").to_numpy()
    y = pd.Series(y).interpolate(limit=interp_max_gap, limit_direction="both").to_numpy()

    def vel(series):
        v = np.full_like(series, np.nan, dtype=float)
        dt = np.diff(frames.astype(float))
        ds = np.diff(series)
        valid = np.isfinite(ds) & (dt > 0)
        v[1:][valid] = (ds[valid] / dt[valid]) * fps
        return v

    vx = vel(x)
    vy = vel(y)

    # smoothing
    vx = pd.Series(vx).rolling(smooth_w, center=True, min_periods=1).median().to_numpy()
    vy = pd.Series(vy).rolling(smooth_w, center=True, min_periods=1).median().to_numpy()

    speed = np.sqrt(vx**2 + vy**2)

    # impulse magnitude |Δv|
    dvx = np.full_like(vx, np.nan); dvx[1:] = np.diff(vx)
    dvy = np.full_like(vy, np.nan); dvy[1:] = np.diff(vy)
    impulse = np.sqrt(dvx**2 + dvy**2)

    speed_map = {int(f): float(speed[i]) for i, f in enumerate(frames) if np.isfinite(speed[i])}
    impulse_map = {int(f): float(impulse[i]) for i, f in enumerate(frames) if np.isfinite(impulse[i])}
    y_map = {int(f): float(y[i]) for i, f in enumerate(frames) if np.isfinite(y[i])}

    return speed_map, impulse_map, y_map

def runs_from_frame_labels(frames_sorted, label_by_frame, target_label):
    """
    Return list of runs: each run is list of frames where label==target_label consecutively.
    """
    runs = []
    cur = []

    for f in frames_sorted:
        if label_by_frame.get(f) == target_label:
            if not cur or f == cur[-1] + 1:
                cur.append(f)
            else:
                runs.append(cur)
                cur = [f]
        else:
            if cur:
                runs.append(cur)
                cur = []
    if cur:
        runs.append(cur)

    return runs

def pick_event_frame(run_frames, score_map):
    return max(run_frames, key=lambda f: score_map.get(f, -1e9))


def consolidate_lstm_predictions_to_events(
    ball_data_int,
    pred_label_by_frame,
    fps=50,
    interp_max_gap=5,
    smooth_w=5,
    min_run_len_hit=2,
    min_run_len_bounce=2,
    hit_near_bounce_exclusion=12,
):
    frames_sorted = sorted(ball_data_int.keys())
    speed_map, impulse_map, y_map = build_point_motion_scores(
        ball_data_int, fps=fps, interp_max_gap=interp_max_gap, smooth_w=smooth_w
    )

    bounce_runs = runs_from_frame_labels(frames_sorted, pred_label_by_frame, "bounce")
    bounce_runs = [r for r in bounce_runs if len(r) >= min_run_len_bounce]
    bounce_events = [pick_event_frame(r, y_map) for r in bounce_runs]
    bounce_events = sorted(set(bounce_events))

    hit_runs = runs_from_frame_labels(frames_sorted, pred_label_by_frame, "hit")
    hit_runs = [r for r in hit_runs if len(r) >= min_run_len_hit]
    hit_events = [pick_event_frame(r, impulse_map) for r in hit_runs]
    hit_events = sorted(set(hit_events))

    hit_events = remove_near(hit_events, bounce_events, exclusion=hit_near_bounce_exclusion)
    hit_events = pick_one_hit_between_bounces(hit_events, bounce_events, score_map=impulse_map)

    return hit_events, bounce_events

def make_enriched_json_with_events(ball_data_json, hit_events, bounce_events):
    ball_data_int = {int(k): v for k, v in ball_data_json.items()}
    enriched = {str(f): dict(info) for f, info in ball_data_int.items()}

    for k in enriched:
        enriched[k]["pred_action"] = "air"

    for b in bounce_events:
        if str(b) in enriched:
            enriched[str(b)]["pred_action"] = "bounce"

    for h in hit_events:
        if str(h) in enriched and enriched[str(h)]["pred_action"] != "bounce":
            enriched[str(h)]["pred_action"] = "hit"

    return enriched

def supervised_hit_bounce_detection_lstm(
    ball_data_json,
    model,
    scaler,
    win_size=20,
    fps=50,
    max_invisible_run=137,
    min_segment_len=30,
    interp_max_gap=5,
    smooth_w=5,
):
    pred_label_by_frame = predict_frame_labels_past_to_next(
        ball_data_json,
        model=model,
        scaler=scaler,
        win_size=win_size,
        fps=fps,
        max_invisible_run=max_invisible_run,
        min_segment_len=min_segment_len,
        interp_max_gap=interp_max_gap,
        smooth_w=smooth_w,
    )

    ball_data_int = {int(k): v for k, v in ball_data_json.items()}
    hit_events, bounce_events = consolidate_lstm_predictions_to_events(
        ball_data_int,
        pred_label_by_frame,
        fps=fps,
        interp_max_gap=interp_max_gap,
        smooth_w=smooth_w,
    )

    enriched = make_enriched_json_with_events(ball_data_json, hit_events, bounce_events)
    return enriched, pred_label_by_frame, hit_events, bounce_events

def match_events_with_tolerance(gt_frames, pred_frames, tol=3):
    gt = sorted(gt_frames)
    pred = sorted(pred_frames)

    matched_gt = set()
    matched_pred = set()

    j = 0
    for p in pred:
        # advance j to first gt that could match
        while j < len(gt) and gt[j] < p - tol:
            j += 1

        # try to match p with the closest gt within [p-tol, p+tol]
        best = None
        best_dist = None
        k = j
        while k < len(gt) and gt[k] <= p + tol:
            if gt[k] not in matched_gt:
                dist = abs(gt[k] - p)
                if best is None or dist < best_dist:
                    best = gt[k]
                    best_dist = dist
            k += 1

        if best is not None:
            matched_gt.add(best)
            matched_pred.add(p)

    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(gt) - len(matched_gt)
    return tp, fp, fn, matched_gt, matched_pred


import random

with open("val_files.txt") as f:
    val_files = [l.strip() for l in f.readlines()]

test_file = random.choice(val_files)
print("Evaluating on:", test_file)

with open(f"{folder_path_back}/{test_file}") as f:
    ball_data_json = json.load(f)

enriched, pred_frame_labels, pred_hits, pred_bounces = supervised_hit_bounce_detection_lstm(
    ball_data_json, model, scaler, win_size=20
)

print("Pred hits:", pred_hits)
print("Pred bounces:", pred_bounces)


ball_data_int = {int(k): v for k, v in ball_data_json.items()}

gt_hits = sorted([f for f, info in ball_data_int.items() if info.get("action")=="hit"])
gt_bounces = sorted([f for f, info in ball_data_int.items() if info.get("action")=="bounce"])

for tol in [3,5,10,25 , 50]:
    tp, fp, fn, _, _ = match_events_with_tolerance(gt_hits, pred_hits, tol=tol)
    print(f"HIT tol=±{tol}: TP={tp}, FP={fp}, FN={fn}")

for tol in [3,5,10,25 , 50]:
    tp, fp, fn, _, _ = match_events_with_tolerance(gt_bounces, pred_bounces, tol=tol)
    print(f"BOUNCE tol=±{tol}: TP={tp}, FP={fp}, FN={fn}")



# import pickle

# def save_lstm_model(
#     model,
#     scaler,
#     path_prefix="hit_bounce_lstm",
#     model_config=None,
# ):
#     """
#     Saves:
#     - model weights
#     - scaler
#     - model config
#     """
#     if model_config is None:
#         raise ValueError("You must provide model_config to reload the model later.")

#     # 1. Save model weights
#     torch.save(
#         {
#             "model_state_dict": model.state_dict(),
#             "model_config": model_config,
#         },
#         f"{path_prefix}_model.pt"
#     )

#     # 2. Save scaler
#     with open(f"{path_prefix}_scaler.pkl", "wb") as f:
#         pickle.dump(scaler, f)

#     print(f"✅ Model saved to {path_prefix}_model.pt")
#     print(f"✅ Scaler saved to {path_prefix}_scaler.pkl")

# model_config = {
#     "input_dim": model.lstm.input_size,
#     "hidden_dim": model.lstm.hidden_size,
#     "num_layers": model.lstm.num_layers,
#     "dropout": 0.2,
#     "num_classes": 3,
#     "bidirectional": True,
# }

# save_lstm_model(
#     model=model,
#     scaler=scaler,
#     path_prefix="hit_bounce_lstm_past_to_next",
#     model_config=model_config,

# )

 

def load_lstm_model(path_prefix="hit_bounce_lstm_past_to_next", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(f"{path_prefix}_model.pt", map_location=device)
    cfg = checkpoint["model_config"]

    # Rebuild model
    model = BiLSTMNextClassifier(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load scaler
    with open(f"{path_prefix}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print("✅ Model and scaler loaded successfully")
    return model, scaler
