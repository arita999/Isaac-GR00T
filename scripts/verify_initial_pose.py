#!/usr/bin/env python3
"""GR00T v2.1 データセットから episode-start pose を集計する検証スクリプト。

eval_starai.py の TRAINING_INITIAL_POSE と突き合わせて、初期化位置が学習データ
分布から外れていないかを確認するために使う。

推奨値の決定ロジック:
    分布が bimodal (二峰性) の場合、単純な median は 2つの峰の "谷" に落ちて
    学習分布から外れた姿勢を指定してしまう。その場合はより密な峰の中心値を
    採用する。このスクリプトは Hartigan's dip-like な簡易判定(KDE ピーク検出)
    で bimodal を検知し、該当軸は密なモード中心を出力する。

使い方:
    python src/verify_initial_pose.py \
        /home/kazu/data/original_data/20260427_data_groot_v2

出力:
    各motor/gripperの median/mean/std/min/max と、eval_starai.py 内の現行値との差分。
    末尾に推奨値ベースの TRAINING_INITIAL_POSE dict を JSON 形式で表示する。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MOTOR_NAMES = [
    "Motor_0", "Motor_1", "Motor_2", "Motor_3", "Motor_4", "Motor_5", "gripper",
]

# Isaac-GR00T/gr00t/eval/real_robot/StarAI/eval_starai.py の現行値
EVAL_STARAI_INITIAL_POSE = {
    "Motor_0": -0.39,
    "Motor_1": -100.0,
    "Motor_2": 71.57,
    "Motor_3": 7.28,
    "Motor_4": 17.07,
    "Motor_5": -1.48,
    "gripper": 1.0,
}


def _read_info_version(dataset_path: Path) -> str | None:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        with open(info_path) as f:
            return json.load(f).get("codebase_version")
    except Exception:
        return None


def _collect_starts_v21(dataset_path: Path) -> np.ndarray:
    """v2.1 レイアウト: data/chunk-XXX/episode_YYYYYY.parquet → 各ファイルの先頭行"""
    parquet_files = sorted(dataset_path.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_files:
        return np.zeros((0, 0))
    rows = []
    for p in parquet_files:
        df = pd.read_parquet(p)
        if "observation.state" not in df.columns:
            raise ValueError(f"'observation.state' column missing in {p}")
        rows.append(np.asarray(df.iloc[0]["observation.state"], dtype=np.float64))
    return np.stack(rows, axis=0)


def _collect_starts_v30(dataset_path: Path) -> np.ndarray:
    """v3.0 レイアウト: data/chunk-XXX/file-YYY.parquet (複数エピソードが同ファイル内)"""
    parquet_files = sorted(dataset_path.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        return np.zeros((0, 0))

    first_rows: dict[int, pd.Series] = {}
    for p in parquet_files:
        df = pd.read_parquet(p)
        if "observation.state" not in df.columns:
            raise ValueError(f"'observation.state' column missing in {p}")
        for ep_idx, grp in df.groupby("episode_index"):
            ep_idx = int(ep_idx)
            cur = grp.sort_values("frame_index").iloc[0]
            prev = first_rows.get(ep_idx)
            if prev is None or cur["frame_index"] < prev["frame_index"]:
                first_rows[ep_idx] = cur

    ordered = [first_rows[i] for i in sorted(first_rows.keys())]
    return np.stack(
        [np.asarray(r["observation.state"], dtype=np.float64) for r in ordered]
    )


def collect_episode_start_states(dataset_path: Path) -> np.ndarray:
    """各エピソードの先頭フレーム observation.state を (n_episodes, 7) で返す。

    LeRobot v2.1 (episode_*.parquet) と v3.0 (file-*.parquet) のどちらにも対応。
    """
    version = _read_info_version(dataset_path)
    if version == "v3.0":
        arr = _collect_starts_v30(dataset_path)
    elif version == "v2.1":
        arr = _collect_starts_v21(dataset_path)
    else:
        arr = _collect_starts_v21(dataset_path)
        if arr.size == 0:
            arr = _collect_starts_v30(dataset_path)

    if arr.size == 0:
        raise FileNotFoundError(
            f"No episode parquet files under {dataset_path}/data/chunk-*/ "
            f"(tried both v2.1 episode_*.parquet and v3.0 file-*.parquet)"
        )
    if arr.shape[1] != len(MOTOR_NAMES):
        raise ValueError(
            f"Expected {len(MOTOR_NAMES)} state dims, got {arr.shape[1]}"
        )
    return arr


def _gaussian_kde_peaks(
    values: np.ndarray, n_grid: int = 512, bw_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Silverman bandwidth の Gaussian KDE を手計算し、ピーク(極大点)を返す。

    Returns:
        grid: KDE を評価したx軸 (n_grid,)
        density: KDE 値 (n_grid,)
        peak_idx: 極大点の grid index 配列 (密度の降順で整列)
    """
    n = len(values)
    if n < 2:
        grid = np.array([float(values.mean())])
        return grid, np.array([1.0]), np.array([0])

    std = float(np.std(values, ddof=1))
    if std <= 1e-9:
        grid = np.array([float(values.mean())])
        return grid, np.array([1.0]), np.array([0])

    # Silverman's rule of thumb
    bw = bw_scale * 1.06 * std * (n ** (-1 / 5))

    lo, hi = float(values.min()), float(values.max())
    pad = max(bw * 3, (hi - lo) * 0.05)
    grid = np.linspace(lo - pad, hi + pad, n_grid)

    # density[j] = (1/(n*bw*sqrt(2π))) * sum_i exp(-0.5*((grid[j]-x[i])/bw)^2)
    diff = (grid[:, None] - values[None, :]) / bw
    density = np.exp(-0.5 * diff * diff).sum(axis=1)
    density /= n * bw * np.sqrt(2 * np.pi)

    # 局所最大 (単純な 3-point 比較)
    peaks = (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])
    peak_idx_raw = np.where(peaks)[0] + 1

    # 密度の降順で並べる
    peak_idx = peak_idx_raw[np.argsort(-density[peak_idx_raw])]
    return grid, density, peak_idx


def recommend_initial_value(
    values: np.ndarray,
    min_relative_height: float = 0.3,
    min_separation_std_mult: float = 0.5,
) -> tuple[float, str]:
    """episode-start 分布から推奨初期姿勢値を決定する。

    - 単峰(または2番目のピークが十分小さい/近い) → median を採用
    - 多峰で明確に分離している → 最も密なモード中心を採用

    Args:
        values: 各エピソードの初期値 (n,)
        min_relative_height: 第2ピークが第1ピークの何倍以上あれば bimodal とみなすか
        min_separation_std_mult: 第1ピーク±(std*これ) 外に第2ピークがあれば bimodal とみなす

    Returns:
        推奨値, モード説明文字列
    """
    if len(values) == 0:
        return 0.0, "empty"

    med = float(np.median(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    grid, density, peak_idx = _gaussian_kde_peaks(values)
    if len(peak_idx) <= 1 or std < 1e-6:
        return med, f"unimodal (median={med:.3f})"

    # Top 2 peaks
    p1_x = float(grid[peak_idx[0]])
    p1_d = float(density[peak_idx[0]])
    p2_x = float(grid[peak_idx[1]])
    p2_d = float(density[peak_idx[1]])

    rel_height = p2_d / p1_d if p1_d > 0 else 0.0
    separation = abs(p2_x - p1_x)
    is_bimodal = (
        rel_height >= min_relative_height
        and separation >= min_separation_std_mult * std
    )

    if not is_bimodal:
        return med, f"near-unimodal (median={med:.3f}, top-peak={p1_x:.3f})"

    # bimodal: median がモード間の谷にあるか
    # 単純に median が [min(p1,p2), max(p1,p2)] の valley に入るかを判定
    lo_x, hi_x = sorted([p1_x, p2_x])
    if lo_x < med < hi_x:
        return p1_x, (
            f"BIMODAL, median in valley "
            f"(median={med:.3f}, peaks=[{p1_x:.3f} d={p1_d:.3f}, "
            f"{p2_x:.3f} d={p2_d:.3f}]; use denser peak)"
        )

    return med, (
        f"bimodal but median outside valley "
        f"(median={med:.3f}, peaks=[{p1_x:.3f}, {p2_x:.3f}])"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare TRAINING_INITIAL_POSE in eval_starai.py against dataset",
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="GR00T v2.1 dataset root (contains data/chunk-*/episode_*.parquet)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    arr = collect_episode_start_states(dataset_path)

    print(f"Dataset:  {dataset_path}")
    print(f"Episodes: {arr.shape[0]}\n")

    header = (
        f"{'motor':<10} {'median':>9} {'mean':>9} {'std':>9} "
        f"{'min':>9} {'max':>9} {'eval.py':>9} {'Δ(rec-eval)':>13}  mode"
    )
    print(header)
    print("-" * len(header))

    recommended: dict[str, float] = {}
    for i, name in enumerate(MOTOR_NAMES):
        col = arr[:, i]
        med = float(np.median(col))
        mean = float(np.mean(col))
        std = float(np.std(col))
        mn = float(np.min(col))
        mx = float(np.max(col))
        ref = EVAL_STARAI_INITIAL_POSE[name]
        rec, mode = recommend_initial_value(col)
        delta = rec - ref
        recommended[name] = round(rec, 2)
        print(
            f"{name:<10} {med:>9.3f} {mean:>9.3f} {std:>9.3f} "
            f"{mn:>9.3f} {mx:>9.3f} {ref:>9.3f} {delta:>+13.3f}  {mode}"
        )

    print("\n--- Recommended TRAINING_INITIAL_POSE ---")
    print("(Copy into eval_starai.py if Δ is large)")
    print(json.dumps(recommended, indent=4, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
