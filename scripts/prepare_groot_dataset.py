#!/usr/bin/env python3
"""
LeRobot v3.0データセットをGR00T LeRobot v2.1形式に変換するスクリプト

GR00T (NVIDIA Isaac GR00T) で使用するために、LeRobot v3.0形式のデータセットを
LeRobot v2.1形式に変換し、meta/modality.json を自動生成する。

変換内容:
  - data/: 統合parquetをエピソード別parquetに分割
  - videos/: ディレクトリ構造をv2.1レイアウトに変更
  - meta/tasks.parquet → meta/tasks.jsonl
  - meta/episodes/ → meta/episodes.jsonl
  - meta/info.json: v3.0 → v2.1 スキーマ更新
  - meta/modality.json: 自動生成 (GR00T固有)
  - meta/stats.json: コピー

使い方:
    # 単一データセットの変換
    python src/prepare_groot_dataset.py /path/to/lerobot_v3_dataset

    # 出力先を指定
    python src/prepare_groot_dataset.py /path/to/dataset --output /path/to/output

    # タスク名を上書き
    python src/prepare_groot_dataset.py /path/to/dataset --task-name "untangle cable"

    # 一括変換 (ディレクトリ内の全データセット)
    python src/prepare_groot_dataset.py /path/to/datasets_dir --batch

参照: Isaac-GR00T/getting_started/data_preparation.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# v2.1 path templates
# ---------------------------------------------------------------------------
V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _to_serializable(value: Any) -> Any:
    """numpy/pyarrow の値を標準Python型に変換"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


def _ffmpeg_env() -> dict[str, str]:
    """Return an environment that lets system ffmpeg use its system libraries.

    Some local shells set LD_LIBRARY_PATH to custom ffmpeg/CUDA library
    directories. On this machine that makes /usr/bin/ffmpeg fail while loading
    libopenmpt/libmpg123, so video extraction must run with library overrides
    removed.
    """
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    return env


def _run_ffmpeg(args: list[str]) -> None:
    ffmpeg = shutil.which(os.environ.get("FFMPEG_BINARY", "ffmpeg"))
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found. Install ffmpeg or set FFMPEG_BINARY.")

    subprocess.run(
        [ffmpeg, *args],
        check=True,
        timeout=300,
        env=_ffmpeg_env(),
    )


def _extract_task_description(task_label: Any) -> str:
    """タスクラベルから説明文字列を抽出する

    lerobot v3の tasks.parquet のインデックスは dict や文字列の場合がある:
      - "pick and place"
      - {'cring_home': 'pick and place'}
      - "True"
    """
    if isinstance(task_label, dict):
        values = list(task_label.values())
        return str(values[0]) if values else str(task_label)
    s = str(task_label)
    # dict文字列をパース: "{'key': 'value'}"
    if s.startswith("{") and ":" in s:
        try:
            d = eval(s)
            if isinstance(d, dict):
                values = list(d.values())
                return str(values[0]) if values else s
        except Exception:
            pass
    return s


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_info(dataset_path: Path) -> dict:
    with open(dataset_path / "meta" / "info.json") as f:
        return json.load(f)


def load_episode_records(dataset_path: Path) -> list[dict]:
    """meta/episodes/chunk-*/file-*.parquet からエピソードメタデータを読み込む"""
    episodes_dir = dataset_path / "meta" / "episodes"
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}")
    records = []
    for pq_path in pq_paths:
        table = pq.read_table(pq_path)
        records.extend(table.to_pylist())
    records.sort(key=lambda r: int(r["episode_index"]))
    return records


def load_tasks_from_parquet(dataset_path: Path) -> list[dict[str, Any]]:
    """meta/tasks.parquet からタスクリストを読み込む"""
    import pandas as pd

    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return [{"task_index": 0, "task": "task"}]
    df = pd.read_parquet(tasks_path)
    tasks = []
    for task_label, row in df.iterrows():
        tasks.append({
            "task_index": int(row["task_index"]),
            "task": _extract_task_description(task_label),
        })
    return sorted(tasks, key=lambda t: t["task_index"])


# ---------------------------------------------------------------------------
# Modality auto-detection
# ---------------------------------------------------------------------------
def detect_modality_groups(
    feature_names: list[str],
    total_dim: int,
) -> dict[str, dict[str, int]]:
    """特徴量名からstate/actionのモダリティグループを自動検出

    Returns: {"group_name": {"start": int, "end": int}, ...}
    """
    if not feature_names:
        return {"arm": {"start": 0, "end": total_dim}}

    def _index_range(indices: list[int]) -> dict[str, int]:
        """連続した index リストから {"start": min, "end": max+1} を作る"""
        return {"start": min(indices), "end": max(indices) + 1}

    has_left = any(n.startswith("left_") for n in feature_names)
    has_right = any(n.startswith("right_") for n in feature_names)

    if has_left and has_right:
        # Bi-arm robot: split by left/right, then arm/gripper
        left_arm = [i for i, n in enumerate(feature_names)
                     if n.startswith("left_") and "gripper" not in n]
        left_gripper = [i for i, n in enumerate(feature_names)
                         if n.startswith("left_") and "gripper" in n]
        right_arm = [i for i, n in enumerate(feature_names)
                      if n.startswith("right_") and "gripper" not in n]
        right_gripper = [i for i, n in enumerate(feature_names)
                          if n.startswith("right_") and "gripper" in n]

        groups: dict[str, dict[str, int]] = {}
        if left_arm:
            groups["left_arm"] = _index_range(left_arm)
        if left_gripper:
            groups["left_gripper"] = _index_range(left_gripper)
        if right_arm:
            groups["right_arm"] = _index_range(right_arm)
        if right_gripper:
            groups["right_gripper"] = _index_range(right_gripper)
        return groups if groups else {"arm": {"start": 0, "end": total_dim}}

    # Single arm: split arm/gripper
    arm_indices = [i for i, n in enumerate(feature_names) if "gripper" not in n]
    gripper_indices = [i for i, n in enumerate(feature_names) if "gripper" in n]

    groups = {}
    if arm_indices:
        groups["single_arm"] = _index_range(arm_indices)
    if gripper_indices:
        groups["gripper"] = _index_range(gripper_indices)
    return groups if groups else {"arm": {"start": 0, "end": total_dim}}


def generate_modality_json(info: dict) -> dict:
    """info.json からGR00T用 modality.json を生成"""
    features = info["features"]

    # State / Action grouping
    state_feat = features.get("observation.state", {})
    state_names = state_feat.get("names") or []
    state_dim = state_feat.get("shape", [0])[0]
    action_feat = features.get("action", {})
    action_names = action_feat.get("names") or []
    action_dim = action_feat.get("shape", [0])[0]

    state_groups = detect_modality_groups(state_names, state_dim)
    action_groups = detect_modality_groups(action_names, action_dim)

    # Video modality
    video_map: dict[str, dict[str, str]] = {}
    for key, feat in features.items():
        if feat.get("dtype") == "video":
            short_name = key.replace("observation.images.", "")
            video_map[short_name] = {"original_key": key}

    # Annotation
    annotation: dict[str, dict] = {
        "human.action.task_description": {"original_key": "task_index"},
    }

    return {
        "state": state_groups,
        "action": action_groups,
        "video": video_map,
        "annotation": annotation,
    }


# ---------------------------------------------------------------------------
# v3.0 → v2.1 structure conversion
# ---------------------------------------------------------------------------
def convert_data(
    src_root: Path,
    dst_root: Path,
    episode_records: list[dict],
    chunks_size: int,
) -> None:
    """統合parquetファイルをエピソード別parquetに分割"""
    # Group episodes by source data file
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for rec in episode_records:
        key = (int(rec["data/chunk_index"]), int(rec["data/file_index"]))
        grouped[key].append(rec)

    total_episodes = 0
    for (chunk_idx, file_idx), records in sorted(grouped.items()):
        src = src_root / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"Data file not found: {src}")

        table = pq.read_table(src)
        records = sorted(records, key=lambda r: int(r["dataset_from_index"]))
        file_offset = int(records[0]["dataset_from_index"])

        for rec in records:
            ep_idx = int(rec["episode_index"])
            start = int(rec["dataset_from_index"]) - file_offset
            stop = int(rec["dataset_to_index"]) - file_offset
            length = stop - start

            if length <= 0:
                raise ValueError(
                    f"Invalid episode length: ep={ep_idx}, start={start}, stop={stop}"
                )

            ep_table = table.slice(start, length)
            dest_chunk = ep_idx // chunks_size
            dest = dst_root / V21_DATA_PATH.format(
                episode_chunk=dest_chunk, episode_index=ep_idx,
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(ep_table, dest)
            total_episodes += 1

    print(f"  Data: {total_episodes} episode parquet files written")


def convert_videos(
    src_root: Path,
    dst_root: Path,
    episode_records: list[dict],
    video_keys: list[str],
    chunks_size: int,
) -> None:
    """ビデオファイルをv2.1レイアウトにコピー/分割"""
    for video_key in video_keys:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        ts_from_col = f"videos/{video_key}/from_timestamp"
        ts_to_col = f"videos/{video_key}/to_timestamp"

        # Group episodes by source video file
        grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for rec in episode_records:
            c_idx = rec.get(chunk_col)
            f_idx = rec.get(file_col)
            if c_idx is None or f_idx is None:
                continue
            grouped[(int(c_idx), int(f_idx))].append(rec)

        copied = 0
        extracted = 0
        for (c_idx, f_idx), records in sorted(grouped.items()):
            src = src_root / f"videos/{video_key}/chunk-{c_idx:03d}/file-{f_idx:03d}.mp4"
            if not src.exists():
                print(f"  Warning: Video not found: {src}")
                continue

            if len(records) == 1:
                # 1エピソード=1ファイル → そのままコピー
                rec = records[0]
                ep_idx = int(rec["episode_index"])
                dest_chunk = ep_idx // chunks_size
                dest = dst_root / V21_VIDEO_PATH.format(
                    episode_chunk=dest_chunk,
                    video_key=video_key,
                    episode_index=ep_idx,
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                copied += 1
            else:
                # 複数エピソード=1ファイル → ffmpegでセグメント抽出
                for rec in sorted(records, key=lambda r: float(r.get(ts_from_col, 0))):
                    ep_idx = int(rec["episode_index"])
                    start_ts = float(rec.get(ts_from_col, 0))
                    end_ts = float(rec.get(ts_to_col, 0))
                    duration = max(end_ts - start_ts, 1e-6)

                    dest_chunk = ep_idx // chunks_size
                    dest = dst_root / V21_VIDEO_PATH.format(
                        episode_chunk=dest_chunk,
                        video_key=video_key,
                        episode_index=ep_idx,
                    )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    _run_ffmpeg(
                        [
                            "-hide_banner", "-loglevel", "error",
                            "-ss", f"{start_ts:.6f}",
                            "-i", str(src),
                            "-t", f"{duration:.6f}",
                            "-c", "copy",
                            "-avoid_negative_ts", "1",
                            "-y", str(dest),
                        ],
                    )
                    extracted += 1

        short_key = video_key.replace("observation.images.", "")
        print(f"  Video [{short_key}]: {copied} copied, {extracted} extracted")


# ---------------------------------------------------------------------------
# Write v2.1 metadata
# ---------------------------------------------------------------------------
def write_tasks_jsonl(dst_root: Path, tasks: list[dict]) -> None:
    path = dst_root / "meta" / "tasks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"  tasks.jsonl: {len(tasks)} tasks")


def write_episodes_jsonl(dst_root: Path, episode_records: list[dict]) -> None:
    path = dst_root / "meta" / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in sorted(episode_records, key=lambda r: int(r["episode_index"])):
            tasks_val = rec.get("tasks", [])
            if isinstance(tasks_val, str):
                tasks_val = [tasks_val]
            entry = {
                "episode_index": int(rec["episode_index"]),
                "tasks": _to_serializable(tasks_val),
                "length": int(rec.get("length", 0)),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  episodes.jsonl: {len(episode_records)} episodes")


def write_info_v21(
    dst_root: Path,
    info: dict,
    episode_records: list[dict],
    video_keys: list[str],
) -> None:
    total_episodes = info.get("total_episodes", len(episode_records))
    chunks_size = info.get("chunks_size", 1000)

    v2_info = dict(info)
    v2_info["codebase_version"] = "v2.1"
    v2_info["data_path"] = V21_DATA_PATH
    v2_info["video_path"] = V21_VIDEO_PATH if video_keys else None
    v2_info.pop("data_files_size_in_mb", None)
    v2_info.pop("video_files_size_in_mb", None)
    v2_info["total_chunks"] = (
        math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    )
    v2_info["total_videos"] = total_episodes * len(video_keys)

    path = dst_root / "meta" / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(v2_info, f, indent=4)
    print("  info.json: v2.1")


def _compute_column_stats(np_data: np.ndarray) -> dict:
    return {
        "min": np.min(np_data, axis=0).tolist(),
        "max": np.max(np_data, axis=0).tolist(),
        "mean": np.mean(np_data, axis=0).tolist(),
        "std": np.std(np_data, axis=0).tolist(),
        "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
        "q10": np.quantile(np_data, 0.10, axis=0).tolist(),
        "q50": np.quantile(np_data, 0.50, axis=0).tolist(),
        "q90": np.quantile(np_data, 0.90, axis=0).tolist(),
        "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        "count": [int(np_data.shape[0])],
    }


def compute_stats(src_root: Path, dst_root: Path, info: dict) -> None:
    """v2.1 parquet全フレームから GR00T 互換の stats.json を再計算する。

    v3 の meta/stats.json は per-episode 集計のため、q01/q99/std 等がフレーム分布と
    ずれている(GR00T の normalization で誤動作する)。ここでは GR00T の
    gr00t/data/stats.py:calculate_dataset_statistics と同じ方法で、全フレームを
    連結してから min/max/mean/std/q01/q10/q50/q90/q99/count を計算する。
    """
    import pandas as pd

    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted((dst_root / "data").rglob("episode_*.parquet"))
    if not parquet_files:
        print("  stats.json: no parquet files found in output, skipping")
        return

    features = info.get("features", {})
    frames = [pd.read_parquet(p) for p in parquet_files]
    all_frames = pd.concat(frames, axis=0, ignore_index=True)

    stats: dict[str, dict] = {}
    for feature_name, feature_info in features.items():
        dtype = feature_info.get("dtype", "")
        if dtype == "video":
            continue
        if feature_name not in all_frames.columns:
            continue
        col = all_frames[feature_name]
        if col.dtype == object:
            np_data = np.vstack([np.asarray(v, dtype=np.float64) for v in col])
        else:
            np_data = np.asarray(col.to_numpy(), dtype=np.float64).reshape(-1, 1)
        stats[feature_name] = _compute_column_stats(np_data)

    # video stats は正規化に使われないが、v3 の既存 stats.json にある値を引き継ぐ
    v3_stats_src = src_root / "meta" / "stats.json"
    if v3_stats_src.exists():
        with open(v3_stats_src) as f:
            v3_stats = json.load(f)
        for feature_name, feature_info in features.items():
            if feature_info.get("dtype") != "video":
                continue
            if feature_name in v3_stats:
                stats[feature_name] = v3_stats[feature_name]

    with open(dst_meta / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    print(f"  stats.json: recomputed from {len(parquet_files)} episodes, "
          f"{len(all_frames)} frames")

    relative_stats_src = src_root / "meta" / "relative_stats.json"
    if relative_stats_src.exists():
        shutil.copy2(relative_stats_src, dst_meta / "relative_stats.json")
        print("  relative_stats.json: copied")
    else:
        print(
            "  relative_stats.json: not found in source "
            "(required later if your GR00T action config uses RELATIVE actions)"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dataset(dst_root: Path) -> bool:
    """出力データセットの検証"""
    required_files = [
        "meta/info.json",
        "meta/modality.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
    ]
    ok = True
    for rel in required_files:
        if not (dst_root / rel).exists():
            print(f"  MISSING: {rel}")
            ok = False

    data_files = sorted((dst_root / "data").rglob("episode_*.parquet"))
    if not data_files:
        print("  MISSING: No data parquet files found")
        ok = False
    else:
        print(f"  Data: {len(data_files)} episode parquet files")

    video_files = sorted((dst_root / "videos").rglob("episode_*.mp4")) if (dst_root / "videos").exists() else []
    print(f"  Videos: {len(video_files)} files")

    # Validate modality.json references against info.json
    modality_path = dst_root / "meta" / "modality.json"
    info_path = dst_root / "meta" / "info.json"
    if modality_path.exists() and info_path.exists():
        with open(modality_path) as f:
            modality = json.load(f)
        with open(info_path) as f:
            info = json.load(f)
        features = info.get("features", {})

        # Check video keys reference valid features
        for short_name, vconf in modality.get("video", {}).items():
            orig_key = vconf.get("original_key", "")
            if orig_key and orig_key not in features:
                print(f"  WARNING: modality.json video '{short_name}' references "
                      f"unknown feature '{orig_key}'")

        # Check state/action dimensions
        for section in ("state", "action"):
            groups = modality.get(section, {})
            feat = features.get("observation.state" if section == "state" else "action", {})
            dim = feat.get("shape", [0])[0]
            for name, indices in groups.items():
                if indices.get("end", 0) > dim:
                    print(f"  WARNING: modality.json {section}.{name} end={indices['end']} "
                          f"exceeds dimension {dim}")

    if (dst_root / "meta" / "stats.json").exists():
        print("  stats.json: present")
    else:
        print("  stats.json: MISSING (run gr00t/data/stats.py to generate)")

    return ok


# ---------------------------------------------------------------------------
# Main conversion flow
# ---------------------------------------------------------------------------
def convert_single_dataset(
    input_dir: Path,
    output_dir: Path,
    task_name_override: str | None = None,
) -> int:
    """単一データセットを変換"""
    info = load_info(input_dir)
    version = info.get("codebase_version", "unknown")
    robot_type = info.get("robot_type", "unknown")
    total_episodes = info.get("total_episodes", "?")
    total_frames = info.get("total_frames", "?")

    print(f"Dataset:  {input_dir.name}")
    print(f"Version:  {version}")
    print(f"Robot:    {robot_type}")
    print(f"Episodes: {total_episodes}, Frames: {total_frames}")
    print(f"Output:   {output_dir}")

    if version not in ("v3.0",):
        print(f"Error: This script converts v3.0 datasets. Got version: {version}")
        print("  For v2.x datasets, manually add meta/modality.json.")
        return 1

    video_keys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    chunks_size = info.get("chunks_size", 1000)

    # --- Load v3 metadata ---
    episode_records = load_episode_records(input_dir)
    tasks = load_tasks_from_parquet(input_dir)
    if task_name_override:
        tasks = [{"task_index": t["task_index"], "task": task_name_override} for t in tasks]

    # --- Convert structure v3 → v2.1 ---
    print("\n--- Converting data ---")
    convert_data(input_dir, output_dir, episode_records, chunks_size)

    if video_keys:
        print("\n--- Converting videos ---")
        convert_videos(input_dir, output_dir, episode_records, video_keys, chunks_size)

    print("\n--- Writing metadata ---")
    write_tasks_jsonl(output_dir, tasks)
    write_episodes_jsonl(output_dir, episode_records)
    write_info_v21(output_dir, info, episode_records, video_keys)
    compute_stats(input_dir, output_dir, info)

    # --- Generate modality.json ---
    modality = generate_modality_json(info)
    modality_path = output_dir / "meta" / "modality.json"
    with open(modality_path, "w", encoding="utf-8") as f:
        json.dump(modality, f, indent=4)
    print("  modality.json: generated")
    print(f"    state:  {list(modality['state'].keys())}")
    print(f"    action: {list(modality['action'].keys())}")
    print(f"    video:  {list(modality['video'].keys())}")

    # --- Validate ---
    print("\n--- Validation ---")
    ok = validate_dataset(output_dir)

    if ok:
        print(f"\n=== Done: {output_dir} ===")
    else:
        print("\n=== Validation warnings (see above) ===")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LeRobot v3.0データセットをGR00T LeRobot v2.1形式に変換",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="入力データセット (またはデータセット群の親ディレクトリ)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力ディレクトリ (default: {input}_groot_v2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="--batch時の出力親ディレクトリ",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="タスク名を上書き (例: 'untangle cable')",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="input_dir 内の全データセットを一括変換",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="出力ディレクトリが存在する場合、上書きする",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()

    if not input_dir.exists():
        print(f"Error: Input not found: {input_dir}")
        return 1

    if args.batch:
        # 一括変換モード
        subdirs = sorted(
            d for d in input_dir.iterdir()
            if d.is_dir() and (d / "meta" / "info.json").exists()
        )
        if not subdirs:
            print(f"Error: No datasets found in {input_dir}")
            return 1

        print(f"Found {len(subdirs)} datasets in {input_dir}\n")
        output_parent = args.output_dir or input_dir.parent / f"{input_dir.name}_groot_v2"
        output_parent.mkdir(parents=True, exist_ok=True)

        failed = []
        for sub in subdirs:
            out = output_parent / sub.name
            if out.exists():
                if args.force:
                    shutil.rmtree(out)
                else:
                    print(f"SKIP: {sub.name} (output exists, use --force)")
                    continue

            out.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*60}")
            try:
                rc = convert_single_dataset(sub, out, args.task_name)
                if rc != 0:
                    failed.append(sub.name)
            except Exception as exc:
                print(f"Error: {exc}")
                failed.append(sub.name)

        print(f"\n{'='*60}")
        print(f"Batch complete: {len(subdirs) - len(failed)}/{len(subdirs)} succeeded")
        if failed:
            print(f"Failed: {', '.join(failed)}")
            return 1
        return 0

    # 単一データセット変換
    if not (input_dir / "meta" / "info.json").exists():
        print(f"Error: Not a dataset (meta/info.json not found): {input_dir}")
        return 1

    output_dir = args.output or input_dir.parent / f"{input_dir.name}_groot_v2"
    if output_dir.exists():
        if args.force:
            shutil.rmtree(output_dir)
        else:
            print(f"Error: Output already exists: {output_dir}")
            print("  Use --force to overwrite")
            return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    return convert_single_dataset(input_dir, output_dir, args.task_name)


if __name__ == "__main__":
    raise SystemExit(main())
