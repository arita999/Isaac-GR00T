#!/usr/bin/env python3
"""
StarAI Cello Real-Robot GR00T Policy Evaluation Script

This script runs closed-loop policy evaluation on the StarAI Cello robot
using the GR00T Policy API via ZMQ Server-Client architecture.

Supports asynchronous inference to eliminate stop-and-go behavior:
the robot keeps executing actions while the next inference runs
in a background thread.

Based on the SO100 reference implementation (eval_so100.py) and
GR00T real_world_deployment.md recommendations.

Usage:
  Terminal 1 (GR00T server):
    cd ~/Isaac-GR00T && source .venv/bin/activate
    python gr00t/eval/run_gr00t_server.py \
        --embodiment-tag NEW_EMBODIMENT \
        --model-path /home/kazu/data/GR00T/outputs/run_20260428_1839/checkpoint-20000 \
        --device cuda:0 --host 0.0.0.0 --port 5555

  Terminal 2 (this script):
    python src/groot_starai_predict_runtime.py \
        --robot-port /dev/ttyUSB1 \
        --robot-id my_awesome_staraicello_arm4 \
        --side-camera /dev/video4 \
        --rear-camera /dev/video6 \
        --onhand-camera /dev/video10 \
        --task "pick_and_place" \
        --expected-model-path /media/tmc/data/data/GR00T/output/run_20260428_1839/checkpoint-20000 \
        --training-dataset-path /media/tmc/data/data/original_data/20260427_data \
        --execute-steps 8 \
        --duration-sec 60
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
# Isaac-GR00T は ~/Isaac-GR00T へ移設済み(旧: ~/yaskawa/Isaac-GR00T)。
# gr00t は .venv の editable install 経由で import 可能だが、念のため後方互換で両候補を見る。
_GROOT_CANDIDATES = [
    Path.home() / "Isaac-GR00T",
    Path(__file__).resolve().parents[1] / "Isaac-GR00T",
]
GROOT_REPO = next((p for p in _GROOT_CANDIDATES if p.is_dir()), _GROOT_CANDIDATES[0])
if GROOT_REPO.is_dir():
    sys.path.insert(0, str(GROOT_REPO))

from gr00t.policy.server_client import PolicyClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot_robot_cello.config_starai_cello import StaraiCelloConfig
from lerobot_robot_cello.starai_cello import StaraiCello

PREVIEW_WINDOW_NAME = "GR00T StarAI Preview"
DEFAULT_LOG_ROOT = GROOT_REPO / "gr00t/eval/real_robot/StarAI/logs"
_MJPG_FOURCC = "MJPG"
ROBOT_ACTION_KEYS = [
    "Motor_0.pos",
    "Motor_1.pos",
    "Motor_2.pos",
    "Motor_3.pos",
    "Motor_4.pos",
    "Motor_5.pos",
    "gripper.pos",
]

TRAINING_INITIAL_POSE: dict[str, float] = {
    "Motor_0": 0.03,
    "Motor_1": -100.0,
    "Motor_2": 59.56,
    "Motor_3": -0.44,
    "Motor_4": 30.32,
    "Motor_5": -0.75,
    "gripper": 50.0,
}

MAX_ACTION_STEP: dict[str, float] = {
    "Motor_0.pos": 2.0,
    "Motor_1.pos": 5.0,
    "Motor_2.pos": 4.0,
    "Motor_3.pos": 2.0,
    "Motor_4.pos": 4.0,
    "Motor_5.pos": 0.15,
    "gripper.pos": 10.0,
}

DEFAULT_TRAINING_GUIDE_TOLERANCE: dict[str, float] = {
    "Motor_0.pos": 3.0,
    "Motor_1.pos": 6.0,
    "Motor_2.pos": 8.0,
    "Motor_3.pos": 3.0,
    "Motor_4.pos": 3.0,
    "Motor_5.pos": 0.3,
    "gripper.pos": 6.0,
}

DEFAULT_TRAINING_DATASET_CANDIDATES = [
    Path("/media/tmc/DATA/data/original_data/20260427_data"),
    Path("/media/tmc/DATA/data/GR00T/20260427_data"),
    Path("/media/tmc/data/data/original_data/20260427_data"),
    Path("/media/tmc/data/data/GR00T/20260427_data"),
]


def _patched_move_to_initial_position(self: StaraiCello) -> dict[str, Any]:
    position = self.get_action()
    goal_pos = {
        key.removesuffix(".pos"): val
        for key, val in position.items()
        if key.endswith(".pos")
    }
    goal_pos.update(TRAINING_INITIAL_POSE)
    self.bus.sync_write("Goal_Position", goal_pos, motion_time=1500)
    time.sleep(1.5)
    return {f"{motor}.pos": val for motor, val in goal_pos.items()}


StaraiCello.move_to_initial_position = _patched_move_to_initial_position


def clamp_action_to_step(
    action: dict[str, float],
    reference_state: dict[str, float],
    max_step: dict[str, float],
) -> dict[str, float]:
    clamped: dict[str, float] = {}
    for key, target in action.items():
        ref = reference_state.get(key)
        limit = max_step.get(key)
        if ref is None or limit is None:
            clamped[key] = target
            continue
        clamped[key] = float(np.clip(target, ref - limit, ref + limit))
    return clamped


def apply_task_safety_overrides(
    action: dict[str, float],
    reference_state: dict[str, float],
    keep_gripper_open_until_motor1: float,
    gripper_open_value: float,
    elapsed_sec: float,
    keep_gripper_open_until_sec: float,
) -> dict[str, float]:
    if elapsed_sec > keep_gripper_open_until_sec:
        return action
    if reference_state.get("Motor_1.pos", -100.0) >= keep_gripper_open_until_motor1:
        return action

    guarded = dict(action)
    current_gripper = reference_state.get("gripper.pos", gripper_open_value)
    guarded["gripper.pos"] = max(
        float(guarded.get("gripper.pos", current_gripper)),
        float(current_gripper),
        float(gripper_open_value),
    )
    return guarded


class TrainingActionGuide:
    def __init__(
        self,
        dataset_path: Path,
        source_file: Path,
        timestamps: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        self.dataset_path = dataset_path
        self.source_file = source_file
        self.timestamps = timestamps
        self.actions = actions

    def action_at(self, elapsed_sec: float) -> dict[str, float]:
        idx = int(np.searchsorted(self.timestamps, elapsed_sec, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.timestamps) - 1))
        return {
            key: float(self.actions[idx, key_idx])
            for key_idx, key in enumerate(ROBOT_ACTION_KEYS)
        }

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_path": str(self.dataset_path),
            "source_file": str(self.source_file),
            "num_frames": int(len(self.timestamps)),
            "start_sec": float(self.timestamps[0]),
            "end_sec": float(self.timestamps[-1]),
        }


def _find_first_episode_parquet(dataset_path: Path) -> Path:
    data_root = dataset_path.expanduser() / "data"
    episode_files = sorted(data_root.glob("**/episode_*.parquet"))
    if episode_files:
        return episode_files[0]
    file_parquets = sorted(data_root.glob("**/file-*.parquet"))
    if file_parquets:
        return file_parquets[0]
    raise FileNotFoundError(f"No episode parquet files found under {data_root}")


def resolve_training_dataset_path(dataset_path: str | None) -> Path | None:
    if dataset_path:
        return Path(dataset_path).expanduser()
    for candidate in DEFAULT_TRAINING_DATASET_CANDIDATES:
        if (candidate / "data").is_dir():
            return candidate
    return None


def normalize_training_action_guide(mode: str) -> str:
    if mode in {"on", "true", "yes", "1"}:
        return "source_envelope"
    return mode


def load_training_action_guide(dataset_path: str | None) -> TrainingActionGuide | None:
    dataset_root = resolve_training_dataset_path(dataset_path)
    if dataset_root is None:
        return None

    source_file = _find_first_episode_parquet(dataset_root)
    df = pd.read_parquet(source_file)
    if "episode_index" in df.columns:
        first_episode_index = int(df["episode_index"].min())
        df = df[df["episode_index"] == first_episode_index]
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index")

    actions = np.stack(
        [np.asarray(action, dtype=np.float64) for action in df["action"].to_list()],
        axis=0,
    )
    if actions.shape[1] != len(ROBOT_ACTION_KEYS):
        raise ValueError(f"Expected {len(ROBOT_ACTION_KEYS)} action dims, got {actions.shape[1]}")

    if "timestamp" in df.columns:
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    elif "frame_index" in df.columns:
        timestamps = df["frame_index"].to_numpy(dtype=np.float64) / 30.0
    else:
        timestamps = np.arange(len(df), dtype=np.float64) / 30.0
    timestamps = timestamps - float(timestamps[0])
    return TrainingActionGuide(dataset_root, source_file, timestamps, actions)


def to_json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return to_json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    return value


def get_policy_modality_config(policy_client: PolicyClient) -> dict[str, Any] | None:
    try:
        return to_json_safe(policy_client.get_modality_config())
    except Exception as exc:
        logging.warning("Failed to fetch policy modality config from server: %s", exc)
        return None


def apply_training_action_guide(
    action: dict[str, float],
    elapsed_sec: float,
    guide: TrainingActionGuide | None,
    mode: str,
    until_sec: float,
    setup_sec: float,
) -> dict[str, float]:
    mode = normalize_training_action_guide(mode)
    if guide is None or mode == "off" or elapsed_sec > until_sec:
        return action

    source_action = guide.action_at(elapsed_sec)
    if mode == "source_replay" or elapsed_sec <= setup_sec:
        return {key: float(source_action.get(key, value)) for key, value in action.items()}

    if mode != "source_envelope":
        raise ValueError(
            f"Unsupported training_action_guide={mode!r}; "
            "use 'off', 'source_envelope', or 'source_replay'."
        )

    guided = dict(action)
    for key, source_value in source_action.items():
        if key not in guided:
            continue
        tolerance = DEFAULT_TRAINING_GUIDE_TOLERANCE[key]
        guided[key] = float(
            np.clip(float(guided[key]), source_value - tolerance, source_value + tolerance)
        )
    return guided


def select_receding_horizon_action_index(
    actions: list[dict[str, float]],
    reference_state: dict[str, float],
    strategy: str,
    action_select_index: int,
    approach_motor1_until: float,
) -> int:
    if not actions:
        raise ValueError("Policy returned an empty action chunk.")

    fixed_index = min(max(action_select_index, 0), len(actions) - 1)
    if strategy == "fixed_index":
        return fixed_index

    raise ValueError(
        f"Unsupported action_selection_strategy={strategy!r}; "
        "use 'fixed_index'."
    )


def send_robot_action(
    robot: StaraiCello,
    action: dict[str, float],
    motion_time_ms: int,
) -> dict[str, float]:
    if motion_time_ms <= 0:
        return robot.send_action(action)

    goal_pos = {
        key.removesuffix(".pos"): val
        for key, val in action.items()
        if key.endswith(".pos")
    }
    robot.bus.sync_write("Goal_Position", goal_pos, motion_time=motion_time_ms)
    return {f"{motor}.pos": val for motor, val in goal_pos.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StarAI Cello real-robot GR00T policy evaluation.",
    )
    # Policy server
    parser.add_argument("--policy-host", default="localhost", help="GR00T policy server host.")
    parser.add_argument("--policy-port", type=int, default=5555, help="GR00T policy server port.")

    # Robot
    parser.add_argument("--robot-port", required=True, help="StarAI robot serial port.")
    parser.add_argument("--robot-id", required=True, help="StarAI robot calibration id.")

    # Cameras (finetuning used side + rear + onhand)
    parser.add_argument("--side-camera", required=True, help="Side camera device path or index.")
    parser.add_argument("--rear-camera", required=True, help="Rear camera device path or index.")
    parser.add_argument("--onhand-camera", required=True, help="On-hand camera device path or index.")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera capture height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera capture fps.")

    # Inference
    parser.add_argument("--task", required=True, help="Task description for language conditioning.")
    parser.add_argument("--action-horizon", type=int, default=16, help="Total action chunk length from model.")
    parser.add_argument(
        "--execute-steps", type=int, default=8,
        help="Steps to execute per chunk before re-observing. Use 8 or 16 for autonomous "
             "rollouts; execute-steps=1 is a slow diagnostic mode.",
    )
    parser.add_argument(
        "--action-select-index",
        type=int,
        default=0,
        help="Action chunk index to execute when --execute-steps is 1.",
    )
    parser.add_argument(
        "--action-selection-strategy",
        choices=["fixed_index"],
        default="fixed_index",
        help="When --execute-steps is 1, choose how to select one action from the chunk.",
    )
    parser.add_argument(
        "--approach-motor1-until",
        type=float,
        default=20.0,
        help="Use the approach_motor1 selector until Motor_1 reaches this value.",
    )
    parser.add_argument(
        "--allow-slow-receding-horizon",
        action="store_true",
        help="Allow execute-steps=1 diagnostic runs even though they cannot match the 30Hz training rate.",
    )
    parser.add_argument(
        "--policy-samples",
        type=int,
        default=1,
        help="Number of policy samples to aggregate per observation.",
    )
    parser.add_argument(
        "--action-aggregation",
        choices=["median", "mean"],
        default="median",
        help="Aggregation used when --policy-samples is greater than 1.",
    )
    parser.add_argument("--control-fps", type=int, default=30, help="Control loop fps.")
    parser.add_argument(
        "--inference-prefetch-steps",
        type=int,
        default=6,
        help="Start the next policy request this many action steps before the current chunk ends.",
    )
    parser.add_argument(
        "--motion-time-ms",
        type=int,
        default=0,
        help="StarAI servo motion_time for each Goal_Position command. Use <=0 for the robot default.",
    )
    parser.add_argument(
        "--no-clamp-action-step",
        action="store_true",
        help="Disable per-cycle action step limiting before sending commands.",
    )
    parser.add_argument(
        "--keep-gripper-open-until-motor1",
        type=float,
        default=20.0,
        help="Keep the gripper at least --gripper-open-value until Motor_1 reaches this value.",
    )
    parser.add_argument(
        "--gripper-open-value",
        type=float,
        default=80.0,
        help="Minimum gripper command while approaching the workpiece.",
    )
    parser.add_argument(
        "--keep-gripper-open-until-sec",
        type=float,
        default=0.0,
        help="Apply the open-gripper guard only for this many seconds. The default 0 keeps policy control autonomous.",
    )
    parser.add_argument(
        "--training-action-guide",
        choices=["off", "on", "source_envelope", "source_replay"],
        default="source_envelope",
        help="Optional diagnostic/fallback mode that constrains actions using the first source episode.",
    )
    parser.add_argument(
        "--training-action-guide-until-sec",
        type=float,
        default=12.0,
        help="Stop applying the source action guide after this rollout time.",
    )
    parser.add_argument(
        "--training-action-guide-setup-sec",
        type=float,
        default=2.05,
        help="Replay the source setup action exactly until this rollout time.",
    )
    parser.add_argument(
        "--duration-sec", type=float, default=60.0,
        help="Maximum autonomous runtime duration in seconds.",
    )
    parser.add_argument(
        "--expected-model-path",
        help="Expected GR00T checkpoint path served by the policy server; recorded in session_meta.json.",
    )
    parser.add_argument(
        "--training-dataset-path",
        help="Optional GR00T dataset root used to derive the episode-start initial pose.",
    )
    parser.add_argument(
        "--log-root",
        default=str(DEFAULT_LOG_ROOT),
        help="Directory where GR00T rollout logs are written.",
    )
    parser.add_argument(
        "--record-video",
        dest="record_video",
        action="store_true",
        help="Record side/rear/onhand videos into the rollout log directory.",
    )
    parser.add_argument(
        "--no-record-video",
        dest="record_video",
        action="store_false",
        help="Do not record side/rear/onhand videos.",
    )
    parser.add_argument(
        "--run-log",
        dest="run_log",
        action="store_true",
        help="Write joint_log.csv and session_meta.json for this rollout.",
    )
    parser.add_argument(
        "--no-run-log",
        dest="run_log",
        action="store_false",
        help="Disable rollout logging.",
    )
    parser.add_argument("--display-data", action="store_true", help="Show live camera preview.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and print settings without connecting hardware.",
    )
    parser.set_defaults(record_video=True, run_log=True)
    return parser.parse_args()


def _resolve_camera_source(value: str) -> int | Path:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    match = re.fullmatch(r"/dev/video(\d+)", stripped)
    if match is not None:
        return int(match.group(1))
    return Path(stripped)


def _camera_device_index_from_path(camera_id: str) -> int | None:
    match = re.fullmatch(r"/dev/video(\d+)", camera_id.strip())
    if match is None:
        return None
    return int(match.group(1))


def _probe_camera_supports_mjpg(device_path: str) -> bool:
    if cv2 is None:
        return False
    idx = _camera_device_index_from_path(device_path)
    if idx is None:
        return False
    try:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return False
        try:
            cap.set(cv2.CAP_PROP_FOURCC, float(cv2.VideoWriter_fourcc(*_MJPG_FOURCC)))
            actual = int(cap.get(cv2.CAP_PROP_FOURCC))
            return actual == cv2.VideoWriter_fourcc(*_MJPG_FOURCC)
        finally:
            cap.release()
    except Exception:
        return False


def _resolve_training_initial_pose(dataset_path: str | None) -> dict[str, float]:
    dataset_root = resolve_training_dataset_path(dataset_path)
    if dataset_root is None:
        return dict(TRAINING_INITIAL_POSE)

    try:
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        from verify_initial_pose import (  # type: ignore
            MOTOR_NAMES,
            collect_episode_start_states,
        )

        starts = collect_episode_start_states(dataset_root)
        return {
            motor_name: round(float(starts[0, axis_index]), 2)
            for axis_index, motor_name in enumerate(MOTOR_NAMES)
        }
    except Exception as exc:
        logging.warning(
            "Failed to derive initial pose from training dataset %s; using embedded defaults: %s",
            dataset_root,
            exc,
        )
        return dict(TRAINING_INITIAL_POSE)


def _set_training_initial_pose(initial_pose: dict[str, float]) -> None:
    TRAINING_INITIAL_POSE.clear()
    TRAINING_INITIAL_POSE.update(initial_pose)


def recursive_add_extra_dim(obs: dict) -> dict:
    """Recursively add an extra dim to arrays or scalars.

    GR00T Policy Server expects obs with shape (batch=1, time=1, ...).
    Call this twice to add both B and T dimensions.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


def recursive_repeat_batch(value: Any, batch_size: int) -> Any:
    if batch_size <= 1:
        return value
    if isinstance(value, np.ndarray):
        return np.repeat(value, batch_size, axis=0)
    if isinstance(value, dict):
        return {key: recursive_repeat_batch(val, batch_size) for key, val in value.items()}
    if isinstance(value, list) and len(value) == 1:
        return [value[0] for _ in range(batch_size)]
    return value


def aggregate_action_chunks(
    chunks: list[list[dict[str, float]]],
    joint_keys: list[str],
    aggregation: str,
) -> list[dict[str, float]]:
    if len(chunks) == 1:
        return chunks[0]

    horizon = min(len(chunk) for chunk in chunks)
    values = np.array(
        [
            [[float(chunk[t][key]) for key in joint_keys] for t in range(horizon)]
            for chunk in chunks
        ],
        dtype=np.float32,
    )
    if aggregation == "median":
        merged = np.median(values, axis=0)
    elif aggregation == "mean":
        merged = np.mean(values, axis=0)
    else:
        raise ValueError(f"Unsupported action_aggregation={aggregation!r}; use 'median' or 'mean'.")

    return [
        {key: float(merged[t, key_idx]) for key_idx, key in enumerate(joint_keys)}
        for t in range(horizon)
    ]


def validate_control_config(args: argparse.Namespace, execute_steps: int) -> None:
    args.training_action_guide = normalize_training_action_guide(args.training_action_guide)
    if args.training_action_guide not in {"off", "source_envelope", "source_replay"}:
        raise ValueError(
            "training_action_guide must be 'off', 'on', 'source_envelope', or 'source_replay'."
        )
    if args.training_action_guide_until_sec < 0:
        raise ValueError("training_action_guide_until_sec must be >= 0.")
    if args.training_action_guide_setup_sec < 0:
        raise ValueError("training_action_guide_setup_sec must be >= 0.")
    if args.keep_gripper_open_until_sec < 0:
        raise ValueError("keep_gripper_open_until_sec must be >= 0.")
    if args.inference_prefetch_steps < 0:
        raise ValueError("inference-prefetch-steps must be >= 0.")
    if args.action_selection_strategy != "fixed_index":
        raise ValueError("action_selection_strategy must be 'fixed_index'.")
    if execute_steps <= 1 and not args.allow_slow_receding_horizon:
        raise ValueError(
            "execute-steps=1 runs at policy-inference speed instead of the 30Hz training rate "
            "and is only useful for diagnostics. Use --execute-steps 8 or 16 for autonomous "
            "rollouts, or pass --allow-slow-receding-horizon if you intentionally want "
            "the slow diagnostic mode."
        )
    if execute_steps <= 1 and args.action_select_index != 0:
        raise ValueError(
            "action_select_index must be 0 when execute_steps <= 1. "
            "GR00T action[0] is the immediate command; later indices are future actions."
        )
    if args.policy_samples < 1:
        raise ValueError("policy_samples must be >= 1.")
    if execute_steps <= 1 and args.policy_samples != 1:
        raise ValueError(
            "policy_samples must be 1 when execute_steps <= 1. "
            "Multiple samples slow the real-robot loop far below the 30Hz training data."
        )


class StaraiCelloAdapter:
    """Adapter between StaraiCello robot observations and GR00T VLA format.

    Finetuning config (conf.yaml) used:
      - video keys: side, rear, onhand
      - state keys: single_arm (6 dim = Motor_0..Motor_5), gripper (1 dim)
      - action keys: single_arm (6 dim), gripper (1 dim)
      - language key: annotation.human.action.task_description
    """

    # Motor names matching the dataset feature order
    ARM_MOTOR_KEYS = ROBOT_ACTION_KEYS[:6]
    GRIPPER_KEY = "gripper.pos"
    CAMERA_KEYS = ["side", "rear", "onhand"]

    def __init__(self, policy_client: PolicyClient):
        self.policy = policy_client

    def obs_to_policy_inputs(self, obs: dict[str, Any], task: str, batch_size: int = 1) -> dict:
        """Convert raw robot observation dict into GR00T VLA input format."""
        model_obs = {}

        # (1) Video: side + rear + onhand cameras
        model_obs["video"] = {k: obs[k] for k in self.CAMERA_KEYS}

        # (2) State: single_arm (6 joints) + gripper (1)
        arm_state = np.array(
            [obs[k] for k in self.ARM_MOTOR_KEYS], dtype=np.float32,
        )
        gripper_state = np.array([obs[self.GRIPPER_KEY]], dtype=np.float32)
        model_obs["state"] = {
            "single_arm": arm_state,
            "gripper": gripper_state,
        }

        # (3) Language
        model_obs["language"] = {
            "annotation.human.action.task_description": task,
        }

        # (4) Add (B=1, T=1) dims
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_repeat_batch(model_obs, batch_size)
        return model_obs

    def decode_action_chunk(self, chunk: dict, t: int, batch_index: int = 0) -> dict[str, float]:
        """Decode action chunk at timestep t into robot motor commands.

        chunk["single_arm"]: (B, T, 6)
        chunk["gripper"]:    (B, T, 1)
        """
        single_arm = chunk["single_arm"][batch_index][t]  # (6,)
        gripper = chunk["gripper"][batch_index][t]         # (1,)
        full = np.concatenate([single_arm, gripper], axis=0)  # (7,)

        all_keys = self.ARM_MOTOR_KEYS + [self.GRIPPER_KEY]
        return {name: float(full[i]) for i, name in enumerate(all_keys)}

    def get_action(
        self,
        obs: dict,
        task: str,
        num_samples: int = 1,
        aggregation: str = "median",
    ) -> list[dict[str, float]]:
        num_samples = max(1, int(num_samples))
        chunks = self._get_batched_action_chunks(obs, task, num_samples)
        return aggregate_action_chunks(chunks, ROBOT_ACTION_KEYS, aggregation)

    def _get_batched_action_chunks(
        self,
        obs: dict,
        task: str,
        batch_size: int,
    ) -> list[list[dict[str, float]]]:
        """Run inference and return a list of motor commands (one per timestep)."""
        model_input = self.obs_to_policy_inputs(obs, task, batch_size=batch_size)
        action_chunk, info = self.policy.get_action(model_input)

        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) -> T
        actual_batch_size = action_chunk[any_key].shape[0]

        return [
            [self.decode_action_chunk(action_chunk, t, batch_index=b) for t in range(horizon)]
            for b in range(actual_batch_size)
        ]


class RolloutLogger:
    def __init__(
        self,
        args: argparse.Namespace,
        resolved: dict[str, Any],
        initial_pose: dict[str, float],
    ) -> None:
        self.enabled = bool(args.run_log)
        self.record_video = bool(args.record_video)
        self.session_dir: Path | None = None
        self.csv_file = None
        self.csv_writer: csv.DictWriter | None = None
        self.video_writers: dict[str, Any] = {}
        self.video_sizes: dict[str, tuple[int, int]] = {}
        self.video_paths: dict[str, str] = {}
        self.control_fps = int(args.control_fps)
        self.start_time = time.monotonic()

        if not self.enabled:
            return

        log_root = Path(args.log_root).expanduser()
        base_name = time.strftime("%Y%m%d_%H%M%S")
        session_dir = log_root / base_name
        if session_dir.exists():
            session_dir = log_root / f"{base_name}_{time.time_ns()}"
        session_dir.mkdir(parents=True, exist_ok=False)
        self.session_dir = session_dir

        session_meta = {
            "created_at_unix": time.time(),
            "runtime": "groot_starai_predict_runtime",
            "expected_model_path": (
                str(Path(args.expected_model_path).expanduser().resolve())
                if args.expected_model_path
                else None
            ),
            "training_dataset_path": (
                str(Path(args.training_dataset_path).expanduser().resolve())
                if args.training_dataset_path
                else None
            ),
            "initial_pose": initial_pose,
            "resolved_runtime": resolved,
            "action_keys": ROBOT_ACTION_KEYS,
            "camera_keys": StaraiCelloAdapter.CAMERA_KEYS,
        }
        (session_dir / "session_meta.json").write_text(
            json.dumps(session_meta, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        self.csv_file = (session_dir / "joint_log.csv").open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=["step", "elapsed_sec", "type", *ROBOT_ACTION_KEYS],
        )
        self.csv_writer.writeheader()
        logging.info("Saving GR00T rollout log to %s", session_dir)

        if self.record_video and cv2 is None:
            logging.warning("OpenCV is unavailable; disabling rollout video recording.")
            self.record_video = False

    def reset_clock(self) -> None:
        self.start_time = time.monotonic()

    def _write_row(self, step: int, row_type: str, values: dict[str, Any]) -> None:
        if self.csv_writer is None:
            return
        row: dict[str, Any] = {
            "step": step,
            "elapsed_sec": f"{time.monotonic() - self.start_time:.6f}",
            "type": row_type,
        }
        for key in ROBOT_ACTION_KEYS:
            value = values.get(key)
            row[key] = "" if value is None else float(value)
        self.csv_writer.writerow(row)
        if self.csv_file is not None:
            self.csv_file.flush()

    def _record_videos(self, observation: dict[str, Any]) -> None:
        if not self.record_video or self.session_dir is None or cv2 is None:
            return

        for camera_name in StaraiCelloAdapter.CAMERA_KEYS:
            image = observation.get(camera_name)
            if not isinstance(image, np.ndarray) or image.ndim != 3:
                continue
            frame_bgr = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            height, width = frame_bgr.shape[:2]
            writer = self.video_writers.get(camera_name)
            if writer is None:
                video_path = self.session_dir / f"{camera_name}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    float(self.control_fps),
                    (width, height),
                )
                if not writer.isOpened():
                    logging.warning("Failed to open rollout video writer: %s", video_path)
                    continue
                self.video_writers[camera_name] = writer
                self.video_sizes[camera_name] = (width, height)
                self.video_paths[camera_name] = str(video_path)
            else:
                target_width, target_height = self.video_sizes[camera_name]
                if (width, height) != (target_width, target_height):
                    frame_bgr = cv2.resize(frame_bgr, (target_width, target_height))
            writer.write(frame_bgr)

    def record_state(self, step: int, observation: dict[str, Any]) -> None:
        self._write_row(step, "state", observation)
        self._record_videos(observation)

    def record_action_chunk(self, step: int, actions: list[dict[str, float]]) -> None:
        for action_index, action in enumerate(actions):
            self._write_row(step, f"action_{action_index}", action)

    def record_sent_action(self, step: int, action: dict[str, float]) -> None:
        self._write_row(step, "sent_action", action)

    def record_guided_action(self, step: int, action: dict[str, float]) -> None:
        self._write_row(step, "guided_action", action)

    def close(self) -> None:
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None


def _build_robot(args: argparse.Namespace) -> StaraiCello:
    camera_sources = {
        "side": args.side_camera,
        "rear": args.rear_camera,
        "onhand": args.onhand_camera,
    }
    cameras: dict[str, OpenCVCameraConfig] = {}
    for name, device in camera_sources.items():
        fourcc = _MJPG_FOURCC if _probe_camera_supports_mjpg(device) else None
        if fourcc:
            logging.info("Camera %s (%s) using MJPG fourcc for bandwidth savings.", name, device)
        cameras[name] = OpenCVCameraConfig(
            index_or_path=_resolve_camera_source(device),
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
            fourcc=fourcc,
        )

    robot_config = StaraiCelloConfig(
        id=args.robot_id,
        port=args.robot_port,
        cameras=cameras,
    )
    return StaraiCello(robot_config)


def _build_preview_frame(observation: dict[str, Any]) -> np.ndarray | None:
    if cv2 is None:
        return None
    frames = []
    for cam_name in StaraiCelloAdapter.CAMERA_KEYS:
        image = observation.get(cam_name)
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            continue
        frames.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not frames:
        return None
    return np.hstack(frames)


def _request_inference(
    adapter: StaraiCelloAdapter,
    obs: dict,
    task: str,
    policy_samples: int,
    action_aggregation: str,
) -> list[dict[str, float]]:
    """Run inference in a thread-safe way (called from background thread)."""
    return adapter.get_action(
        obs,
        task,
        num_samples=policy_samples,
        aggregation=action_aggregation,
    )


def main() -> int:
    args = parse_args()
    args.training_action_guide = normalize_training_action_guide(args.training_action_guide)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    execute_steps = args.execute_steps if args.execute_steps is not None else args.action_horizon
    validate_control_config(args, execute_steps)
    resolved_training_dataset_path = resolve_training_dataset_path(args.training_dataset_path)
    initial_pose = _resolve_training_initial_pose(args.training_dataset_path)
    _set_training_initial_pose(initial_pose)
    training_action_guide = (
        load_training_action_guide(args.training_dataset_path)
        if args.training_action_guide != "off"
        else None
    )

    resolved = {
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "expected_model_path": args.expected_model_path,
        "training_dataset_path": args.training_dataset_path,
        "resolved_training_dataset_path": (
            str(resolved_training_dataset_path) if resolved_training_dataset_path is not None else None
        ),
        "robot_port": args.robot_port,
        "robot_id": args.robot_id,
        "side_camera": args.side_camera,
        "rear_camera": args.rear_camera,
        "onhand_camera": args.onhand_camera,
        "task": args.task,
        "action_horizon": args.action_horizon,
        "execute_steps": execute_steps,
        "action_select_index": args.action_select_index,
        "action_selection_strategy": args.action_selection_strategy,
        "approach_motor1_until": args.approach_motor1_until,
        "allow_slow_receding_horizon": args.allow_slow_receding_horizon,
        "policy_samples": args.policy_samples,
        "action_aggregation": args.action_aggregation,
        "clamp_action_step": not args.no_clamp_action_step,
        "keep_gripper_open_until_motor1": args.keep_gripper_open_until_motor1,
        "gripper_open_value": args.gripper_open_value,
        "keep_gripper_open_until_sec": args.keep_gripper_open_until_sec,
        "training_action_guide": args.training_action_guide,
        "training_action_guide_until_sec": args.training_action_guide_until_sec,
        "training_action_guide_setup_sec": args.training_action_guide_setup_sec,
        "training_action_guide_tolerance": DEFAULT_TRAINING_GUIDE_TOLERANCE,
        "training_action_guide_source": (
            training_action_guide.summary() if training_action_guide is not None else None
        ),
        "control_fps": args.control_fps,
        "inference_prefetch_steps": args.inference_prefetch_steps,
        "motion_time_ms": args.motion_time_ms,
        "duration_sec": args.duration_sec,
        "log_root": args.log_root,
        "run_log": args.run_log,
        "record_video": args.record_video,
        "display_data": args.display_data,
        "initial_pose": initial_pose,
    }
    logging.info("Resolved GR00T StarAI runtime: %s", json.dumps(resolved, ensure_ascii=True))
    if args.dry_run:
        return 0
    if training_action_guide is not None:
        logging.info("Using training action guide: %s", training_action_guide.summary())

    # Connect to GR00T policy server
    policy_client = PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=15000,
        strict=False,
    )
    if not policy_client.ping():
        logging.error("Cannot connect to GR00T policy server at %s:%d", args.policy_host, args.policy_port)
        return 1
    logging.info("Connected to GR00T policy server at %s:%d", args.policy_host, args.policy_port)
    resolved["policy_modality_config"] = get_policy_modality_config(policy_client)

    adapter = StaraiCelloAdapter(policy_client)

    # Build and connect robot
    preview_enabled = args.display_data and cv2 is not None
    loop_count = 0
    inference_step = 0
    started = time.monotonic()
    control_started = started
    deadline = started + args.duration_sec
    control_period = 1.0 / args.control_fps

    executor = ThreadPoolExecutor(max_workers=1)
    pending_future: Future | None = None
    pending_future_step: int | None = None
    current_actions: list[dict[str, float]] | None = None

    rollout_logger = RolloutLogger(args, resolved, initial_pose)
    robot = _build_robot(args)
    try:
        robot.connect()
        logging.info("StarAI robot connected. Running async GR00T inference for up to %.1fs.", args.duration_sec)

        # First blocking inference to get the initial action chunk
        control_started = time.monotonic()
        deadline = control_started + args.duration_sec
        rollout_logger.reset_clock()
        obs = robot.get_observation()
        rollout_logger.record_state(inference_step, obs)
        current_actions = adapter.get_action(
            obs,
            args.task,
            num_samples=args.policy_samples,
            aggregation=args.action_aggregation,
        )
        rollout_logger.record_action_chunk(inference_step, current_actions)
        inference_step += 1
        logging.info("First inference complete. Starting async control loop.")

        while time.monotonic() < deadline:
            # Execute actions from the current chunk
            reference_state = {key: float(obs[key]) for key in ROBOT_ACTION_KEYS}
            if execute_steps <= 1:
                selected_index = select_receding_horizon_action_index(
                    current_actions,
                    reference_state,
                    args.action_selection_strategy,
                    args.action_select_index,
                    args.approach_motor1_until,
                )
                action_indices = [selected_index]
            else:
                action_indices = list(range(min(execute_steps, len(current_actions))))

            prefetch_at = max(0, len(action_indices) - max(1, args.inference_prefetch_steps))

            for local_action_idx, i in enumerate(action_indices):
                if time.monotonic() >= deadline:
                    break

                tic = time.perf_counter()
                guide_elapsed_sec = time.monotonic() - control_started
                selected_action = apply_training_action_guide(
                    current_actions[i],
                    guide_elapsed_sec,
                    training_action_guide,
                    args.training_action_guide,
                    args.training_action_guide_until_sec,
                    args.training_action_guide_setup_sec,
                )
                selected_action = apply_task_safety_overrides(
                    selected_action,
                    reference_state,
                    args.keep_gripper_open_until_motor1,
                    args.gripper_open_value,
                    guide_elapsed_sec,
                    args.keep_gripper_open_until_sec,
                )
                if selected_action != current_actions[i]:
                    rollout_logger.record_guided_action(loop_count + 1, selected_action)
                if args.no_clamp_action_step:
                    action_to_send = selected_action
                else:
                    action_to_send = clamp_action_to_step(
                        selected_action,
                        reference_state,
                        MAX_ACTION_STEP,
                    )
                action_to_send = send_robot_action(robot, action_to_send, args.motion_time_ms)
                reference_state = action_to_send
                loop_count += 1
                rollout_logger.record_sent_action(loop_count, action_to_send)

                if preview_enabled:
                    try:
                        preview_frame = _build_preview_frame(obs)
                        if preview_frame is not None:
                            cv2.imshow(PREVIEW_WINDOW_NAME, preview_frame)
                            if cv2.waitKey(1) & 0xFF == 27:
                                logging.info("ESC pressed; stopping.")
                                raise KeyboardInterrupt
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        logging.warning("Disabling preview: %s", exc)
                        preview_enabled = False

                elapsed = time.perf_counter() - tic
                remaining = control_period - elapsed
                if remaining > 0:
                    time.sleep(remaining)

                if local_action_idx == prefetch_at and pending_future is None:
                    obs = robot.get_observation()
                    rollout_logger.record_state(inference_step, obs)
                    pending_future_step = inference_step
                    inference_step += 1
                    pending_future = executor.submit(
                        _request_inference,
                        adapter,
                        obs,
                        args.task,
                        args.policy_samples,
                        args.action_aggregation,
                    )

            # Swap in the next chunk from the async inference
            if pending_future is not None:
                current_actions = pending_future.result()
                rollout_logger.record_action_chunk(
                    pending_future_step if pending_future_step is not None else inference_step,
                    current_actions,
                )
                pending_future = None
                pending_future_step = None
            else:
                # Fallback: synchronous inference if no future was started
                obs = robot.get_observation()
                rollout_logger.record_state(inference_step, obs)
                current_actions = adapter.get_action(
                    obs,
                    args.task,
                    num_samples=args.policy_samples,
                    aggregation=args.action_aggregation,
                )
                rollout_logger.record_action_chunk(inference_step, current_actions)
                inference_step += 1

            if loop_count % 100 == 0 and loop_count > 0:
                logging.info(
                    "Running: elapsed=%.1fs loops=%d",
                    time.monotonic() - started,
                    loop_count,
                )

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt; stopping.")
    finally:
        executor.shutdown(wait=False)
        rollout_logger.close()
        if preview_enabled and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        try:
            if robot.is_connected and time.monotonic() >= deadline:
                robot.move_to_initial_position()
            if robot.is_connected:
                robot.disconnect()
        except Exception as exc:
            logging.warning("Failed to disconnect robot: %s", exc)

    logging.info(
        "GR00T StarAI runtime exited after %.2fs (%d action steps).",
        time.monotonic() - started,
        loop_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
