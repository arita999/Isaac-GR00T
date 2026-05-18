"""
StarAI Real-Robot Gr00T Policy Evaluation Script

This script runs closed-loop policy evaluation on the StarAI Cello
(single arm) robot using the GR00T Policy API.

Key differences from eval_so100.py:
    - 6 arm motors (Motor_0..Motor_5) + 1 gripper = 7 control channels
    - 3 cameras (side, rear, onhand) to match starai_single_arm_config.py
    - Language key is resolved from the running GR00T server's modality config.
    - The stock StarAiCello.connect() calls ``move_to_initial_position()``
      with a fixed home pose [0, -100, 60, 0, 30, 0, 50]. We override that
      home pose with the task-start pose from the StarAI pick_and_place
      training data so that the policy starts inside the reset distribution.
"""

# =============================================================================
# Imports
# =============================================================================

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
import json
import logging
import os
from pprint import pformat
import time
from typing import Any, Dict, List, Tuple

import cv2
import draccus
from gr00t.policy.server_client import PolicyClient

# Importing various robot configs ensures CLI autocompletion works
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.utils.utils import init_logging, log_say
import numpy as np
import pandas as pd

# Register StarAI Cello robot type (side-effect import)
from lerobot.robots.starai_follower import StaraiCello


# Task-start reset pose from the source StarAI pick_and_place data. Do not use
# all-episode medians here: this dataset contains starts from intermediate task
# phases, and those medians put the robot outside the reset distribution.
TRAINING_INITIAL_POSE: Dict[str, float] = {
    "Motor_0": 0.03,
    "Motor_1": -100.0,
    "Motor_2": 59.56,
    "Motor_3": -0.44,
    "Motor_4": 30.32,
    "Motor_5": -0.75,
    "gripper": 50.0,
}

MOTOR_NAMES = ["Motor_0", "Motor_1", "Motor_2", "Motor_3", "Motor_4", "Motor_5", "gripper"]

DEFAULT_TRAINING_DATASET_CANDIDATES = [
    # Current default checkpoint is trained from the n710 StarAI dataset.
    "/media/tmc/DATA/data/original_data/20260427_data_n710",
    # Older checkpoints used the 300-episode original_data copy.
    "/media/tmc/DATA/data/original_data/20260427_data",
    "/media/tmc/DATA/data/GR00T/20260427_data",
    "/media/tmc/data/data/original_data/20260427_data",
    "/media/tmc/data/data/GR00T/20260427_data",
]

MAX_ACTION_STEP: Dict[str, float] = {
    "Motor_0.pos": 4.0,
    "Motor_1.pos": 15.0,
    "Motor_2.pos": 12.0,
    "Motor_3.pos": 3.0,
    "Motor_4.pos": 6.0,
    "Motor_5.pos": 0.3,
    "gripper.pos": 10.0,
}

DEFAULT_TRAINING_GUIDE_TOLERANCE: Dict[str, float] = {
    "Motor_0.pos": 3.0,
    "Motor_1.pos": 6.0,
    "Motor_2.pos": 8.0,
    "Motor_3.pos": 3.0,
    "Motor_4.pos": 3.0,
    "Motor_5.pos": 0.3,
    "gripper.pos": 6.0,
}

MIN_ARM_MOTION_TIME_MS = 100


def _patched_move_to_initial_position(self: StaraiCello) -> Dict[str, Any]:
    """Replacement for StaraiCello.move_to_initial_position().

    Drives all 7 motors to ``TRAINING_INITIAL_POSE`` instead of the
    hard-coded home pose. This keeps the robot near the training-data
    episode-start distribution, which is required for the policy to
    behave sensibly from the very first step.
    """
    goal_pos = self.bus.sync_read("Present_Position")
    goal_pos.update(TRAINING_INITIAL_POSE)
    self.bus.sync_write("Goal_Position", goal_pos, motion_time=1500)
    if hasattr(self, "_last_commanded_positions"):
        self._last_commanded_positions.update({motor: float(pos) for motor, pos in goal_pos.items()})
    time.sleep(1.5)
    return {f"{motor}.pos": val for motor, val in goal_pos.items()}


# Monkey-patch once at import time so the override is in place before
# robot.connect() (which calls move_to_initial_position) is invoked.
StaraiCello.move_to_initial_position = _patched_move_to_initial_position


def collect_episode_start_states(dataset_path: str) -> np.ndarray:
    root = os.path.expanduser(dataset_path)

    episode_files = sorted(
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(os.path.join(root, "data"))
        for filename in filenames
        if filename.startswith("episode_") and filename.endswith(".parquet")
    )
    if episode_files:
        starts = []
        for path in episode_files:
            df = pd.read_parquet(path, columns=["observation.state"])
            starts.append(np.asarray(df.iloc[0]["observation.state"], dtype=np.float64))
        return np.stack(starts, axis=0)

    file_parquets = sorted(
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(os.path.join(root, "data"))
        for filename in filenames
        if filename.startswith("file-") and filename.endswith(".parquet")
    )
    if file_parquets:
        first_rows: Dict[int, Any] = {}
        for path in file_parquets:
            df = pd.read_parquet(path)
            for ep_idx, group in df.groupby("episode_index"):
                row = group.sort_values("frame_index").iloc[0]
                current = first_rows.get(int(ep_idx))
                if current is None or int(row["frame_index"]) < int(current["frame_index"]):
                    first_rows[int(ep_idx)] = row
        return np.stack(
            [
                np.asarray(first_rows[idx]["observation.state"], dtype=np.float64)
                for idx in sorted(first_rows)
            ],
            axis=0,
        )

    raise FileNotFoundError(f"No episode parquet files found under {root}/data")


def resolve_training_dataset_path(dataset_path: str | None) -> str | None:
    if dataset_path:
        return dataset_path

    for candidate in DEFAULT_TRAINING_DATASET_CANDIDATES:
        if os.path.isdir(os.path.join(os.path.expanduser(candidate), "data")):
            return candidate
    return None


def normalize_training_action_guide(mode: Any) -> str:
    if isinstance(mode, bool):
        return "source_envelope" if mode else "off"

    normalized = str(mode).strip().lower()
    if normalized in {"on", "true", "yes", "1"}:
        return "source_envelope"
    if normalized in {"off", "false", "no", "0", "none", "null"}:
        return "off"
    return normalized


def action_source_label(training_action_guide: Any) -> str:
    mode = normalize_training_action_guide(training_action_guide)
    if mode == "off":
        return "learned_policy"
    if mode == "source_replay":
        return "training_dataset_replay"
    if mode == "source_envelope":
        return "learned_policy_clipped_to_training_dataset"
    return mode


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


def get_policy_modality_config(policy_client: PolicyClient) -> Dict[str, Any] | None:
    try:
        return to_json_safe(policy_client.get_modality_config())
    except Exception as exc:
        logging.warning("Failed to fetch policy modality config from server: %s", exc)
        return None


def resolve_policy_language_key(
    requested_language_key: str,
    policy_modality_config: Dict[str, Any] | None,
) -> str:
    if not policy_modality_config:
        return requested_language_key

    language_config = policy_modality_config.get("language")
    if not isinstance(language_config, dict):
        return requested_language_key

    modality_keys = language_config.get("modality_keys")
    if not modality_keys:
        return requested_language_key

    if requested_language_key in modality_keys:
        return requested_language_key

    resolved = modality_keys[0]
    logging.warning(
        "Requested language key %r is not supported by the policy server; using %r instead.",
        requested_language_key,
        resolved,
    )
    return resolved


def resolve_initial_pose(dataset_path: str | None, strategy: str) -> Dict[str, float]:
    if not dataset_path:
        return dict(TRAINING_INITIAL_POSE)

    starts = collect_episode_start_states(dataset_path)
    if starts.shape[1] != len(MOTOR_NAMES):
        raise ValueError(f"Expected {len(MOTOR_NAMES)} state dims, got {starts.shape[1]}")

    if strategy == "first_episode":
        values = starts[0]
    elif strategy == "median":
        values = np.median(starts, axis=0)
    else:
        raise ValueError(
            f"Unsupported initial_pose_strategy={strategy!r}; use 'first_episode' or 'median'."
        )

    return {name: round(float(value), 2) for name, value in zip(MOTOR_NAMES, values)}


@dataclass
class TrainingActionGuide:
    dataset_path: str
    source_file: str
    timestamps: np.ndarray
    actions: np.ndarray

    def action_at(self, elapsed_sec: float) -> Dict[str, float]:
        idx = int(np.searchsorted(self.timestamps, elapsed_sec, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.timestamps) - 1))
        return {
            key: float(self.actions[idx, key_idx])
            for key_idx, key in enumerate(f"{name}.pos" for name in MOTOR_NAMES)
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "source_file": self.source_file,
            "num_frames": int(len(self.timestamps)),
            "start_sec": float(self.timestamps[0]),
            "end_sec": float(self.timestamps[-1]),
        }


def effective_training_action_guide_until_sec(
    guide: TrainingActionGuide | None,
    until_sec: float,
) -> float:
    if guide is None:
        return float(until_sec)
    return min(float(until_sec), float(guide.timestamps[-1]))


def _find_first_episode_parquet(dataset_path: str) -> str:
    root = os.path.expanduser(dataset_path)
    data_root = os.path.join(root, "data")

    episode_files = sorted(
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(data_root)
        for filename in filenames
        if filename.startswith("episode_") and filename.endswith(".parquet")
    )
    if episode_files:
        return episode_files[0]

    file_parquets = sorted(
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(data_root)
        for filename in filenames
        if filename.startswith("file-") and filename.endswith(".parquet")
    )
    if file_parquets:
        return file_parquets[0]

    raise FileNotFoundError(f"No episode parquet files found under {data_root}")


def load_training_action_guide(dataset_path: str) -> TrainingActionGuide:
    source_file = _find_first_episode_parquet(dataset_path)
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
    if actions.shape[1] != len(MOTOR_NAMES):
        raise ValueError(f"Expected {len(MOTOR_NAMES)} action dims, got {actions.shape[1]}")

    if "timestamp" in df.columns:
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    elif "frame_index" in df.columns:
        timestamps = df["frame_index"].to_numpy(dtype=np.float64) / 30.0
    else:
        timestamps = np.arange(len(df), dtype=np.float64) / 30.0

    timestamps = timestamps - float(timestamps[0])
    return TrainingActionGuide(
        dataset_path=os.path.expanduser(dataset_path),
        source_file=source_file,
        timestamps=timestamps,
        actions=actions,
    )


def apply_training_action_guide(
    action: Dict[str, float],
    elapsed_sec: float,
    guide: TrainingActionGuide | None,
    mode: str,
    until_sec: float,
    setup_sec: float,
) -> Dict[str, float]:
    mode = normalize_training_action_guide(mode)
    if guide is None or mode == "off":
        return action
    if elapsed_sec > effective_training_action_guide_until_sec(guide, until_sec):
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


def clamp_action_to_step(
    action: Dict[str, float],
    reference_state: Dict[str, float],
    max_step: Dict[str, float],
) -> Dict[str, float]:
    """Limit each joint command to a bounded step from the latest known state."""
    clamped: Dict[str, float] = {}
    for key, target in action.items():
        ref = reference_state.get(key)
        limit = max_step.get(key)
        if ref is None or limit is None:
            clamped[key] = target
            continue
        clamped[key] = float(np.clip(target, ref - limit, ref + limit))
    return clamped


def apply_task_safety_overrides(
    action: Dict[str, float],
    reference_state: Dict[str, float],
    keep_gripper_open_until_motor1: float,
    gripper_open_value: float,
    elapsed_sec: float,
    keep_gripper_open_until_sec: float,
) -> Dict[str, float]:
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


def select_receding_horizon_action_index(
    raw_actions: List[Dict[str, float]],
    reference_state: Dict[str, float],
    strategy: str,
    action_select_index: int,
    approach_motor1_until: float,
) -> int:
    if not raw_actions:
        raise ValueError("Policy returned an empty action chunk.")

    fixed_index = min(max(action_select_index, 0), len(raw_actions) - 1)
    if strategy == "fixed_index":
        return fixed_index

    raise ValueError(
        f"Unsupported action_selection_strategy={strategy!r}; "
        "use 'fixed_index'."
    )


def action_was_clamped(raw_action: Dict[str, float], sent_action: Dict[str, float]) -> bool:
    return any(
        not np.isclose(float(raw_action[key]), float(sent_action[key]))
        for key in raw_action
    )


def temporal_ensemble_action(
    action_chunks: List[Dict[str, Any]],
    target_tick: int,
    current_step: int,
    joint_keys: List[str],
    decay: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    candidates = []
    for chunk in action_chunks:
        chunk_index = target_tick - int(chunk["base_tick"])
        actions = chunk["actions"]
        if chunk_index < 0 or chunk_index >= len(actions):
            continue
        age = max(0, current_step - int(chunk["step"]))
        candidates.append((age, chunk_index, actions[chunk_index]))

    if not candidates:
        raise ValueError(f"No temporal ensemble candidates for target_tick={target_tick}")

    ages = np.asarray([age for age, _, _ in candidates], dtype=np.float64)
    weights = np.exp(-float(decay) * ages)
    weights = weights / np.sum(weights)

    action = {
        key: float(
            sum(
                float(candidate[key]) * float(weight)
                for weight, (_, _, candidate) in zip(weights, candidates)
            )
        )
        for key in joint_keys
    }
    return action, {
        "candidate_count": int(len(candidates)),
        "candidate_chunk_indices": [int(chunk_index) for _, chunk_index, _ in candidates],
        "candidate_ages": [int(age) for age, _, _ in candidates],
        "weights": [float(weight) for weight in weights],
    }


def write_joint_log_row(
    joint_log_file,
    step: int,
    elapsed_sec: float,
    row_type: str,
    values: Dict[str, Any],
    joint_keys: List[str],
) -> None:
    row = [str(step), f"{elapsed_sec:.6f}", row_type]
    row.extend(f"{float(values[key]):.4f}" for key in joint_keys)
    joint_log_file.write(",".join(row) + "\n")
    joint_log_file.flush()


def write_video_frame_log_row(
    video_frame_log_file,
    step: int,
    elapsed_sec: float,
    camera_key: str,
    frame_index: int,
    video_file: str,
) -> None:
    row = [
        str(step),
        f"{elapsed_sec:.6f}",
        camera_key,
        str(frame_index),
        video_file,
    ]
    video_frame_log_file.write(",".join(row) + "\n")
    video_frame_log_file.flush()


def write_policy_log_record(policy_log_file, record: Dict[str, Any]) -> None:
    policy_log_file.write(json.dumps(to_json_safe(record), ensure_ascii=True) + "\n")
    policy_log_file.flush()


def send_robot_action(
    robot: Robot,
    action: Dict[str, float],
    motion_time_ms: int,
) -> Dict[str, float]:
    if motion_time_ms <= 0:
        return robot.send_action(action)

    goal_pos = {
        key.removesuffix(".pos"): val
        for key, val in action.items()
        if key.endswith(".pos")
    }
    robot.bus.sync_write("Goal_Position", goal_pos, motion_time=motion_time_ms)
    if hasattr(robot, "_last_commanded_positions"):
        robot._last_commanded_positions.update(
            {motor: float(pos) for motor, pos in goal_pos.items()}
        )
    return {f"{motor}.pos": val for motor, val in goal_pos.items()}


def wait_after_motion_command(send_started_at: float, motion_time_ms: int) -> float:
    """Wait partway into a timed motion before reading state again."""
    return wait_after_motion_command_with_fraction(send_started_at, motion_time_ms, 0.0)


def wait_after_motion_command_with_fraction(
    send_started_at: float,
    motion_time_ms: int,
    wait_fraction: float,
) -> float:
    """Wait a configurable fraction of a timed motion before reading state again."""
    if motion_time_ms <= 0:
        return 0.0

    wait_sec = (float(motion_time_ms) / 1000.0) * float(wait_fraction)
    remaining_sec = wait_sec - (time.monotonic() - send_started_at)
    if remaining_sec > 0:
        time.sleep(remaining_sec)
    return max(0.0, remaining_sec)


def read_robot_state(robot: Robot, joint_keys: List[str]) -> Dict[str, float]:
    """Read only joint state, avoiding the camera reads in robot.get_observation()."""
    if hasattr(robot, "bus") and hasattr(robot.bus, "sync_read"):
        state = robot.bus.sync_read("Present_Position")

        config = getattr(robot, "config", None)
        last_commanded = getattr(robot, "_last_commanded_positions", {})
        if (
            getattr(config, "use_gripper_command_as_state", False)
            and "gripper" in last_commanded
        ):
            state["gripper"] = last_commanded["gripper"]

        state_with_suffix = {f"{motor}.pos": float(val) for motor, val in state.items()}
        return {key: state_with_suffix[key] for key in joint_keys}

    obs = robot.get_observation()
    return {key: float(obs[key]) for key in joint_keys}


def frame_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    if frame.shape[2] == 3:
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if frame.shape[2] == 4:
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))
    return np.ascontiguousarray(frame)


def measured_video_fps(timestamps: List[float], fallback_fps: float) -> float:
    if len(timestamps) < 2:
        return float(fallback_fps)

    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0:
        return float(fallback_fps)

    fps = float(len(timestamps) / duration)
    return float(np.clip(fps, 0.1, max(float(fallback_fps), 0.1)))


def rewrite_video_with_fps(video_path: str, output_fps: float) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"rewritten": False, "error": "failed_to_open_input"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or width <= 0 or height <= 0:
        cap.release()
        return {"rewritten": False, "error": "invalid_input_video"}

    tmp_path = f"{os.path.splitext(video_path)[0]}.realtime_tmp.mp4"
    writer = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return {"rewritten": False, "error": "failed_to_open_output"}

    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()

    if written <= 0:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return {"rewritten": False, "error": "no_frames_written"}

    os.replace(tmp_path, video_path)
    return {
        "rewritten": True,
        "fps": float(output_fps),
        "frame_count": int(written),
        "duration_sec": float(written / output_fps),
    }


def aggregate_action_chunks(
    chunks: List[List[Dict[str, float]]],
    joint_keys: List[str],
    aggregation: str,
) -> List[Dict[str, float]]:
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


def validate_control_config(cfg: "EvalConfig") -> None:
    cfg.training_action_guide = normalize_training_action_guide(cfg.training_action_guide)
    if cfg.training_action_guide not in {"off", "source_envelope", "source_replay"}:
        raise ValueError(
            "training_action_guide must be 'off', 'on', 'source_envelope', or 'source_replay'."
        )
    if cfg.timeout < 0:
        raise ValueError("timeout must be >= 0. Use timeout=0 to disable the runtime limit.")
    if cfg.training_action_guide_until_sec < 0:
        raise ValueError("training_action_guide_until_sec must be >= 0.")
    if cfg.training_action_guide_setup_sec < 0:
        raise ValueError("training_action_guide_setup_sec must be >= 0.")
    if cfg.keep_gripper_open_until_sec < 0:
        raise ValueError("keep_gripper_open_until_sec must be >= 0.")
    if cfg.motion_post_send_wait_fraction < 0 or cfg.motion_post_send_wait_fraction > 1:
        raise ValueError("motion_post_send_wait_fraction must be between 0 and 1.")
    if cfg.action_step_interval_ms < 0:
        raise ValueError("action_step_interval_ms must be >= 0. Use 0 to disable pacing.")
    if 0 < cfg.motion_time_ms < MIN_ARM_MOTION_TIME_MS:
        raise ValueError(
            f"motion_time_ms must be 0 or >= {MIN_ARM_MOTION_TIME_MS}. "
            "StarAI arm motions use 50ms accel + 50ms decel by default, so shorter "
            "timed motions can be ignored by the arm while the gripper still moves."
        )
    if cfg.action_selection_strategy != "fixed_index":
        raise ValueError("action_selection_strategy must be 'fixed_index'.")
    if cfg.execute_steps <= 1 and not cfg.allow_slow_receding_horizon:
        raise ValueError(
            "execute_steps=1 runs at policy-inference speed instead of the 30Hz training rate "
            "and is only useful for diagnostics. Use --execute_steps 4 or 8 for autonomous "
            "rollouts, or set --allow_slow_receding_horizon true if you intentionally want "
            "the slow diagnostic mode."
        )
    if cfg.execute_steps <= 1 and cfg.action_select_index != 0:
        raise ValueError(
            "action_select_index must be 0 when execute_steps <= 1. "
            "GR00T action[0] is the immediate command; later indices are future actions."
        )
    if cfg.policy_samples < 1:
        raise ValueError("policy_samples must be >= 1.")
    if cfg.temporal_ensemble_decay < 0:
        raise ValueError("temporal_ensemble_decay must be >= 0.")
    if cfg.temporal_ensemble_window < 1:
        raise ValueError("temporal_ensemble_window must be >= 1.")
    if cfg.execute_steps <= 1 and cfg.policy_samples != 1:
        raise ValueError(
            "policy_samples must be 1 when execute_steps <= 1. "
            "Multiple samples slow the real-robot loop far below the 30Hz training data."
        )


def recursive_add_extra_dim(obs: Dict) -> Dict:
    """
    Recursively add an extra dim to arrays or scalars.

    GR00T Policy Server expects:
        obs: (batch=1, time=1, ...)
    Calling this function twice achieves that.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]  # scalar → [scalar]
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


class StarAIAdapter:
    """
    Adapter between:
        • Raw StarAI robot observation dictionary
        • GR00T VLA input format (matches starai_single_arm_config.py)
        • GR00T action chunk → robot joint commands

    Layout:
        state.single_arm: (6,)  Motor_0..Motor_5
        state.gripper:    (1,)  gripper
        video.{side, rear, onhand}
        language.<server modality language key>
    """

    def __init__(
        self,
        policy_client: PolicyClient,
        camera_keys: List[str],
        language_key: str = "annotation.human.task_description",
    ):
        self.policy = policy_client

        # StarAI joint ordering used for BOTH training + robot execution.
        # First 6 entries form ``single_arm``; last entry is ``gripper``.
        self.robot_state_keys = [
            "Motor_0.pos",
            "Motor_1.pos",
            "Motor_2.pos",
            "Motor_3.pos",
            "Motor_4.pos",
            "Motor_5.pos",
            "gripper.pos",
        ]
        self.camera_keys = camera_keys
        self.language_key = language_key
        self.last_policy_info: Dict[str, Any] = {}

    def obs_to_policy_inputs(self, obs: Dict[str, Any], batch_size: int = 1) -> Dict:
        model_obs: Dict[str, Any] = {}

        # (1) Cameras
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}

        # (2) Arm + gripper state (7ch: 6 arm motors + 1 gripper)
        state = np.array([obs[k] for k in self.robot_state_keys], dtype=np.float32)
        model_obs["state"] = {
            "single_arm": state[:6],  # (6,)
            "gripper": state[6:7],    # (1,)
        }

        # (3) Language
        model_obs["language"] = {self.language_key: obs["lang"]}

        # (4) Add (B=1, T=1) dims
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_repeat_batch(model_obs, batch_size)
        return model_obs

    def decode_action_chunk(self, chunk: Dict, t: int, batch_index: int = 0) -> Dict[str, float]:
        """
        chunk["single_arm"]: (B, T, 6)
        chunk["gripper"]:    (B, T, 1)
        """
        single_arm = chunk["single_arm"][batch_index][t]  # (6,)
        gripper = chunk["gripper"][batch_index][t]        # (1,)

        full = np.concatenate([single_arm, gripper], axis=0)  # (7,)
        return {joint_name: float(full[i]) for i, joint_name in enumerate(self.robot_state_keys)}

    def get_action(
        self,
        obs: Dict,
        num_samples: int = 1,
        aggregation: str = "median",
    ) -> List[Dict[str, float]]:
        adapter_tic = time.perf_counter()
        num_samples = max(1, int(num_samples))
        chunks, info, adapter_profile = self._get_batched_action_chunks(obs, num_samples)

        tic = time.perf_counter()
        actions = aggregate_action_chunks(chunks, self.robot_state_keys, aggregation)
        adapter_profile["adapter_aggregate_action_chunks_sec"] = time.perf_counter() - tic
        adapter_profile["adapter_total_sec"] = time.perf_counter() - adapter_tic

        self.last_policy_info = dict(info) if isinstance(info, dict) else {"raw_info": info}
        self.last_policy_info["adapter_profile"] = adapter_profile
        return actions

    def _get_batched_action_chunks(
        self,
        obs: Dict,
        batch_size: int,
    ) -> Tuple[List[List[Dict[str, float]]], Dict[str, Any], Dict[str, float]]:
        adapter_profile: Dict[str, float] = {}

        tic = time.perf_counter()
        model_input = self.obs_to_policy_inputs(obs, batch_size=batch_size)
        adapter_profile["adapter_obs_to_policy_inputs_sec"] = time.perf_counter() - tic

        tic = time.perf_counter()
        action_chunk, info = self.policy.get_action(model_input)
        adapter_profile["adapter_policy_client_get_action_sec"] = time.perf_counter() - tic

        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) → T
        actual_batch_size = action_chunk[any_key].shape[0]

        tic = time.perf_counter()
        chunks = [
            [self.decode_action_chunk(action_chunk, t, batch_index=b) for t in range(horizon)]
            for b in range(actual_batch_size)
        ]
        adapter_profile["adapter_decode_action_chunk_sec"] = time.perf_counter() - tic
        adapter_profile["adapter_action_horizon"] = float(horizon)
        adapter_profile["adapter_action_batch_size"] = float(actual_batch_size)
        adapter_profile["adapter_requested_batch_size"] = float(batch_size)
        return chunks, info, adapter_profile


# =============================================================================
# Evaluation Config
# =============================================================================


@dataclass
class EvalConfig:
    """CLI configuration for StarAI real-robot policy evaluation."""

    robot: RobotConfig
    policy_host: str = "localhost"
    policy_port: int = 5555
    action_horizon: int = 16
    # Must match the task string used during finetuning
    # (tasks.jsonl of the training dataset).
    lang_instruction: str = "pick_and_place"
    language_key: str = "annotation.human.task_description"
    camera_keys: List[str] = field(default_factory=lambda: ["side", "rear", "onhand"])
    play_sounds: bool = False
    timeout: int = 60
    control_hz: float = 30.0
    clamp_action_step: bool = False
    execute_steps: int = 4
    action_select_index: int = 0
    action_selection_strategy: str = "fixed_index"
    approach_motor1_until: float = 20.0
    allow_slow_receding_horizon: bool = False
    policy_samples: int = 1
    action_aggregation: str = "median"
    temporal_ensemble: bool = False
    temporal_ensemble_decay: float = 0.4
    temporal_ensemble_window: int = 8
    motion_time_ms: int = 250
    motion_post_send_wait_fraction: float = 0.0
    action_step_interval_ms: int = 0
    keep_gripper_open_until_motor1: float = 20.0
    gripper_open_value: float = 80.0
    keep_gripper_open_until_sec: float = 0.0
    training_action_guide: str = "off"
    training_action_guide_until_sec: float = 60.0
    training_action_guide_setup_sec: float = 0.0
    training_dataset_path: str | None = None
    initial_pose_strategy: str = "first_episode"


# =============================================================================
# Main Eval Loop
# =============================================================================


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    cfg.training_action_guide = normalize_training_action_guide(cfg.training_action_guide)
    logging.info(pformat(asdict(cfg)))
    validate_control_config(cfg)

    resolved_training_dataset_path = resolve_training_dataset_path(cfg.training_dataset_path)
    initial_pose = resolve_initial_pose(resolved_training_dataset_path, cfg.initial_pose_strategy)
    TRAINING_INITIAL_POSE.clear()
    TRAINING_INITIAL_POSE.update(initial_pose)
    logging.info("Using initial pose: %s", TRAINING_INITIAL_POSE)
    training_action_guide = (
        load_training_action_guide(resolved_training_dataset_path)
        if resolved_training_dataset_path and cfg.training_action_guide != "off"
        else None
    )
    if training_action_guide is not None:
        logging.info("Using training action guide: %s", training_action_guide.summary())
    if cfg.training_action_guide != "off":
        logging.warning(
            "training_action_guide=%s modifies learned policy actions. "
            "Use training_action_guide=off to follow the learned GR00T policy trajectory.",
            cfg.training_action_guide,
        )

    # -------------------------------------------------------------------------
    # 1. Initialize Robot Hardware
    #    NOTE: StaraiCello.connect() will call the (patched)
    #    move_to_initial_position(), driving the arm to the training-data
    #    episode-start pose defined in TRAINING_INITIAL_POSE.
    # -------------------------------------------------------------------------
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    missing = [k for k in cfg.camera_keys if k not in robot.cameras]
    if missing:
        raise RuntimeError(
            f"Camera keys {missing} not found on robot. Configure cameras "
            f"under --robot.cameras with names matching {cfg.camera_keys}."
        )

    log_say("Initializing robot (StarAI Cello)", cfg.play_sounds, blocking=True)

    # -------------------------------------------------------------------------
    # 2. Initialize Policy Wrapper + Client
    # -------------------------------------------------------------------------
    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy_modality_config = get_policy_modality_config(policy_client)
    runtime_language_key = resolve_policy_language_key(cfg.language_key, policy_modality_config)
    policy = StarAIAdapter(
        policy_client,
        camera_keys=cfg.camera_keys,
        language_key=runtime_language_key,
    )

    log_say(
        f'Policy ready with instruction: "{cfg.lang_instruction}"',
        cfg.play_sounds,
        blocking=True,
    )

    # -------------------------------------------------------------------------
    # 3. Set up logging (video + joint values)
    # -------------------------------------------------------------------------
    log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "logs",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(log_dir, exist_ok=True)

    session_meta_path = os.path.join(log_dir, "session_meta.json")
    session_meta = {
        "created_at_unix": time.time(),
        "runtime": "eval_starai",
        "policy_host": cfg.policy_host,
        "policy_port": cfg.policy_port,
        "timeout_sec": cfg.timeout,
        "timeout_enabled": cfg.timeout > 0,
        "lang_instruction": cfg.lang_instruction,
        "language_key": runtime_language_key,
        "requested_language_key": cfg.language_key,
        "camera_keys": cfg.camera_keys,
        "action_source": action_source_label(cfg.training_action_guide),
        "action_horizon": cfg.action_horizon,
        "execute_steps": cfg.execute_steps,
        "action_select_index": cfg.action_select_index,
        "action_selection_strategy": cfg.action_selection_strategy,
        "approach_motor1_until": cfg.approach_motor1_until,
        "allow_slow_receding_horizon": cfg.allow_slow_receding_horizon,
        "policy_samples": cfg.policy_samples,
        "action_aggregation": cfg.action_aggregation,
        "temporal_ensemble": cfg.temporal_ensemble,
        "temporal_ensemble_decay": cfg.temporal_ensemble_decay,
        "temporal_ensemble_window": cfg.temporal_ensemble_window,
        "control_hz": cfg.control_hz,
        "policy_profile_logging": True,
        "motion_time_ms": cfg.motion_time_ms,
        "motion_post_send_wait_fraction": cfg.motion_post_send_wait_fraction,
        "action_step_interval_ms": cfg.action_step_interval_ms,
        "action_step_interval_target_hz": (
            float(1000.0 / cfg.action_step_interval_ms)
            if cfg.action_step_interval_ms > 0
            else None
        ),
        "keep_gripper_open_until_motor1": cfg.keep_gripper_open_until_motor1,
        "gripper_open_value": cfg.gripper_open_value,
        "keep_gripper_open_until_sec": cfg.keep_gripper_open_until_sec,
        "training_action_guide": cfg.training_action_guide,
        "training_action_guide_until_sec": cfg.training_action_guide_until_sec,
        "training_action_guide_effective_until_sec": effective_training_action_guide_until_sec(
            training_action_guide,
            cfg.training_action_guide_until_sec,
        ),
        "training_action_guide_setup_sec": cfg.training_action_guide_setup_sec,
        "training_action_guide_tolerance": DEFAULT_TRAINING_GUIDE_TOLERANCE,
        "training_action_guide_source": (
            training_action_guide.summary() if training_action_guide is not None else None
        ),
        "clamp_action_step": cfg.clamp_action_step,
        "training_dataset_path": cfg.training_dataset_path,
        "resolved_training_dataset_path": resolved_training_dataset_path,
        "initial_pose_strategy": cfg.initial_pose_strategy,
        "initial_pose": initial_pose,
        "max_action_step": MAX_ACTION_STEP,
        "wait_for_motion_completion": cfg.motion_post_send_wait_fraction >= 1.0,
        "clamp_reference": "pre_action_state" if cfg.clamp_action_step else "disabled",
        "policy_modality_config": policy_modality_config,
        "robot": str(cfg.robot),
    }
    with open(session_meta_path, "w", encoding="utf-8") as session_meta_file:
        json.dump(session_meta, session_meta_file, ensure_ascii=True, indent=2)

    video_writers: Dict[str, cv2.VideoWriter] = {}
    video_paths: Dict[str, str] = {}
    video_frame_counts: Dict[str, int] = {}
    video_frame_timestamps: Dict[str, List[float]] = {}
    video_frame_log_path = os.path.join(log_dir, "video_frame_log.csv")
    video_frame_log_file = open(video_frame_log_path, "w")
    video_frame_log_file.write("step,elapsed_sec,camera_key,frame_index,video_file\n")

    joint_log_path = os.path.join(log_dir, "joint_log.csv")
    joint_log_file = open(joint_log_path, "w")

    header = "step,elapsed_sec,type," + ",".join(policy.robot_state_keys) + "\n"
    joint_log_file.write(header)

    policy_log_path = os.path.join(log_dir, "policy_log.jsonl")
    policy_log_file = open(policy_log_path, "w")
    session_meta["log_files"] = {
        "joint_log_csv": os.path.basename(joint_log_path),
        "policy_log_jsonl": os.path.basename(policy_log_path),
        "video_frame_log_csv": os.path.basename(video_frame_log_path),
        "camera_videos": {cam_key: f"{cam_key}.mp4" for cam_key in cfg.camera_keys},
    }
    with open(session_meta_path, "w", encoding="utf-8") as session_meta_file:
        json.dump(session_meta, session_meta_file, ensure_ascii=True, indent=2)

    step = 0
    start_time = time.monotonic()
    timeout_deadline = start_time + cfg.timeout if cfg.timeout > 0 else None
    stop_reason = None
    state_elapsed_samples: List[float] = []
    action_send_elapsed_samples: List[float] = []
    last_action_send_started_at: float | None = None
    action_tick = 0
    temporal_ensemble_chunks: List[Dict[str, Any]] = []

    logging.info(f"Logging to: {log_dir}")
    if timeout_deadline is not None:
        logging.info("Runtime timeout: %s seconds", cfg.timeout)
    else:
        logging.info("Runtime timeout disabled.")

    # -------------------------------------------------------------------------
    # 4. Main real-time control loop
    # -------------------------------------------------------------------------
    try:
        while True:
            now = time.monotonic()
            if timeout_deadline is not None and now >= timeout_deadline:
                stop_reason = "timeout"
                logging.info(
                    "Timeout reached after %.3fs (limit %ss). Saving logs...",
                    now - start_time,
                    cfg.timeout,
                )
                break

            obs = robot.get_observation()
            obs["lang"] = cfg.lang_instruction

            # --- Log camera frames ---
            for cam_key in cfg.camera_keys:
                frame = obs[cam_key]
                if cam_key not in video_writers:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_path = os.path.join(log_dir, f"{cam_key}.mp4")
                    writer = cv2.VideoWriter(video_path, fourcc, cfg.control_hz, (w, h))
                    if not writer.isOpened():
                        logging.warning("Failed to open video writer: %s", video_path)
                        continue
                    video_writers[cam_key] = writer
                    video_paths[cam_key] = video_path
                    video_frame_counts[cam_key] = 0
                    video_frame_timestamps[cam_key] = []
                frame_bgr = frame_to_bgr(frame)
                video_writers[cam_key].write(frame_bgr)
                frame_index = video_frame_counts[cam_key]
                video_frame_counts[cam_key] += 1
                frame_elapsed_sec = time.monotonic() - start_time
                video_frame_timestamps[cam_key].append(frame_elapsed_sec)
                write_video_frame_log_row(
                    video_frame_log_file,
                    step,
                    frame_elapsed_sec,
                    cam_key,
                    frame_index,
                    os.path.basename(video_paths[cam_key]),
                )

            # --- Log observed state ---
            state_dbg = {k: obs[k] for k in policy.robot_state_keys}
            state_elapsed_sec = time.monotonic() - start_time
            state_elapsed_samples.append(state_elapsed_sec)
            print(f"state: {state_dbg}")
            write_joint_log_row(
                joint_log_file,
                step,
                state_elapsed_sec,
                "state",
                state_dbg,
                policy.robot_state_keys,
            )

            inference_tic = time.monotonic()
            actions = policy.get_action(
                obs,
                num_samples=cfg.policy_samples,
                aggregation=cfg.action_aggregation,
            )
            inference_sec = time.monotonic() - inference_tic
            policy_profile = getattr(policy, "last_policy_info", {})
            control_period = 1.0 / cfg.control_hz
            if inference_sec > control_period and step % 30 == 0:
                logging.warning(
                    "Policy inference took %.3fs, so this loop cannot actually run at %.1fHz.",
                    inference_sec,
                    cfg.control_hz,
                )
            reference_state = {k: float(obs[k]) for k in policy.robot_state_keys}
            raw_actions = actions[: cfg.action_horizon]
            if cfg.temporal_ensemble:
                temporal_ensemble_chunks.append(
                    {
                        "step": step,
                        "base_tick": action_tick,
                        "actions": raw_actions,
                    }
                )
                temporal_ensemble_chunks = temporal_ensemble_chunks[-cfg.temporal_ensemble_window :]
            else:
                temporal_ensemble_chunks = []
            policy_log_record = {
                "step": step,
                "state_elapsed_sec": state_elapsed_sec,
                "base_action_tick": action_tick,
                "inference_sec": inference_sec,
                "policy_profile": policy_profile,
                "language": cfg.lang_instruction,
                "language_key": runtime_language_key,
                "state": reference_state,
                "raw_actions": raw_actions,
                "executed_actions": [],
            }

            for i, raw_action in enumerate(raw_actions):
                write_joint_log_row(
                    joint_log_file,
                    step,
                    time.monotonic() - start_time,
                    f"raw_action_{i}",
                    raw_action,
                    policy.robot_state_keys,
                )

            if cfg.execute_steps <= 1:
                selected_index = select_receding_horizon_action_index(
                    raw_actions,
                    reference_state,
                    cfg.action_selection_strategy,
                    cfg.action_select_index,
                    cfg.approach_motor1_until,
                )
                actions_to_execute = [(selected_index, raw_actions[selected_index])]
            else:
                steps_to_execute = min(cfg.execute_steps, len(raw_actions))
                actions_to_execute = list(enumerate(raw_actions[:steps_to_execute]))

            for i, raw_action in actions_to_execute:
                now = time.monotonic()
                if timeout_deadline is not None and now >= timeout_deadline:
                    stop_reason = "timeout"
                    logging.info(
                        "Timeout reached after %.3fs (limit %ss) before action[%d]. Saving logs...",
                        now - start_time,
                        cfg.timeout,
                        i,
                    )
                    break

                action_step_interval_wait_sec = 0.0
                if cfg.action_step_interval_ms > 0 and last_action_send_started_at is not None:
                    next_send_at = last_action_send_started_at + (
                        float(cfg.action_step_interval_ms) / 1000.0
                    )
                    remaining_sec = next_send_at - time.monotonic()
                    if remaining_sec > 0:
                        time.sleep(remaining_sec)
                        action_step_interval_wait_sec = remaining_sec

                    now = time.monotonic()
                    if timeout_deadline is not None and now >= timeout_deadline:
                        stop_reason = "timeout"
                        logging.info(
                            "Timeout reached after %.3fs (limit %ss) before paced action[%d]. Saving logs...",
                            now - start_time,
                            cfg.timeout,
                            i,
                        )
                        break

                tic = time.monotonic()
                target_tick = action_tick
                if cfg.temporal_ensemble:
                    policy_action, temporal_info = temporal_ensemble_action(
                        temporal_ensemble_chunks,
                        target_tick,
                        step,
                        policy.robot_state_keys,
                        cfg.temporal_ensemble_decay,
                    )
                    if action_was_clamped(raw_action, policy_action):
                        write_joint_log_row(
                            joint_log_file,
                            step,
                            time.monotonic() - start_time,
                            f"ensembled_action_{i}",
                            policy_action,
                            policy.robot_state_keys,
                        )
                else:
                    policy_action = raw_action
                    temporal_info = {
                        "candidate_count": 1,
                        "candidate_chunk_indices": [i],
                        "candidate_ages": [0],
                        "weights": [1.0],
                    }
                pre_action_state = read_robot_state(robot, policy.robot_state_keys)
                pre_action_elapsed_sec = time.monotonic() - start_time
                write_joint_log_row(
                    joint_log_file,
                    step,
                    pre_action_elapsed_sec,
                    f"pre_action_state_{i}",
                    pre_action_state,
                    policy.robot_state_keys,
                )
                action_reference_elapsed_sec = pre_action_elapsed_sec

                guided_action = apply_training_action_guide(
                    policy_action,
                    action_reference_elapsed_sec,
                    training_action_guide,
                    cfg.training_action_guide,
                    cfg.training_action_guide_until_sec,
                    cfg.training_action_guide_setup_sec,
                )
                safe_raw_action = apply_task_safety_overrides(
                    guided_action,
                    pre_action_state,
                    cfg.keep_gripper_open_until_motor1,
                    cfg.gripper_open_value,
                    action_reference_elapsed_sec,
                    cfg.keep_gripper_open_until_sec,
                )
                postprocess_changed = action_was_clamped(policy_action, safe_raw_action)
                if postprocess_changed:
                    write_joint_log_row(
                        joint_log_file,
                        step,
                        time.monotonic() - start_time,
                        f"guided_action_{i}",
                        safe_raw_action,
                        policy.robot_state_keys,
                    )
                sent_action = (
                    clamp_action_to_step(safe_raw_action, pre_action_state, MAX_ACTION_STEP)
                    if cfg.clamp_action_step
                    else safe_raw_action
                )
                print(f"raw_action[{i}]: {raw_action}")
                if cfg.temporal_ensemble:
                    print(f"ensembled_action[{i}]: {policy_action}")
                print(f"sent_action[{i}]: {sent_action}")
                step_clamped = action_was_clamped(safe_raw_action, sent_action)
                was_clamped = action_was_clamped(raw_action, sent_action)
                if step_clamped:
                    logging.warning("Clamped action[%d] before sending to robot.", i)

                # --- Log command sent to the robot ---
                write_joint_log_row(
                    joint_log_file,
                    step,
                    time.monotonic() - start_time,
                    f"sent_action_{i}",
                    sent_action,
                    policy.robot_state_keys,
                )

                send_started_at = time.monotonic()
                last_action_send_started_at = send_started_at
                action_send_elapsed_sec = send_started_at - start_time
                action_send_elapsed_samples.append(action_send_elapsed_sec)
                performed_action = send_robot_action(robot, sent_action, cfg.motion_time_ms)
                write_joint_log_row(
                    joint_log_file,
                    step,
                    time.monotonic() - start_time,
                    f"performed_action_{i}",
                    performed_action,
                    policy.robot_state_keys,
                )
                motion_wait_sec = wait_after_motion_command_with_fraction(
                    send_started_at,
                    cfg.motion_time_ms,
                    cfg.motion_post_send_wait_fraction,
                )
                post_action_state = read_robot_state(robot, policy.robot_state_keys)
                post_action_elapsed_sec = time.monotonic() - start_time
                write_joint_log_row(
                    joint_log_file,
                    step,
                    post_action_elapsed_sec,
                    f"post_action_state_{i}",
                    post_action_state,
                    policy.robot_state_keys,
                )
                tracking_error = {
                    key: float(post_action_state[key]) - float(sent_action[key])
                    for key in policy.robot_state_keys
                }
                policy_log_record["executed_actions"].append(
                    {
                        "chunk_index": i,
                        "target_tick": target_tick,
                        "raw_action": raw_action,
                        "temporal_ensemble_action": policy_action,
                        "temporal_ensemble_info": temporal_info,
                        "pre_action_state": pre_action_state,
                        "pre_action_state_elapsed_sec": pre_action_elapsed_sec,
                        "action_reference_elapsed_sec": action_reference_elapsed_sec,
                        "guided_action": safe_raw_action,
                        "sent_action": sent_action,
                        "performed_action": performed_action,
                        "action_send_elapsed_sec": action_send_elapsed_sec,
                        "action_step_interval_wait_sec": action_step_interval_wait_sec,
                        "motion_wait_sec": motion_wait_sec,
                        "post_action_state": post_action_state,
                        "post_action_state_elapsed_sec": post_action_elapsed_sec,
                        "tracking_error": tracking_error,
                        "clamped": was_clamped,
                        "step_clamped": step_clamped,
                        "postprocess_changed": postprocess_changed,
                        "elapsed_sec": post_action_elapsed_sec,
                    }
                )
                reference_state = post_action_state
                action_tick += 1
                toc = time.monotonic()
                if toc - tic < control_period:
                    time.sleep(control_period - (toc - tic))

            write_policy_log_record(policy_log_file, policy_log_record)
            step += 1
            if stop_reason is not None:
                break

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        logging.info("Interrupted. Saving logs...")
    except Exception:
        stop_reason = "exception"
        raise
    finally:
        joint_log_file.close()
        video_frame_log_file.close()
        policy_log_file.close()
        for writer in video_writers.values():
            writer.release()
        video_rewrite_results: Dict[str, Dict[str, Any]] = {}
        for cam_key, video_path in video_paths.items():
            timestamps = video_frame_timestamps.get(cam_key, [])
            if not timestamps:
                continue
            output_fps = measured_video_fps(timestamps, cfg.control_hz)
            result = rewrite_video_with_fps(video_path, output_fps)
            result["timestamp_span_sec"] = (
                float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
            )
            result["logged_frame_count"] = int(len(timestamps))
            video_rewrite_results[cam_key] = result
            if result.get("rewritten"):
                logging.info(
                    "Rewrote %s at %.3f fps for realtime playback.",
                    os.path.basename(video_path),
                    output_fps,
                )
            else:
                logging.warning(
                    "Failed to rewrite %s for realtime playback: %s",
                    os.path.basename(video_path),
                    result.get("error"),
                )
        if len(state_elapsed_samples) > 1:
            intervals = np.diff(np.asarray(state_elapsed_samples, dtype=np.float64))
            session_meta.update(
                {
                    "actual_state_loop_hz_mean": float(1.0 / np.mean(intervals)),
                    "actual_state_loop_hz_median": float(1.0 / np.median(intervals)),
                    "actual_state_loop_dt_mean_sec": float(np.mean(intervals)),
                    "actual_state_loop_dt_median_sec": float(np.median(intervals)),
                    "num_state_samples": int(len(state_elapsed_samples)),
                    "end_elapsed_sec": float(state_elapsed_samples[-1]),
                }
            )
        if len(action_send_elapsed_samples) > 1:
            intervals = np.diff(np.asarray(action_send_elapsed_samples, dtype=np.float64))
            session_meta.update(
                {
                    "actual_action_send_hz_mean": float(1.0 / np.mean(intervals)),
                    "actual_action_send_hz_median": float(1.0 / np.median(intervals)),
                    "actual_action_send_dt_mean_sec": float(np.mean(intervals)),
                    "actual_action_send_dt_median_sec": float(np.median(intervals)),
                    "num_action_sends": int(len(action_send_elapsed_samples)),
                    "last_action_send_elapsed_sec": float(action_send_elapsed_samples[-1]),
                }
            )
        session_meta["stop_reason"] = stop_reason or "unknown"
        session_meta["wall_elapsed_sec"] = float(time.monotonic() - start_time)
        if video_rewrite_results:
            session_meta["video_rewrite_results"] = video_rewrite_results
        with open(session_meta_path, "w", encoding="utf-8") as session_meta_file:
            json.dump(session_meta, session_meta_file, ensure_ascii=True, indent=2)
        logging.info(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    eval()
