"""Gr00t Policy implementation for inference.

This module provides the core policy classes for running Gr00t models:
- Gr00tPolicy: Base policy class for model inference
- Gr00tSimPolicyWrapper: Wrapper for compatibility with existing Gr00t simulation environments
"""

from pathlib import Path
import inspect
import os
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData

from .policy import BasePolicy, PolicyWrapper


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


def _module_device(module: torch.nn.Module) -> torch.device | None:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def _sync_cuda_device(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype

    normalized = dtype.lower().replace("-", "_")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported compute_dtype={dtype!r}. "
            "Use one of: bfloat16, float16, float32."
        )
    return aliases[normalized]


def _parse_tensorrt_targets(targets: str) -> set[str]:
    normalized_targets = {
        item.strip().lower().replace("-", "_")
        for item in targets.split(",")
        if item.strip()
    }
    if not normalized_targets or normalized_targets == {"none"}:
        return set()
    if "all" in normalized_targets:
        normalized_targets.remove("all")
        normalized_targets.update({"action_head", "backbone"})

    aliases = {
        "action": "action_head",
        "action_head_model": "action_head",
        "dit": "action_head",
        "diffusion": "action_head",
        "vlm": "backbone",
    }
    parsed = {aliases.get(target, target) for target in normalized_targets}
    supported = {"action_head", "backbone"}
    unknown = parsed - supported
    if unknown:
        raise ValueError(
            f"Unsupported tensorrt_target={sorted(unknown)}. "
            "Use one of: none, action_head, backbone, all."
        )
    return parsed


def _filter_tensorrt_options(options: dict[str, Any]) -> dict[str, Any]:
    """Keep Torch-TensorRT compile options compatible across installed versions."""
    try:
        import torch_tensorrt

        settings_cls = torch_tensorrt.dynamo.CompilationSettings
        supported = set(inspect.signature(settings_cls).parameters)
    except Exception:
        return options

    filtered = dict(options)
    if "truncate_long_and_double" in filtered and "truncate_long_and_double" not in supported:
        value = filtered.pop("truncate_long_and_double")
        if "truncate_double" in supported:
            filtered["truncate_double"] = value
    return {key: value for key, value in filtered.items() if key in supported}


def _compile_module_with_tensorrt(
    module: torch.nn.Module,
    *,
    name: str,
    compute_dtype: torch.dtype,
    cache_dir: str,
    min_block_size: int,
    require_full_compilation: bool,
) -> torch.nn.Module:
    import torch_tensorrt  # noqa: F401  # Registers the torch_tensorrt compile backend.

    os.makedirs(cache_dir, exist_ok=True)
    precision = torch.float16 if compute_dtype == torch.bfloat16 else compute_dtype
    enabled_precisions = {precision}
    if precision == torch.float16:
        # Let Torch-TensorRT keep unsupported precision-sensitive fragments in fp32.
        enabled_precisions.add(torch.float32)

    options = _filter_tensorrt_options(
        {
            "enabled_precisions": enabled_precisions,
            "truncate_long_and_double": True,
            "min_block_size": min_block_size,
            "require_full_compilation": require_full_compilation,
            "cache_built_engines": True,
            "reuse_cached_engines": True,
            "timing_cache_path": os.path.join(cache_dir, f"{name}_timing_cache.bin"),
            "runtime_cache_path": os.path.join(cache_dir, f"{name}_runtime_cache.bin"),
        }
    )
    return torch.compile(
        module,
        backend="torch_tensorrt",
        dynamic=False,
        fullgraph=require_full_compilation,
        options=options,
    )


class Gr00tPolicy(BasePolicy):
    """Core policy class for Gr00t model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        compute_dtype: str | torch.dtype = "bfloat16",
        profile_model_detail: bool = True,
        tensorrt_target: str = "none",
        tensorrt_cache_dir: str = "/tmp/gr00t_tensorrt_cache",
        tensorrt_min_block_size: int = 5,
        tensorrt_require_full_compilation: bool = False,
        tensorrt_strict: bool = True,
    ):
        """Initialize the Gr00t Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)
            compute_dtype: Floating point dtype used for model weights and floating inputs.
            profile_model_detail: Whether to split model inference profile into sub-stages.
            tensorrt_target: Comma-separated TensorRT targets: none, action_head, backbone, all.
            tensorrt_cache_dir: Directory for Torch-TensorRT timing/runtime caches.
            tensorrt_min_block_size: Minimum block size for TensorRT graph partitioning.
            tensorrt_require_full_compilation: Require full TensorRT compilation for target graphs.
            tensorrt_strict: Raise if TensorRT setup fails; otherwise continue with PyTorch modules.
        """
        # Import this to register all models.
        import gr00t.model  # noqa: F401

        super().__init__(strict=strict)
        model_dir = Path(model_path)
        self.compute_dtype = resolve_torch_dtype(compute_dtype)
        self.profile_model_detail = profile_model_detail
        self.tensorrt_targets = _parse_tensorrt_targets(tensorrt_target)
        self.tensorrt_status: dict[str, Any] = {
            "enabled": bool(self.tensorrt_targets),
            "requested_targets": sorted(self.tensorrt_targets),
            "compiled_targets": [],
            "failed_targets": {},
        }

        # Load the pretrained model and move to the requested inference precision.
        model = AutoModel.from_pretrained(model_dir, low_cpu_mem_usage=True)
        model.eval()  # Set model to evaluation mode
        model.to(device=device, dtype=self.compute_dtype)
        self.model = model
        self._configure_tensorrt(
            cache_dir=tensorrt_cache_dir,
            min_block_size=tensorrt_min_block_size,
            require_full_compilation=tensorrt_require_full_compilation,
            strict=tensorrt_strict,
        )

        # Load the processor for input/output transformation
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(model_dir)
        self.processor.eval()

        # Store embodiment-specific configurations
        self.embodiment_tag = embodiment_tag
        self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
        self.collate_fn = self.processor.collator

        # Extract and validate language configuration
        # Currently only supports single language input per timestep
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_keys) == 1, "Only one language key is supported"
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        self.language_key = language_keys[0]

    def _configure_tensorrt(
        self,
        *,
        cache_dir: str,
        min_block_size: int,
        require_full_compilation: bool,
        strict: bool,
    ) -> None:
        if not self.tensorrt_targets:
            return

        model_device = _module_device(self.model)
        if model_device is None or model_device.type != "cuda":
            message = "TensorRT acceleration requires a CUDA model device."
            if strict:
                raise RuntimeError(message)
            self.tensorrt_status["failed_targets"]["all"] = message
            print(f"[TensorRT] {message} Continuing without TensorRT.")
            return

        if self.compute_dtype == torch.bfloat16:
            message = (
                "TensorRT acceleration is configured for float16/float32 inference here. "
                "Rerun with --compute-dtype float16 when using --tensorrt-target."
            )
            if strict:
                raise RuntimeError(message)
            self.tensorrt_status["failed_targets"]["all"] = message
            print(f"[TensorRT] {message} Continuing without TensorRT.")
            return

        try:
            import torch_tensorrt  # noqa: F401
        except ImportError as exc:
            message = (
                "torch_tensorrt is not installed. Install it in the GR00T venv, e.g. "
                "python -m pip install torch-tensorrt "
                "--extra-index-url https://download.pytorch.org/whl/cu128"
            )
            if strict:
                raise RuntimeError(message) from exc
            self.tensorrt_status["failed_targets"]["all"] = message
            print(f"[TensorRT] {message} Continuing without TensorRT.")
            return

        if "action_head" in self.tensorrt_targets:
            try:
                self.model.action_head.model = _compile_module_with_tensorrt(
                    self.model.action_head.model,
                    name="action_head",
                    compute_dtype=self.compute_dtype,
                    cache_dir=cache_dir,
                    min_block_size=min_block_size,
                    require_full_compilation=require_full_compilation,
                )
                self.tensorrt_status["compiled_targets"].append("action_head")
                print("[TensorRT] Compiled action_head diffusion model via torch_tensorrt.")
            except Exception as exc:
                message = str(exc)
                if strict:
                    raise RuntimeError(
                        f"Failed to compile action_head with TensorRT: {message}"
                    ) from exc
                self.tensorrt_status["failed_targets"]["action_head"] = message
                print(f"[TensorRT] Failed to compile action_head: {message}")

        if "backbone" in self.tensorrt_targets:
            try:
                self.model.backbone = _compile_module_with_tensorrt(
                    self.model.backbone,
                    name="backbone",
                    compute_dtype=self.compute_dtype,
                    cache_dir=cache_dir,
                    min_block_size=min_block_size,
                    require_full_compilation=require_full_compilation,
                )
                self.tensorrt_status["compiled_targets"].append("backbone")
                print("[TensorRT] Compiled backbone via torch_tensorrt.")
            except Exception as exc:
                message = str(exc)
                if strict:
                    raise RuntimeError(f"Failed to compile backbone with TensorRT: {message}") from exc
                self.tensorrt_status["failed_targets"]["backbone"] = message
                print(f"[TensorRT] Failed to compile backbone: {message}")

    def _get_model_action_with_profile(
        self, collated_inputs: dict[str, Any], options: dict[str, Any] | None
    ) -> tuple[Any, dict[str, Any]]:
        """Run model inference, using a detailed path when the loaded model supports it."""
        model_device = _module_device(self.model)
        profile: dict[str, Any] = {
            "policy_model_detail_profile_enabled": self.profile_model_detail,
            "policy_model_detail_profile_supported": False,
        }

        supports_detail = (
            self.profile_model_detail
            and "inputs" in collated_inputs
            and hasattr(self.model, "prepare_input")
            and hasattr(self.model, "backbone")
            and hasattr(self.model, "action_head")
        )
        if not supports_detail:
            tic = time.perf_counter()
            with torch.inference_mode():
                model_pred = self.model.get_action(**collated_inputs)
            _sync_cuda_device(model_device)
            profile["policy_model_forward_sec"] = time.perf_counter() - tic
            return model_pred, profile

        profile["policy_model_detail_profile_supported"] = True
        total_tic = time.perf_counter()

        tic = time.perf_counter()
        with torch.inference_mode():
            backbone_inputs, action_inputs = self.model.prepare_input(collated_inputs["inputs"])
        _sync_cuda_device(model_device)
        profile["policy_model_prepare_input_sec"] = time.perf_counter() - tic

        tic = time.perf_counter()
        with torch.inference_mode():
            backbone_outputs = self.model.backbone(backbone_inputs)
        _sync_cuda_device(model_device)
        profile["policy_model_backbone_sec"] = time.perf_counter() - tic

        tic = time.perf_counter()
        with torch.inference_mode():
            if hasattr(self.model.action_head, "get_action_profiled"):
                model_pred, action_head_profile = self.model.action_head.get_action_profiled(
                    backbone_outputs,
                    action_inputs,
                    options=options,
                )
            else:
                try:
                    model_pred = self.model.action_head.get_action(
                        backbone_outputs,
                        action_inputs,
                        options,
                    )
                except TypeError:
                    model_pred = self.model.action_head.get_action(
                        backbone_outputs,
                        action_inputs,
                    )
                action_head_profile = {}
        _sync_cuda_device(model_device)
        profile["policy_model_action_head_sec"] = time.perf_counter() - tic
        profile.update(action_head_profile)
        profile["policy_model_forward_sec"] = time.perf_counter() - total_tic
        return model_pred, profile

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert a single observation into a VLAStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            VLAStepData object ready for processor input
        """
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the model's expectations.

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}: {observation[modality]}"
            )

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        # Validate each video stream defined in the modality config
        for video_key in self.modality_configs["video"].modality_keys:
            # Set or verify batch size consistency across all video keys
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"
                )

            # Check that the expected video key exists in the observation
            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )

            batched_video = observation["video"][video_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Validate each state stream defined in the modality config
        for state_key in self.modality_configs["state"].modality_keys:
            # Set or verify batch size consistency across all state keys
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"
                )

            # Check that the expected state key exists in the observation
            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )

            batched_state = observation["state"][state_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Validate each language stream defined in the modality config
        for language_key in self.modality_configs["language"].modality_keys:
            # Set or verify batch size consistency (language uses len instead of .shape)
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"
                )

            # Check that the expected language key exists in the observation
            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )

            batched_language: list[list[str]] = observation["language"][language_key]

            # Verify outer structure is a list (batch dimension)
            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )

            # Validate each batch item
            for batch_item in batched_language:
                # Verify temporal dimension matches expected horizon
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"
                )

                # Verify inner structure is also a list (temporal dimension)
                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )

                # Current implementation expects exactly one language instruction per timestep
                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )

                # Verify the instruction itself is a string
                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        profile_tic = time.perf_counter()
        profile: dict[str, Any] = {}

        # Step 1: Split batched observation into individual observations
        tic = time.perf_counter()
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []
        profile["policy_unbatch_observation_sec"] = time.perf_counter() - tic

        # Step 2: Process each observation through the VLA processor
        tic = time.perf_counter()
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))
        profile["policy_processor_sec"] = time.perf_counter() - tic

        # Step 3: Collate processed inputs into a single batch for model
        tic = time.perf_counter()
        collated_inputs = self.collate_fn(processed_inputs)
        profile["policy_collate_sec"] = time.perf_counter() - tic

        tic = time.perf_counter()
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=self.compute_dtype)
        profile["policy_dtype_cast_sec"] = time.perf_counter() - tic
        profile["policy_compute_dtype"] = str(self.compute_dtype).removeprefix("torch.")
        profile["policy_tensorrt_enabled"] = bool(self.tensorrt_targets)
        profile["policy_tensorrt_active"] = bool(
            self.tensorrt_status.get("compiled_targets", [])
        )
        profile["policy_tensorrt_requested_targets"] = sorted(self.tensorrt_targets)
        profile["policy_tensorrt_compiled_targets"] = list(
            self.tensorrt_status.get("compiled_targets", [])
        )
        profile["policy_tensorrt_failed_targets"] = dict(
            self.tensorrt_status.get("failed_targets", {})
        )

        # Step 4: Run model inference to predict actions
        model_pred, model_profile = self._get_model_action_with_profile(collated_inputs, options)
        profile.update(model_profile)

        model_device = _module_device(self.model)
        tic = time.perf_counter()
        normalized_action = model_pred["action_pred"].float()
        _sync_cuda_device(model_device)
        profile["policy_action_float_cast_sec"] = time.perf_counter() - tic

        # Step 5: Decode actions from normalized space back to physical units
        tic = time.perf_counter()
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        profile["policy_state_stack_sec"] = time.perf_counter() - tic

        tic = time.perf_counter()
        normalized_action_np = normalized_action.cpu().numpy()
        profile["policy_action_to_cpu_numpy_sec"] = time.perf_counter() - tic

        tic = time.perf_counter()
        unnormalized_action = self.processor.decode_action(
            normalized_action_np, self.embodiment_tag, batched_states
        )
        profile["policy_decode_action_sec"] = time.perf_counter() - tic

        # Cast all actions to float32 for consistency
        tic = time.perf_counter()
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        profile["policy_output_cast_sec"] = time.perf_counter() - tic
        profile["policy_total_sec"] = time.perf_counter() - profile_tic
        return casted_action, {"model_profile": profile}

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        This method ensures that all required action keys are present and that their
        data types, shapes, and dimensions match the model's action space.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension (e.g., joint positions, velocities, gripper state)

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails
        """
        # Validate each action key defined in the modality config
        for action_key in self.modality_configs["action"].modality_keys:
            # Check that the expected action key exists
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        return {}


class Gr00tSimPolicyWrapper(PolicyWrapper):
    """Wrapper for Gr00tPolicy to enable compatibility with existing Gr00t simulation environments.

    This wrapper is specifically designed for retro-fitting the Gr00t policy with the current
    Gr00t simulation environment interface. It handles the transformation between the flat
    observation format used by Gr00t sim environments (with keys like 'video.camera_name',
    'state.joint_positions') and the nested format expected by Gr00tPolicy.

    **Important**: If you are using other environments, custom robots, or building new environments,
    you should use `Gr00tPolicy` directly and format your observations according to its interface.
    This wrapper is only needed for compatibility with the existing Gr00t sim infrastructure.

    Key transformations performed by this wrapper:
    - Observation keys: 'video.cam' -> observation['video']['cam']
    - Observation keys: 'state.joints' -> observation['state']['joints']
    - Language keys: 'task' or 'annotation.human.coarse_action' -> observation['language']['task']
    - Action keys: action['joints'] -> 'action.joints'
    """

    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        """Initialize the wrapper around a Gr00tPolicy instance.

        Args:
            policy: The Gr00tPolicy instance to wrap
            strict: Whether to enforce strict validation (default: True)
        """
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy
        assert len(self.policy.modality_configs["language"].delta_indices) == 1, (
            "Only one language delta index is supported"
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation from Gr00t sim environment format.

        This validation is specific to the flat observation format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_observation which expects nested dicts, this expects flat keys.

        Expected observation structure (Gr00t sim format):
            - Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]
            - Flat keys like 'state.state_name': np.ndarray[np.float32, (B, T, D)]
            - Language keys: tuple[str] or list[str] with shape (B,)
                - Key can be 'task' or 'annotation.human.coarse_action' (for DC envs)

        Args:
            observation: Flat observation dictionary from Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # ===== VIDEO VALIDATION =====
        # Check video modalities with flat key format: 'video.camera_name'
        for video_key in modality_configs["video"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"video.{video_key}"
            assert parsed_key in observation, f"Video key '{parsed_key}' must be in observation"

            batched_video = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Check state modalities with flat key format: 'state.state_name'
        for state_key in modality_configs["state"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"state.{state_key}"
            assert parsed_key in observation, f"State key '{parsed_key}' must be in observation"

            batched_state = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Check language modalities (special handling for DC environment compatibility)
        for language_key in modality_configs["language"].modality_keys:
            # PATCH: Legacy compatibility for DC environments
            # DC envs use 'annotation.human.coarse_action' instead of 'task'
            if language_key == "task" and "annotation.human.coarse_action" in observation:
                language_key = "annotation.human.coarse_action"
            # /PATCH

            # Check that the expected language key exists
            assert language_key in observation, (
                f"Language key '{language_key}' must be in observation"
            )

            # In Gr00t sim format, language is a tuple of strings (B,)
            batched_language: tuple[str] | list[str] = observation[language_key]  # (B,)

            # Verify outer structure is a tuple (batch dimension)
            assert isinstance(batched_language, (tuple, list)), (
                f"Language key '{language_key}' must be a tuple or list. Got {type(batched_language)}"
            )

            # Verify each batch item is a string
            assert isinstance(batched_language[0], str), (
                f"Language batch item must be a string. Got {type(batched_language[0])}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform Gr00t sim observation format and compute actions.

        This method transforms the flat observation format from Gr00t sim environments
        into the nested format expected by Gr00tPolicy, computes actions, and transforms
        them back to the flat format expected by Gr00t sim environments.

        Input format (Gr00t sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (Gr00t sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from Gr00t sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)
        """
        # Transform flat observation format to nested format expected by Gr00tPolicy
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying Gr00tPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format for Gr00t sim environment
        # action['joints'] -> 'action.joints'
        return {f"action.{key}": action[key] for key in action}, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action in Gr00t sim environment format.

        This validation is specific to the flat action format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_action which expects nested dicts, this expects flat keys.

        Expected action structure (Gr00t sim format):
            - Flat keys like 'action.action_name': np.ndarray[np.float32, (B, T, D)]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Flat action dictionary for Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # Validate each action key defined in the modality config
        for action_key in modality_configs["action"].modality_keys:
            # Construct flat key expected in Gr00t sim environment (e.g., 'action.joints')
            parsed_key = f"action.{action_key}"
            assert parsed_key in action, f"Action key '{parsed_key}' must be in action"

            action_arr = action[parsed_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the underlying policy.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.policy.get_modality_config()
