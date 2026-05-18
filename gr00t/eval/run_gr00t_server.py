from dataclasses import dataclass
import json
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    compute_dtype: str = "bfloat16"
    """Floating point dtype for GR00T model inference: bfloat16, float16, or float32."""

    profile_model_detail: bool = True
    """Log detailed model inference breakdown. Disable with --no-profile-model-detail."""

    tensorrt_target: str = "none"
    """Comma-separated TensorRT compile targets: none, action_head, backbone, or all."""

    tensorrt_cache_dir: str = "/tmp/gr00t_tensorrt_cache"
    """Directory used for Torch-TensorRT timing/runtime caches."""

    tensorrt_min_block_size: int = 5
    """Minimum Torch-TensorRT partition size. Larger values reduce tiny TensorRT engines."""

    tensorrt_require_full_compilation: bool = False
    """Require selected target graphs to compile fully with TensorRT."""

    tensorrt_strict: bool = True
    """Raise if TensorRT setup fails. Disable to fall back to PyTorch modules."""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""


def main(config: ServerConfig):
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Compute dtype: {config.compute_dtype}")
    print(f"  Profile model detail: {config.profile_model_detail}")
    print(f"  TensorRT target: {config.tensorrt_target}")
    print(f"  TensorRT cache dir: {config.tensorrt_cache_dir}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # check if the model path exists
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Create and start the server
    if config.model_path is not None:
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
            compute_dtype=config.compute_dtype,
            profile_model_detail=config.profile_model_detail,
            tensorrt_target=config.tensorrt_target,
            tensorrt_cache_dir=config.tensorrt_cache_dir,
            tensorrt_min_block_size=config.tensorrt_min_block_size,
            tensorrt_require_full_compilation=config.tensorrt_require_full_compilation,
            tensorrt_strict=config.tensorrt_strict,
        )
    elif config.dataset_path is not None:
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
