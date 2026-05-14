from dataclasses import dataclass
import io
import time
from typing import Any, Callable

import msgpack
import numpy as np
import zmq

from gr00t.data.types import ModalityConfig
from gr00t.data.utils import to_json_serializable

from .policy import BasePolicy


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return ModalityConfig(**obj["as_json"])
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, ModalityConfig):
            return {
                "__ModalityConfig_class__": True,
                "as_json": to_json_serializable(obj),
            }
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(
        self,
        policy: BasePolicy,
        host: str = "*",
        port: int = 5555,
        api_token: str = None,
    ):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("get_action", self.policy.get_action)
        self.register_endpoint("reset", self.policy.reset)
        self.register_endpoint(
            "get_modality_config",
            getattr(self.policy, "get_modality_config", lambda: {}),
            requires_input=False,
        )

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    @staticmethod
    def _attach_profile_to_policy_result(
        result: Any,
        profile_key: str,
        profile: dict[str, Any],
    ) -> Any:
        """Attach profiling metadata to a ``(action, info)`` policy result."""
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            info = dict(result[1])
            info[profile_key] = profile
            return result[0], info
        if isinstance(result, list) and len(result) == 2 and isinstance(result[1], dict):
            info = dict(result[1])
            info[profile_key] = profile
            return [result[0], info]
        return result

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                recv_tic = time.perf_counter()
                message = self.socket.recv()
                received_at = time.perf_counter()
                request_decode_tic = time.perf_counter()
                request = MsgSerializer.from_bytes(message)
                request_decode_sec = time.perf_counter() - request_decode_tic

                # Validate token before processing request
                if not self._validate_token(request):
                    self.socket.send(
                        MsgSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
                    continue

                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                handler_tic = time.perf_counter()
                result = (
                    handler.handler(**request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                handler_sec = time.perf_counter() - handler_tic

                server_profile = {
                    "endpoint": endpoint,
                    "server_idle_recv_wait_sec": received_at - recv_tic,
                    "server_request_bytes": len(message),
                    "server_request_decode_sec": request_decode_sec,
                    "server_handler_sec": handler_sec,
                    "server_total_before_response_encode_sec": time.perf_counter() - received_at,
                }
                if endpoint == "get_action":
                    result = self._attach_profile_to_policy_result(
                        result,
                        "server_profile",
                        server_profile,
                    )

                response_encode_tic = time.perf_counter()
                response_payload = MsgSerializer.to_bytes(result)
                server_profile["server_response_encode_sec"] = (
                    time.perf_counter() - response_encode_tic
                )
                server_profile["server_response_bytes"] = len(response_payload)
                if endpoint == "get_action":
                    response_payload = MsgSerializer.to_bytes(result)

                send_tic = time.perf_counter()
                self.socket.send(response_payload)
                server_profile["server_send_sec"] = time.perf_counter() - send_tic
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))

    @staticmethod
    def start_server(policy: BasePolicy, port: int, host: str = "*", api_token: str = None):
        server = PolicyServer(policy, host=host, port=port, api_token=api_token)
        server.run()


class PolicyClient(BasePolicy):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
        strict: bool = False,
    ):
        super().__init__(strict=strict)
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.last_call_profile: dict[str, Any] = {}
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        call_tic = time.perf_counter()
        request_encode_tic = time.perf_counter()
        request_payload = MsgSerializer.to_bytes(request)
        request_encode_sec = time.perf_counter() - request_encode_tic

        send_tic = time.perf_counter()
        self.socket.send(request_payload)
        send_sec = time.perf_counter() - send_tic

        recv_tic = time.perf_counter()
        message = self.socket.recv()
        recv_sec = time.perf_counter() - recv_tic
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure we are running the correct policy server.")

        response_decode_tic = time.perf_counter()
        response = MsgSerializer.from_bytes(message)
        response_decode_sec = time.perf_counter() - response_decode_tic
        self.last_call_profile = {
            "endpoint": endpoint,
            "client_request_bytes": len(request_payload),
            "client_response_bytes": len(message),
            "client_request_encode_sec": request_encode_sec,
            "client_send_sec": send_sec,
            "client_recv_sec": recv_sec,
            "client_response_decode_sec": response_decode_sec,
            "client_roundtrip_sec": time.perf_counter() - call_tic,
        }

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        action, info = tuple(response)  # Convert list (from msgpack) to tuple of (action, info)
        if isinstance(info, dict):
            info = dict(info)
            info["client_profile"] = dict(self.last_call_profile)
        return action, info

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def check_observation(self, observation: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_observation is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )

    def check_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_action is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )
