"""Online WebSocket server for WNM-3D InteriorGS checkpoints.

Rank 0 receives WebSocket requests, keeps a rolling RGB frame buffer, converts
the latest observations into the same InteriorGS eval transform input used by
training, and broadcasts the converted observation to worker ranks.  For the 3D
VGGT/TGE path the server feeds a merged [visible-history 33 | target 33] RGB
clip; the WNM3DTransform then produces past_images for VGGT and images for the
Wan target video.  Online inference replans from the latest frame, so both the
VGGT history window and target clip include the latest/current frame, and the
response uses the first action block.

Client observation format:
  - video.rgb, image, rgb, observation/image, or observation/rgb:
      HWC uint8 single frame, or THWC uint8 frames.
  - state.nav_pose / nav_pose / state: optional [x, y, yaw], defaults to zeros.
  - annotation.language.language_instruction / prompt / instruction / language:
      optional language instruction.
  - session_id: optional; changing it resets frame buffers.

Response format:
  - action.nav_delta: (8, 3) physical [dx, dy, dyaw] by default.
  - action.nav_delta_scaled: (8, 3) unnormalised training-scale action.
  - action: alias of the primary returned action for simple clients.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import http
import logging
import os
import pickle
import socket
import time
import traceback
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
import torch.distributed as dist
import tyro
import websockets.asyncio.server as _server
import websockets.frames
from einops import rearrange
from openpi_client import msgpack_numpy
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from gammanav.vln.data.schema import EmbodimentTag
from gammanav.vln.model.wnm_3d.inference_policy import (
    PolicyBatch,
    WNM3DInferencePolicy,
)
from wnm_3d_common import (
    as_numpy as _as_numpy,
    batch_get as _batch_get,
    configure_torch_dynamo as _configure_torch_dynamo,
    load_checkpoint_metadata as _load_checkpoint_metadata,
    reset_causal_state as _reset_causal_state,
    resolve_nav_action_scale as _resolve_nav_action_scale,
    restore_policy_video_metadata as _restore_policy_video_metadata,
    set_causal_eval_state_horizon as _set_causal_eval_state_horizon,
)


logger = logging.getLogger("wnm_3d_server")

CONTINUE_SIGNAL = 0
SHUTDOWN_SIGNAL = 1
IDLE_SIGNAL = 2
RESET_SIGNAL = 3


@dataclasses.dataclass
class Args:
    model_path: str
    port: int = 8000
    host: str = "127.0.0.1"
    timeout_seconds: int = 50000
    max_message_size_bytes: int = 64 * 1024 * 1024
    enable_dit_cache: bool = False
    num_inference_steps: int | None = None
    enable_cfg: bool = False
    cfg_scale: float = 5.0
    index: int = 0
    output_dir: str | None = None
    nav_action_scale: float | None = None
    return_scaled_action: bool = False
    resize_input_to_checkpoint_resolution: bool = False
    history_sampling: str = "uniform"
    history_long_range_anchors: int = 8
    save_input_clips: bool = False
    save_generated_video: bool = False
    max_chunk_size: int | None = None
    profile_module_timings: bool = False


def init_mesh() -> DeviceMesh:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device_id = None
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device_id = torch.device("cuda", local_rank)

    dist.init_process_group(backend, device_id=device_id)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        device = device_id
        logger.info("Rank %s/%s using device %s", rank, world_size, device)
        return init_device_mesh(
            device_type="cuda",
            mesh_shape=(world_size,),
            mesh_dim_names=("ip",),
        )

    logger.info("Rank %s/%s using CPU distributed mesh", rank, world_size)
    return init_device_mesh(
        device_type="cpu",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )


def _health_check(
    connection: _server.ServerConnection,
    request: _server.Request,
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _broadcast_signal(signal_group: dist.ProcessGroup, signal: int) -> None:
    tensor = torch.tensor([signal], dtype=torch.int32, device="cpu")
    dist.broadcast(tensor, src=0, group=signal_group)


def _broadcast_obs(obs: dict[str, Any]) -> None:
    serialized = pickle.dumps(obs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size_tensor = torch.tensor([len(serialized)], dtype=torch.int64, device=device)
    dist.broadcast(size_tensor, src=0)
    data_tensor = torch.frombuffer(bytearray(serialized), dtype=torch.uint8).to(device)
    dist.broadcast(data_tensor, src=0)


def _receive_obs() -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size_tensor = torch.zeros(1, dtype=torch.int64, device=device)
    dist.broadcast(size_tensor, src=0)
    data_tensor = torch.zeros(int(size_tensor.item()), dtype=torch.uint8, device=device)
    dist.broadcast(data_tensor, src=0)
    return pickle.loads(data_tensor.cpu().numpy().tobytes())


def _reset_policy_state(policy: WNM3DInferencePolicy) -> None:
    _reset_causal_state(policy)
    action_head = policy.trained_model.action_head
    if hasattr(action_head, "language"):
        action_head.language = None


def _first_present(obs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in obs:
            return obs[key]
    raise KeyError(f"None of the expected keys are present: {keys}")


def _string_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 0:
        return ""
    return str(arr.reshape(-1)[0])


class InteriorGSNavOnlinePolicy:
    def __init__(
        self,
        policy: WNM3DInferencePolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None,
        raw_resolution: tuple[int, int],
        nav_action_scale: float,
        return_scaled_action: bool,
        resize_input_to_checkpoint_resolution: bool,
        history_sampling: str,
        history_long_range_anchors: int,
        save_input_clips: bool,
        save_generated_video: bool,
    ) -> None:
        self._policy = policy
        self._signal_group = signal_group
        self._output_dir = output_dir
        self._raw_resolution = raw_resolution
        self._nav_action_scale = nav_action_scale
        self._return_scaled_action = return_scaled_action
        self._resize_input_to_checkpoint_resolution = (
            resize_input_to_checkpoint_resolution
        )
        self._history_sampling = history_sampling
        self._history_long_range_anchors = history_long_range_anchors
        self._should_save_input_clips = save_input_clips
        self._should_save_generated_video = save_generated_video

        action_head = policy.trained_model.action_head
        self.video_key = policy.modality_configs.video.modality_keys[0]
        self.state_key = policy.modality_configs.state.modality_keys[0]
        self.action_key = policy.modality_configs.action.modality_keys[0]
        self.language_key = policy.modality_configs.language.modality_keys[0]
        self.action_horizon = int(action_head.action_horizon)
        self._target_num_frames = int(action_head.num_frames)

        self._frame_buffer: list[np.ndarray] = []
        self._is_first_call = True
        self._current_session_id: str | None = None
        self._msg_index = 0
        self.video_across_time: list[torch.Tensor] = []

        if self._output_dir and (
            self._should_save_input_clips or self._should_save_generated_video
        ):
            os.makedirs(self._output_dir, exist_ok=True)
            if self._should_save_input_clips:
                os.makedirs(os.path.join(self._output_dir, "inputs"), exist_ok=True)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._maybe_reset_for_obs(obs)
        self._msg_index += 1

        converted_obs = self._convert_observation(obs)
        logger.debug(
            "Converted obs: video=%s state=%s language=%r target_prefix=%s",
            converted_obs[self.video_key].shape,
            converted_obs[self.state_key].shape,
            converted_obs[self.language_key],
            converted_obs["target_prefix_frames"],
        )
        self._save_input_clip(converted_obs)

        _broadcast_signal(self._signal_group, CONTINUE_SIGNAL)
        _broadcast_obs(converted_obs)

        dist.barrier()
        forward_start = time.perf_counter()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(
                PolicyBatch(obs=converted_obs)
            )
        dist.barrier()
        logger.info("Forward time: %.3f seconds", time.perf_counter() - forward_start)

        if video_pred is not None and self._should_save_generated_video:
            self.video_across_time.append(video_pred)

        pred_scaled = _as_numpy(_batch_get(result_batch.act, self.action_key)).astype(
            np.float32
        )
        if pred_scaled.ndim == 3 and pred_scaled.shape[0] == 1:
            pred_scaled = pred_scaled[0]
        if pred_scaled.ndim == 1:
            pred_scaled = pred_scaled.reshape(1, -1)
        pred_scaled = self._select_online_action_block(pred_scaled)
        pred_scaled = pred_scaled[:, :3].astype(np.float32, copy=False)
        pred_physical = (pred_scaled / self._nav_action_scale).astype(np.float32)
        logger.debug(
            "Pred action: scaled_shape=%s scaled_range=(%.6f, %.6f) physical_range=(%.6f, %.6f)",
            pred_scaled.shape,
            float(pred_scaled.min()) if pred_scaled.size else 0.0,
            float(pred_scaled.max()) if pred_scaled.size else 0.0,
            float(pred_physical.min()) if pred_physical.size else 0.0,
            float(pred_physical.max()) if pred_physical.size else 0.0,
        )

        primary_action = pred_scaled if self._return_scaled_action else pred_physical
        if self._is_first_call:
            self._is_first_call = False

        return {
            "action_type": "nav_delta",
            "actions": pred_physical,
            "action": primary_action,
            self.action_key: pred_physical,
            f"{self.action_key}_scaled": pred_scaled,
            "nav_action_scale": np.asarray(self._nav_action_scale, dtype=np.float32),
            "num_actions": np.asarray(primary_action.shape[0], dtype=np.int64),
            "unit": "scaled" if self._return_scaled_action else "physical",
        }

    def reset(self, send_to_workers: bool = True, save_video: bool = True) -> None:
        if send_to_workers and dist.get_world_size() > 1:
            _broadcast_signal(self._signal_group, RESET_SIGNAL)
        self._reset_local(save_video=save_video)

    def shutdown_workers(self) -> None:
        if dist.get_world_size() > 1:
            _broadcast_signal(self._signal_group, SHUTDOWN_SIGNAL)

    def _maybe_reset_for_obs(self, obs: dict[str, Any]) -> None:
        if bool(obs.get("reset", False)):
            self.reset(send_to_workers=True, save_video=True)
            return

        session_id = obs.get("session_id")
        if session_id is None:
            return

        session_id = _string_value(session_id)
        if self._current_session_id is None:
            self._current_session_id = session_id
            logger.info("New session started: %s", session_id)
            return

        if session_id != self._current_session_id:
            logger.info(
                "Session changed from %s to %s; resetting InteriorGS state",
                self._current_session_id,
                session_id,
            )
            self.reset(send_to_workers=True, save_video=True)
            self._current_session_id = session_id

    def _reset_local(self, save_video: bool) -> None:
        if save_video:
            self._save_generated_video()
        _reset_policy_state(self._policy)
        self._frame_buffer = []
        self._is_first_call = True
        self.video_across_time = []

    def _convert_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        image_value = _first_present(
            obs,
            (
                self.video_key,
                "video.rgb",
                "images",
                "observation/rgb",
                "observation/image",
                "rgb",
                "image",
            ),
        )
        incoming_frames = self._to_frame_list(image_value)
        logger.debug(
            "Incoming frames: count=%d first_shape=%s first_dtype=%s",
            len(incoming_frames),
            incoming_frames[0].shape if incoming_frames else None,
            incoming_frames[0].dtype if incoming_frames else None,
        )
        self._frame_buffer.extend(incoming_frames)

        frames, target_prefix_frames = self._select_past_target_buffer_frames()

        video = np.stack(frames, axis=0).astype(np.uint8, copy=False)
        state = self._extract_state(obs)
        language = self._extract_language(obs)

        return {
            self.video_key: video,
            self.state_key: state,
            self.language_key: language,
            "target_prefix_frames": np.asarray(target_prefix_frames, dtype=np.int64),
        }

    def _to_frame_list(self, value: Any) -> list[np.ndarray]:
        arr = np.asarray(value)
        if arr.ndim == 3:
            return [self._prepare_frame(arr)]
        if arr.ndim == 4:
            return [self._prepare_frame(frame) for frame in arr]
        raise ValueError(
            f"InteriorGS image input must be HWC or THWC, got shape {arr.shape}"
        )

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.ndim != 3:
            raise ValueError(f"Frame must be HWC, got shape {frame.shape}")
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.shape[-1] != 3:
            raise ValueError(f"Frame must have 3 RGB channels, got shape {frame.shape}")

        if frame.dtype != np.uint8:
            frame = frame.astype(np.float32)
            min_val = float(np.nanmin(frame))
            max_val = float(np.nanmax(frame))
            if min_val >= -1.1 and max_val <= 1.1:
                if min_val < 0:
                    frame = (frame + 1.0) * 127.5
                else:
                    frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        raw_w, raw_h = self._raw_resolution
        h, w = frame.shape[:2]
        if (w, h) != (raw_w, raw_h):
            if not self._resize_input_to_checkpoint_resolution:
                raise ValueError(
                    f"InteriorGS frame has resolution {(w, h)}, expected raw "
                    f"checkpoint resolution {(raw_w, raw_h)}. Send raw frames or "
                    "start with --resize-input-to-checkpoint-resolution."
                )
            import cv2

            frame = cv2.resize(frame, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)

        return np.ascontiguousarray(frame)

    def _select_past_target_buffer_frames(self) -> tuple[list[np.ndarray], int]:
        if not self._frame_buffer:
            raise ValueError("No InteriorGS frames received")
        current_idx = len(self._frame_buffer) - 1
        max_idx = current_idx

        def clip_index(index: int) -> int:
            return min(max(int(index), 0), max_idx)

        # VGGT 3D-WAM replans from the latest observed frame.  All visible
        # frames, including current, can feed VGGT; target frames seed the
        # current-frame i2v condition/shape.
        past_indices = [
            clip_index(idx) for idx in self._select_history_indices(current_idx)
        ]
        target_indices = [
            clip_index(idx)
            for idx in range(current_idx, current_idx + self._target_num_frames)
        ]
        past_frames = [self._frame_buffer[idx] for idx in past_indices]
        target_frames = [self._frame_buffer[idx] for idx in target_indices]
        target_prefix_frames = 1
        return past_frames + target_frames, target_prefix_frames

    def _select_history_indices(self, current_idx: int) -> list[int]:
        if self._history_sampling == "uniform":
            if current_idx <= 0:
                return [0] * self._target_num_frames
            return np.linspace(
                0,
                current_idx,
                num=self._target_num_frames,
                dtype=np.int64,
            ).tolist()

        if (
            self._history_sampling == "recent"
            or current_idx + 1 <= self._target_num_frames
        ):
            return list(
                range(
                    current_idx - self._target_num_frames + 1,
                    current_idx + 1,
                )
            )

        if self._history_sampling != "mixed":
            raise ValueError(
                "Unsupported history_sampling "
                f"{self._history_sampling!r}; expected 'recent', 'mixed', or 'uniform'."
            )

        anchor_count = min(
            max(0, int(self._history_long_range_anchors)),
            self._target_num_frames - 1,
        )
        if anchor_count == 0:
            return list(
                range(
                    current_idx - self._target_num_frames + 1,
                    current_idx + 1,
                )
            )

        recent_count = self._target_num_frames - anchor_count
        recent_start = current_idx - recent_count + 1
        if recent_start <= 0:
            return list(
                range(
                    current_idx - self._target_num_frames + 1,
                    current_idx + 1,
                )
            )

        early_indices = np.linspace(
            0,
            recent_start - 1,
            num=anchor_count,
            dtype=np.int64,
        ).tolist()
        recent_indices = list(range(recent_start, current_idx + 1))
        return early_indices + recent_indices

    def _select_online_action_block(self, actions: np.ndarray) -> np.ndarray:
        if actions.shape[0] <= self.action_horizon:
            return actions
        # Online VGG-T inference is receding-horizon MPC: execute block 1, then
        # rebuild past VGGT tokens from new simulator observations.
        return actions[: self.action_horizon]

    def _extract_state(self, obs: dict[str, Any]) -> np.ndarray:
        for key in (self.state_key, "state.nav_pose", "nav_pose", "state"):
            if key not in obs:
                continue
            state = np.asarray(obs[key], dtype=np.float32)
            if state.ndim == 1:
                state = state.reshape(1, -1)
            if state.ndim == 2:
                return state[:, :3].astype(np.float32, copy=False)
            raise ValueError(f"State must be shape (3,) or (T,3), got {state.shape}")
        return np.zeros((1, 3), dtype=np.float32)

    def _extract_language(self, obs: dict[str, Any]) -> str:
        for key in (
            self.language_key,
            "annotation.language.language_instruction",
            "prompt",
            "instruction",
            "language",
            "text",
        ):
            if key in obs:
                return _string_value(obs[key])
        return ""

    def _save_input_clip(self, converted_obs: dict[str, Any]) -> None:
        if not self._should_save_input_clips or not self._output_dir:
            return
        frames = converted_obs.get(self.video_key)
        if frames is None:
            return
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        save_dir = os.path.join(
            self._output_dir,
            "inputs",
            f"{self._msg_index:06d}_{timestamp}",
            self.video_key.replace("/", "_").replace(".", "_"),
        )
        try:
            os.makedirs(save_dir, exist_ok=True)
            for idx, frame in enumerate(np.asarray(frames)):
                imageio.imwrite(os.path.join(save_dir, f"f{idx:02d}.png"), frame)
        except Exception as exc:
            logger.warning("Failed to save InteriorGS input frames: %s", exc)

    def _save_generated_video(self) -> None:
        if (
            not self._should_save_generated_video
            or not self._output_dir
            or not self.video_across_time
        ):
            return
        try:
            video_across_time_cat = torch.cat(self.video_across_time, dim=2)
            action_head = self._policy.trained_model.action_head
            frames = action_head.vae.decode(
                video_across_time_cat,
                tiled=action_head.tiled,
                tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                tile_stride=(
                    action_head.tile_stride_height,
                    action_head.tile_stride_width,
                ),
            )
            frames = rearrange(frames, "B C T H W -> B T H W C")[0]
            frames = (
                ((frames.float() + 1) * 127.5)
                .clip(0, 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            all_mp4_files = [
                f for f in os.listdir(self._output_dir) if f.endswith(".mp4")
            ]
            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
            output_path = os.path.join(
                self._output_dir,
                f"{len(all_mp4_files):06}_{timestamp}.mp4",
            )
            imageio.mimsave(output_path, list(frames), fps=5, codec="libx264")
            logger.info("Saved generated video to: %s", output_path)
        except Exception as exc:
            logger.warning("Failed to save generated video: %s", exc)


class InteriorGSNavWebsocketServer:
    def __init__(
        self,
        adapter: InteriorGSNavOnlinePolicy,
        host: str,
        port: int,
        max_message_size: int,
        metadata: dict[str, Any],
    ) -> None:
        self._adapter = adapter
        self._host = host
        self._port = port
        self._max_message_size = max_message_size
        self._metadata = metadata
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=self._max_message_size,
            process_request=_health_check,
            ping_interval=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        try:
            while True:
                try:
                    obs = msgpack_numpy.unpackb(await websocket.recv())
                    endpoint = obs.pop("endpoint", "infer")
                    if endpoint == "reset":
                        self._adapter.reset(send_to_workers=True, save_video=True)
                        await websocket.send(
                            packer.pack({"status": "reset successful"})
                        )
                        continue
                    if endpoint not in ("infer", "action", None):
                        raise ValueError(f"Unsupported endpoint: {endpoint}")

                    start = time.perf_counter()
                    response = self._adapter.infer(obs)
                    logger.info(
                        "Request time: %.3f seconds", time.perf_counter() - start
                    )
                    await websocket.send(packer.pack(response))

                except websockets.ConnectionClosed:
                    logger.info("Connection from %s closed", websocket.remote_address)
                    break
                except Exception:
                    logger.exception("Request from %s failed", websocket.remote_address)
                    try:
                        await websocket.send(
                            packer.pack({"error": "Internal server error"})
                        )
                    finally:
                        await websocket.close(
                            code=websockets.frames.CloseCode.INTERNAL_ERROR,
                            reason="Internal server error.",
                        )
                    raise
        finally:
            self._adapter.reset(send_to_workers=True, save_video=True)


async def worker_loop(
    policy: WNM3DInferencePolicy,
    signal_group: dist.ProcessGroup,
) -> None:
    logger.info("Worker loop started for rank %d", dist.get_rank())
    signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
    while True:
        try:
            dist.broadcast(signal_tensor, src=0, group=signal_group)
            signal = int(signal_tensor.item())
            if signal == SHUTDOWN_SIGNAL:
                logger.info("Rank %d received shutdown signal", dist.get_rank())
                break
            if signal == IDLE_SIGNAL:
                continue
            if signal == RESET_SIGNAL:
                _reset_policy_state(policy)
                continue
            if signal != CONTINUE_SIGNAL:
                raise ValueError(f"Unknown worker signal: {signal}")

            obs = _receive_obs()
            dist.barrier()
            with torch.no_grad():
                policy.lazy_joint_forward_causal(PolicyBatch(obs=obs))
            dist.barrier()
        except Exception as exc:
            logger.error("Worker loop error on rank %d: %s", dist.get_rank(), exc)
            traceback.print_exc()
            break


def _make_output_dir(args: Args) -> str:
    if args.output_dir:
        return args.output_dir
    model_path = Path(args.model_path)
    parent_dir = model_path.parent
    checkpoint_name = model_path.name
    date_suffix = datetime.datetime.now().strftime("%Y%m%d")
    return str(
        parent_dir
        / f"interiorgs_nav_3d_online_{date_suffix}_{args.index}"
        / checkpoint_name
    )


def main(args: Args) -> None:
    if args.max_message_size_bytes <= 0:
        raise ValueError("--max-message-size-bytes must be a positive integer")

    if args.num_inference_steps is not None:
        if int(args.num_inference_steps) <= 0:
            raise ValueError(
                f"--num-inference-steps must be positive, got {args.num_inference_steps}."
            )
        if int(args.num_inference_steps) > 16:
            raise ValueError(
                "--num-inference-steps cannot exceed the fixed 16-step DiT mask."
            )
        os.environ["NUM_INFERENCE_STEPS"] = str(int(args.num_inference_steps))
    if not np.isfinite(args.cfg_scale) or args.cfg_scale <= 0:
        raise ValueError(
            f"--cfg-scale must be a positive finite number, got {args.cfg_scale}."
        )

    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ENABLE_CFG"] = "true" if args.enable_cfg else "false"
    os.environ["CFG_SCALE"] = str(float(args.cfg_scale))
    os.environ["WNM3D_PROFILE_MODULE_TIMES"] = (
        "true" if args.profile_module_timings else "false"
    )
    os.environ.setdefault("ATTENTION_BACKEND", "FA2")
    _configure_torch_dynamo()

    model_path = Path(args.model_path)
    device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)

    policy = WNM3DInferencePolicy(
        embodiment_tag=EmbodimentTag.INTERIORGS,
        model_path=str(model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )
    policy.trained_model.eval()

    if args.max_chunk_size is not None:
        policy.trained_model.action_head.max_chunk_size = int(args.max_chunk_size)

    video_key = policy.modality_configs.video.modality_keys[0]
    checkpoint_metadata = _load_checkpoint_metadata(
        model_path, EmbodimentTag.INTERIORGS
    )
    reset_video_transforms = _restore_policy_video_metadata(
        policy=policy,
        metadata=checkpoint_metadata,
        video_key=video_key,
    )
    state_transforms = _set_causal_eval_state_horizon(policy)

    raw_resolution = checkpoint_metadata.modalities.video[
        video_key.replace("video.", "")
    ].resolution
    action_head = policy.trained_model.action_head
    if rank == 0:
        effective_dit_steps = (
            sum(action_head.dit_step_mask)
            if action_head.enable_dit_cache
            else int(action_head.num_inference_steps)
        )
        logger.info(
            "Inference schedule: steps=%d/%d dit_cache=%s cfg=%s cfg_scale=%g "
            "model_forwards=%d module_profile=%s",
            effective_dit_steps,
            int(action_head.num_inference_steps),
            bool(action_head.enable_dit_cache),
            "on" if action_head.enable_cfg else "off",
            float(action_head.cfg_scale),
            effective_dit_steps * (2 if action_head.enable_cfg else 1),
            bool(action_head.profile_module_times),
        )
    nav_action_scale = _resolve_nav_action_scale(
        policy.train_cfg,
        override=args.nav_action_scale,
        default=1.0,
    )
    history_sampling = str(args.history_sampling).lower()
    if history_sampling not in {"recent", "mixed", "uniform"}:
        raise ValueError(
            "--history-sampling must be 'recent', 'mixed', or 'uniform', "
            f"got {args.history_sampling!r}"
        )
    history_long_range_anchors = int(args.history_long_range_anchors)
    if history_long_range_anchors < 0:
        raise ValueError("--history-long-range-anchors must be >= 0")
    action_horizon = int(action_head.action_horizon)

    output_dir = (
        _make_output_dir(args)
        if rank == 0 and (args.save_input_clips or args.save_generated_video)
        else None
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "embodiment": EmbodimentTag.INTERIORGS.value,
        "model_name": "wnm-3d",
        "model_path": str(model_path),
        "input_format": {
            "image_keys": [
                "video.rgb",
                "images",
                "image",
                "rgb",
                "observation/image",
                "observation/rgb",
            ],
            "image_shape": "HWC uint8 single frame or THWC uint8 frame chunk",
            "raw_resolution": raw_resolution,
            "state_key": "state.nav_pose",
            "language_key": "annotation.language.language_instruction",
            "session_id": "optional; changing it resets frame buffers",
        },
        "output_format": {
            "action_type": "nav_delta",
            "actions": f"({action_horizon}, 3) physical dx, dy, dyaw",
            "action": f"({action_horizon}, 3) primary next action block",
            "action.nav_delta": f"({action_horizon}, 3) physical dx, dy, dyaw",
            "action.nav_delta_scaled": f"({action_horizon}, 3) training-scale dx, dy, dyaw",
            "unit": "scaled" if args.return_scaled_action else "physical",
        },
        "wam_layout": "past 3D obs | noisy video | noisy action | state",
        "target_frames": int(action_head.num_frames),
        "nav_action_scale": nav_action_scale,
        "history_sampling": history_sampling,
        "history_long_range_anchors": history_long_range_anchors,
        "reset_video_transforms": reset_video_transforms,
        "state_transforms": state_transforms,
    }

    if rank == 0:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info(
            "Creating InteriorGS 3D WAM nav server at %s:%s", args.host, args.port
        )
        logger.info("Host: %s (%s)", hostname, local_ip)
        logger.info("Output dir: %s", output_dir)
        logger.info("Metadata: %s", metadata)
        adapter = InteriorGSNavOnlinePolicy(
            policy=policy,
            signal_group=signal_group,
            output_dir=output_dir,
            raw_resolution=raw_resolution,
            nav_action_scale=nav_action_scale,
            return_scaled_action=bool(args.return_scaled_action),
            resize_input_to_checkpoint_resolution=bool(
                args.resize_input_to_checkpoint_resolution
            ),
            history_sampling=history_sampling,
            history_long_range_anchors=history_long_range_anchors,
            save_input_clips=bool(args.save_input_clips),
            save_generated_video=bool(args.save_generated_video),
        )
        server = InteriorGSNavWebsocketServer(
            adapter=adapter,
            host=args.host,
            port=args.port,
            max_message_size=int(args.max_message_size_bytes),
            metadata=metadata,
        )
        server.serve_forever()
    else:
        asyncio.run(worker_loop(policy=policy, signal_group=signal_group))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
