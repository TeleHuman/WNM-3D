import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from gammanav.vln.data.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    EmbodimentTag,
    LeRobotModalityMetadata,
    StateActionMetadata,
    VideoMetadata,
)
from gammanav.vln.data.transform import ComposedModalityTransform

from .lerobot import (
    LE_ROBOT_INFO_FILENAME,
    LE_ROBOT_MODALITY_FILENAME,
    LeRobotSingleDataset,
)

cv2.setNumThreads(0)


ACTION_STATISTICS_VERSION = 2
ACTION_STATISTICS_SCOPE = "eligible_step_filter_full_action_windows"


class ShardedLeRobotSingleActionChunkDatasetInteriorGS(LeRobotSingleDataset):
    """LeRobot dataset whose navigation action is computed from camera extrinsics.

    InteriorGS is stored in LeRobot layout, but ``state.nav_pose`` and
    ``action.nav_delta`` are virtual modalities: they are not columns in the
    parquet files. Video/language sampling follows the normal LeRobot
    ``delta_indices`` path; the action reader projects camera centers into the
    start-frame navigation coordinate system and returns frame-to-frame
    ``[dx, dy, dyaw]`` labels.
    """

    CAM_TO_NAV = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict,
        embodiment_tag: str | EmbodimentTag = EmbodimentTag.INTERIORGS,
        transforms: ComposedModalityTransform | None = None,
        discard_bad_trajectories: bool = True,
        max_chunk_size: int | None = 4,
        num_frames: int = 33,
        action_horizon: int = 8,
        state_horizon: int | None = None,
        num_frame_per_block: int = 2,
        num_action_per_block: int | None = None,
        num_state_per_block: int = 1,
        vae_temporal_stride: int = 4,
        video_frame_stride: int = 1,
        action_frame_stride: int | None = None,
        history_sampling: str = "uniform",
        nav_action_scale: float = 4.0,
        nav_smooth_window: int = 7,
        nav_smooth_passes: int = 2,
        compute_statistics: bool = True,
        max_stats_episodes: int | None = 512,
        stats_start_stride: int = 16,
        **kwargs: Any,
    ):
        self.num_frames = int(num_frames)
        self.action_horizon = int(action_horizon)
        self.num_frame_per_block = int(num_frame_per_block)
        self.num_action_per_block = (
            int(num_action_per_block) if num_action_per_block is not None else None
        )
        self.num_state_per_block = int(num_state_per_block)
        self.vae_temporal_stride = int(vae_temporal_stride)
        self.video_frame_stride = int(video_frame_stride)
        self.action_frame_stride = (
            int(action_frame_stride)
            if action_frame_stride is not None
            else self.video_frame_stride
        )
        self.history_sampling = str(history_sampling).lower()
        if self.history_sampling not in {"recent", "uniform"}:
            raise ValueError(
                "history_sampling must be 'recent' or 'uniform', "
                f"got {history_sampling!r}."
            )
        self.nav_action_scale = float(nav_action_scale)
        self.nav_smooth_window = int(nav_smooth_window)
        self.nav_smooth_passes = int(nav_smooth_passes)
        self.compute_statistics = bool(compute_statistics)
        self.max_stats_episodes = max_stats_episodes
        self.stats_start_stride = int(stats_start_stride)
        self.past_video_num_frames = self.num_frames

        self._validate_temporal_layout(
            max_chunk_size=max_chunk_size, state_horizon=state_horizon
        )
        modality_configs = copy.deepcopy(modality_configs)
        self._fill_default_delta_indices(modality_configs)

        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            transforms=transforms,
            discard_bad_trajectories=discard_bad_trajectories,
            max_chunk_size=self.num_blocks,
            **kwargs,
        )

    def _validate_temporal_layout(
        self, max_chunk_size: int | None, state_horizon: int | None
    ) -> None:
        if self.num_frames < 2:
            raise ValueError(f"num_frames must be >= 2, got {self.num_frames}")
        if self.vae_temporal_stride <= 0:
            raise ValueError(
                f"vae_temporal_stride must be positive, got {self.vae_temporal_stride}"
            )
        if (self.num_frames - 1) % self.vae_temporal_stride != 0:
            raise ValueError(
                f"num_frames - 1 must be divisible by vae_temporal_stride. Got "
                f"num_frames={self.num_frames}, vae_temporal_stride={self.vae_temporal_stride}."
            )

        latent_intervals = (self.num_frames - 1) // self.vae_temporal_stride
        if latent_intervals % self.num_frame_per_block != 0:
            raise ValueError(
                "VAE-compressed video intervals must be divisible by num_frame_per_block. "
                f"Got latent_intervals={latent_intervals}, "
                f"num_frame_per_block={self.num_frame_per_block}."
            )

        self.num_blocks = latent_intervals // self.num_frame_per_block
        if max_chunk_size is not None and int(max_chunk_size) < self.num_blocks:
            raise ValueError(
                f"max_chunk_size must cover the derived target blocks. Expected at least "
                f"{self.num_blocks}, got {max_chunk_size}."
            )

        if self.num_action_per_block is None:
            self.num_action_per_block = self.action_horizon
        elif self.num_action_per_block != self.action_horizon:
            raise ValueError(
                "InteriorGS uses action_horizon as the number of actions per block, "
                f"so num_action_per_block must equal action_horizon. Got "
                f"num_action_per_block={self.num_action_per_block}, "
                f"action_horizon={self.action_horizon}."
            )
        self.total_action_horizon = self.action_horizon * self.num_blocks
        if self.total_action_horizon != self.num_frames - 1:
            raise ValueError(
                "InteriorGS currently assumes one action between adjacent selected video frames. "
                f"Expected action_horizon * num_blocks == num_frames - 1, got "
                f"{self.action_horizon} * {self.num_blocks} != {self.num_frames - 1}."
            )

        expected_state_horizon = self.num_blocks * self.num_state_per_block
        self.state_horizon = (
            int(state_horizon) if state_horizon is not None else expected_state_horizon
        )
        if self.state_horizon != expected_state_horizon:
            raise ValueError(
                f"state_horizon must be num_blocks * num_state_per_block. Expected "
                f"{expected_state_horizon}, got {self.state_horizon}."
            )

    def _fill_default_delta_indices(self, modality_configs: dict) -> None:
        target_video_delta = (
            np.arange(self.num_frames, dtype=np.int64) * self.video_frame_stride
        )
        # This fixed-size layout establishes the merged video shape. Uniform
        # history depends on the sampled current step and is resolved later in
        # _resolve_history_indices().
        past_video_delta = (
            np.arange(1 - self.past_video_num_frames, 1, dtype=np.int64)
            * self.video_frame_stride
        )
        video_delta = np.concatenate([past_video_delta, target_video_delta]).tolist()
        action_delta = (
            np.arange(self.total_action_horizon, dtype=np.int64)
            * self.action_frame_stride
        ).tolist()
        block_stride = (
            self.vae_temporal_stride
            * self.num_frame_per_block
            * self.video_frame_stride
        )
        state_delta = (
            np.arange(self.state_horizon, dtype=np.int64) * block_stride
        ).tolist()

        if "video" in modality_configs:
            self._set_or_validate_delta_indices(
                modality_configs["video"], video_delta, "video"
            )
        if "action" in modality_configs:
            self._set_or_validate_delta_indices(
                modality_configs["action"], action_delta, "action"
            )
        if "state" in modality_configs:
            self._set_or_validate_delta_indices(
                modality_configs["state"], state_delta, "state"
            )

    def _select_history_indices(self, current_idx: int) -> np.ndarray:
        """Select visible history ending at the current episode frame."""
        if current_idx < 0:
            raise ValueError(f"current_idx must be non-negative, got {current_idx}.")
        if self.history_sampling == "uniform":
            return np.linspace(
                0,
                current_idx,
                num=self.past_video_num_frames,
                dtype=np.int64,
            )
        return current_idx + (
            np.arange(1 - self.past_video_num_frames, 1, dtype=np.int64)
            * self.video_frame_stride
        )

    def _resolve_history_indices(
        self, indices: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Replace the merged video's history prefix with per-step indices."""
        resolved_indices = dict(indices)
        expected_video_frames = self.past_video_num_frames + self.num_frames
        for key in self.modality_keys.get("video", []):
            if key not in resolved_indices:
                continue
            video_indices = np.asarray(resolved_indices[key], dtype=np.int64)
            if video_indices.ndim != 1 or len(video_indices) != expected_video_frames:
                raise ValueError(
                    f"InteriorGS video indices for {key!r} must contain "
                    f"{expected_video_frames} merged history/target frames, got "
                    f"shape {video_indices.shape}."
                )
            target_indices = video_indices[self.past_video_num_frames :]
            current_idx = int(target_indices[0])
            history_indices = self._select_history_indices(current_idx)
            resolved_indices[key] = np.concatenate([history_indices, target_indices])
        return resolved_indices

    def get_step_data(self, trajectory_id: int, indices: dict[str, np.ndarray]) -> dict:
        return super().get_step_data(
            trajectory_id, self._resolve_history_indices(indices)
        )

    @staticmethod
    def _set_or_validate_delta_indices(
        config: Any, expected: list[int], name: str
    ) -> None:
        current = list(getattr(config, "delta_indices", []) or [])
        if not current:
            config.delta_indices = expected
            return
        if len(current) != len(expected):
            raise ValueError(
                f"InteriorGS {name}.delta_indices has length {len(current)}, "
                f"but expected {len(expected)}."
            )

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        with modality_meta_path.open("r", encoding="utf-8") as f:
            modality_meta = json.load(f)
        return LeRobotModalityMetadata.model_validate(modality_meta)

    def _check_integrity(self) -> None:
        for modality, modality_config in self.modality_configs.items():
            for key in modality_config.modality_keys:
                if self._is_virtual_state_key(key) or self._is_virtual_action_key(key):
                    continue
                if modality == "language" and key.startswith("annotation."):
                    self.lerobot_modality_meta.get_key_meta(key)
                    continue
                if modality == "video":
                    self.lerobot_modality_meta.get_key_meta(key)

    def _get_metadata(self) -> DatasetMetadata:
        video_modalities = {}
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with le_info_path.open("r", encoding="utf-8") as f:
            le_info = json.load(f)
        for key in self.modality_configs.get("video", {}).modality_keys:
            subkey = key.replace("video.", "")
            original_key = (
                self.lerobot_modality_meta.video[subkey].original_key
                or f"observation.images.{subkey}"
            )
            video_meta = le_info["features"][original_key]
            names = video_meta["names"]
            shape = video_meta["shape"]
            height = int(shape[names.index("height")])
            width = int(shape[names.index("width")])
            channels = int(shape[names.index("channel")])
            fps = float(
                video_meta.get("video_info", {}).get(
                    "video.fps", le_info.get("fps", 1.0)
                )
            )
            if fps <= 0:
                fps = 1.0
            video_modalities[subkey] = VideoMetadata(
                resolution=(width, height),
                channels=channels,
                fps=fps,
            )

        stats = self._load_or_compute_statistics()
        state_modalities = {}
        state_statistics = {}
        for key in self.modality_configs.get("state", {}).modality_keys:
            if not self._is_virtual_state_key(key):
                raise ValueError(
                    f"InteriorGS only supports virtual state keys, got {key}"
                )
            subkey = key.replace("state.", "")
            state_modalities[subkey] = StateActionMetadata(
                shape=(3,),
            )
            state_statistics[subkey] = DatasetStatisticalValues.model_validate(
                stats["state"][subkey]
            )

        action_modalities = {}
        action_statistics = {}
        for key in self.modality_configs.get("action", {}).modality_keys:
            if not self._is_virtual_action_key(key):
                raise ValueError(
                    f"InteriorGS only supports virtual action keys, got {key}"
                )
            subkey = key.replace("action.", "")
            action_modalities[subkey] = StateActionMetadata(
                shape=(3,),
            )
            action_statistics[subkey] = DatasetStatisticalValues.model_validate(
                stats["action"][subkey]
            )

        return DatasetMetadata(
            statistics={
                "state": state_statistics,
                "action": action_statistics,
            },
            modalities={
                "video": video_modalities,
                "state": state_modalities,
                "action": action_modalities,
            },
            embodiment_tag=self.tag,
        )

    @staticmethod
    def _is_virtual_state_key(key: str) -> bool:
        return key == "state.nav_pose"

    @staticmethod
    def _is_virtual_action_key(key: str) -> bool:
        return key == "action.nav_delta"

    def _load_or_compute_statistics(self) -> dict:
        stats_path = self.dataset_path / "meta/interiorgs_nav_stats.json"
        cache_params = self._statistics_cache_params()
        if self.compute_statistics and stats_path.is_file():
            with stats_path.open("r", encoding="utf-8") as f:
                cached_stats = json.load(f)
            if cached_stats.get("_cache_params") == cache_params:
                return cached_stats
            print(f"[InteriorGS] Ignoring stale statistics cache {stats_path}")

        state_key = "nav_pose"
        action_key = "nav_delta"
        state_values = np.zeros((1, 3), dtype=np.float32)
        if self.compute_statistics:
            action_values = self._collect_action_statistics()
        else:
            action_values = np.array(
                [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            )
        stats = {
            "_cache_params": cache_params,
            "state": {state_key: self._stats_from_array(state_values)},
            "action": {action_key: self._stats_from_array(action_values)},
        }
        if not self.compute_statistics:
            return stats

        try:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with stats_path.open("w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4)
        except OSError as exc:
            print(f"[InteriorGS] Could not write statistics cache {stats_path}: {exc}")
        return stats

    def _statistics_cache_params(self) -> dict[str, Any]:
        action_offsets = np.asarray(
            self.modality_configs["action"].delta_indices,
            dtype=np.int64,
        )
        return {
            "action_statistics_version": ACTION_STATISTICS_VERSION,
            "action_statistics_scope": ACTION_STATISTICS_SCOPE,
            "action_delta_indices": action_offsets.tolist(),
            "action_frame_stride": self.action_frame_stride,
            "nav_action_scale": self.nav_action_scale,
            "nav_smooth_window": self.nav_smooth_window,
            "nav_smooth_passes": self.nav_smooth_passes,
        }

    def _collect_action_statistics(self) -> np.ndarray:
        values = []
        episode_ids = self.trajectory_ids
        if self.max_stats_episodes is not None:
            episode_ids = episode_ids[: int(self.max_stats_episodes)]
        eligible_starts_by_episode = self._get_step_filter()
        action_offsets = np.asarray(
            self.modality_configs["action"].delta_indices,
            dtype=np.int64,
        )
        stats_start_stride = max(self.stats_start_stride, 1)
        for episode_index in episode_ids:
            df = pd.read_parquet(self.get_parquet_path(int(episode_index)))
            length = len(df)
            if length < 2:
                continue
            extrinsics = self._get_extrinsics(df)
            max_start = max(0, length - 1 - int(action_offsets.max(initial=0)))
            eligible_starts = np.asarray(
                eligible_starts_by_episode.get(int(episode_index), []),
                dtype=np.int64,
            )
            complete_starts = eligible_starts[
                (eligible_starts >= 0) & (eligible_starts <= max_start)
            ]
            for start in complete_starts[::stats_start_stride]:
                indices = start + action_offsets
                values.append(
                    self._compute_navigation_actions(extrinsics, indices, length)
                )
        if not values:
            return np.array(
                [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            )
        return np.concatenate(values, axis=0).astype(np.float32)

    @staticmethod
    def _stats_from_array(values: np.ndarray) -> dict[str, list[float]]:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        q01 = np.quantile(values, 0.01, axis=0).astype(np.float32)
        q99 = np.quantile(values, 0.99, axis=0).astype(np.float32)
        zero_range = np.isclose(q01, q99)
        if np.any(zero_range):
            span = np.maximum(np.abs(q99), 1.0).astype(np.float32)
            q01 = np.where(zero_range, -span, q01)
            q99 = np.where(zero_range, span, q99)
        return {
            "max": values.max(axis=0).tolist(),
            "min": values.min(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }

    def get_video(
        self, trajectory_id: int, key: str, step_indices: np.ndarray
    ) -> np.ndarray:
        trajectory_index = self.get_trajectory_index(trajectory_id)
        length = int(self.trajectory_lengths[trajectory_index])
        frame_indices = np.clip(step_indices, 0, max(length - 1, 0)).astype(np.int64)
        video_key = key.replace("video.", "")
        return self._read_video_frames(trajectory_id, video_key, frame_indices)

    def get_state(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        if not self._is_virtual_state_key(key):
            raise ValueError(f"InteriorGS only supports state.nav_pose, got {key}")
        return np.zeros((len(step_indices), 3), dtype=np.float32)

    def get_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        if not self._is_virtual_action_key(key):
            raise ValueError(f"InteriorGS only supports action.nav_delta, got {key}")
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        length = len(self.curr_traj_data)
        extrinsics = self._get_extrinsics(self.curr_traj_data)
        return self._compute_navigation_actions(extrinsics, step_indices, length)

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | list[str] | None:
        if modality == "state":
            return self.get_state(trajectory_id, modality, key, step_indices)
        if modality == "action":
            return self.get_action(trajectory_id, modality, key, step_indices)
        if modality == "video":
            return self.get_video(trajectory_id, key, step_indices)
        if modality == "language":
            return self.get_language(trajectory_id, key, step_indices)
        raise ValueError(f"Invalid InteriorGS modality: {modality}")

    def _read_video_frames(
        self,
        episode_index: int,
        video_key: str,
        frame_indices: np.ndarray,
    ) -> np.ndarray:
        video_path = self.get_video_path(episode_index, video_key)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frames = []
        last_index = None
        last_frame = None
        for frame_index in frame_indices:
            frame_index = int(frame_index)
            if last_index == frame_index and last_frame is not None:
                frame = last_frame.copy()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    if last_frame is None:
                        raise RuntimeError(
                            f"Failed to read frame {frame_index} from {video_path}"
                        )
                    frame = last_frame.copy()
                last_index = frame_index
                last_frame = frame.copy()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames, axis=0).astype(np.uint8)

    @staticmethod
    def _get_extrinsics(df: pd.DataFrame) -> np.ndarray:
        extrinsics = []
        for value in df["observation.camera_extrinsic"]:
            array = np.asarray(value)
            if array.dtype == object:
                array = np.stack(value)
            array = np.asarray(array, dtype=np.float32).reshape(4, 4)
            extrinsics.append(array)
        return np.stack(extrinsics, axis=0)

    def _compute_navigation_actions(
        self,
        extrinsics: np.ndarray,
        action_indices: np.ndarray,
        length: int,
    ) -> np.ndarray:
        action_indices = np.asarray(action_indices, dtype=np.int64)
        if len(action_indices) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if len(action_indices) > 1:
            next_step = int(action_indices[-1] - action_indices[-2])
            if next_step <= 0:
                next_step = self.action_frame_stride
        else:
            next_step = self.action_frame_stride
        point_indices_raw = np.concatenate(
            [action_indices, [action_indices[-1] + next_step]]
        )
        valid_mask = (point_indices_raw >= 0) & (point_indices_raw < length)
        valid_count = int(valid_mask.sum())
        point_indices = np.clip(point_indices_raw, 0, max(length - 1, 0)).astype(
            np.int64
        )

        local_points = self._local_nav_points(extrinsics, point_indices)
        local_points = self._smooth_valid_nav_points(local_points, valid_count)
        xyt = self._points_to_frame_xyt(local_points)
        actions = xyt[1:] - xyt[:-1]
        actions[:, 2] = (actions[:, 2] + np.pi) % (2 * np.pi) - np.pi
        return (actions * self.nav_action_scale).astype(np.float32)

    def _local_nav_points(
        self, extrinsics: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        base = extrinsics[int(indices[0])]
        r0 = base[:3, :3].astype(np.float32)
        t0 = base[:3, 3].astype(np.float32)
        tw = extrinsics[indices, :3, 3].astype(np.float32)
        pc = (r0.T @ (tw - t0).T).T
        return (self.CAM_TO_NAV @ pc.T).T.astype(np.float32)

    def _smooth_valid_nav_points(
        self, local_points: np.ndarray, valid_count: int
    ) -> np.ndarray:
        if valid_count <= 0 or valid_count >= local_points.shape[0]:
            return self._smooth_nav_xy(
                local_points,
                window=self.nav_smooth_window,
                passes=self.nav_smooth_passes,
            )

        out = local_points.copy()
        smoothed_prefix = self._smooth_nav_xy(
            out[:valid_count],
            window=self.nav_smooth_window,
            passes=self.nav_smooth_passes,
        )
        out[:valid_count] = smoothed_prefix
        out[valid_count:] = smoothed_prefix[-1]
        return out

    @staticmethod
    def _smooth_nav_xy(
        points: np.ndarray, window: int = 7, passes: int = 2
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        n = points.shape[0]
        if n <= 2 or window <= 1 or passes <= 0:
            return points

        if window == 3:
            kernel = np.array([1, 2, 1], dtype=np.float32)
        elif window == 5:
            kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
        elif window == 7:
            kernel = np.array([1, 3, 6, 7, 6, 3, 1], dtype=np.float32)
        else:
            kernel = np.ones(window, dtype=np.float32)
        kernel /= kernel.sum()

        def smooth_once(p: np.ndarray) -> np.ndarray:
            deltas = p[1:, :2] - p[:-1, :2]
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            dx_pad = np.pad(deltas[:, 0], (pad_left, pad_right), mode="edge")
            dy_pad = np.pad(deltas[:, 1], (pad_left, pad_right), mode="edge")
            dx_smooth = np.convolve(dx_pad, kernel, mode="valid")
            dy_smooth = np.convolve(dy_pad, kernel, mode="valid")

            out = p.copy()
            out[0, :2] = p[0, :2]
            for i in range(1, n):
                out[i, 0] = out[i - 1, 0] + dx_smooth[i - 1]
                out[i, 1] = out[i - 1, 1] + dy_smooth[i - 1]
            out[:, 2] = p[:, 2]
            return out

        out = points.copy()
        for _ in range(passes):
            out = smooth_once(out)
        return out

    @staticmethod
    def _points_to_frame_xyt(local_points: np.ndarray) -> np.ndarray:
        xyt = np.zeros((local_points.shape[0], 3), dtype=np.float32)
        xyt[:, :2] = local_points[:, :2]
        if local_points.shape[0] < 2:
            return xyt

        deltas = local_points[1:, :2] - local_points[:-1, :2]
        init_xy = deltas[0].astype(np.float32)
        if np.linalg.norm(init_xy) < 1e-6:
            init_xy = np.array([1.0, 0.0], dtype=np.float32)

        theta = np.zeros(local_points.shape[0], dtype=np.float32)
        last_theta = 0.0
        for i, delta in enumerate(deltas):
            if np.linalg.norm(delta) < 1e-6:
                theta[i] = last_theta
                continue
            dot_product = float(np.dot(init_xy, delta))
            cross_product = float(np.cross(init_xy, delta))
            last_theta = float(np.arctan2(cross_product, dot_product))
            theta[i] = last_theta
        theta[-1] = last_theta
        xyt[:, 2] = theta
        return xyt
