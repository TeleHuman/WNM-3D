"""Offline open-loop InteriorGS evaluation for a WNM-3D checkpoint.

The evaluator samples one step from each selected LeRobot trajectory, runs the
same causal policy used by the online server, compares the first action block
with ground truth, and writes per-episode predictions plus aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from gammanav.vln.data.dataset.lerobot_sharded import (
    ShardedLeRobotSingleActionChunkDatasetInteriorGS,
)
from gammanav.vln.data.schema import EmbodimentTag
from gammanav.vln.model.wnm_3d.inference_policy import (
    PolicyBatch,
    WNM3DInferencePolicy,
)
from wnm_3d_common import (
    as_numpy,
    batch_get,
    config_get,
    configure_torch_dynamo,
    load_checkpoint_metadata,
    reset_causal_state,
    resolve_nav_action_scale,
    restore_policy_video_metadata,
    set_causal_eval_state_horizon,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _init_device_mesh() -> DeviceMesh | None:
    """Initialize torch.distributed and return a CUDA device mesh when present."""
    device_id = None
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device_id = torch.device("cuda", local_rank)

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, device_id=device_id)

    world_size = dist.get_world_size()
    if not torch.cuda.is_available():
        return None

    return init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )


def _clipped_indices(indices: np.ndarray, length: int) -> np.ndarray:
    return np.clip(indices.astype(np.int64), 0, max(length - 1, 0))


def _visible_history_target_indices(
    step: int,
    num_frames: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the causal visible history and repeated-current target indices."""
    history = np.arange(step - num_frames + 1, step + 1, dtype=np.int64)
    target = np.full(num_frames, step, dtype=np.int64)
    return _clipped_indices(history, length), _clipped_indices(target, length)


def _make_dataset(
    policy: WNM3DInferencePolicy,
    dataset_path: str,
    compute_statistics: bool,
    nav_action_scale: float,
) -> ShardedLeRobotSingleActionChunkDatasetInteriorGS:
    config = policy.train_cfg
    return ShardedLeRobotSingleActionChunkDatasetInteriorGS(
        dataset_path=dataset_path,
        modality_configs=policy.modality_configs,
        embodiment_tag=EmbodimentTag.INTERIORGS,
        transforms=None,
        max_chunk_size=int(config_get(config, "max_chunk_size", 4)),
        num_frames=int(config_get(config, "num_frames", 33)),
        action_horizon=int(config_get(config, "action_horizon", 8)),
        state_horizon=int(config_get(config, "state_horizon", 4)),
        num_frame_per_block=int(config_get(config, "num_frame_per_block", 2)),
        num_action_per_block=int(config_get(config, "num_action_per_block", 8)),
        num_state_per_block=int(config_get(config, "num_state_per_block", 1)),
        vae_temporal_stride=int(config_get(config, "vae_temporal_stride", 4)),
        video_frame_stride=int(config_get(config, "video_frame_stride", 1)),
        action_frame_stride=int(config_get(config, "action_frame_stride", 1)),
        nav_action_scale=nav_action_scale,
        nav_smooth_window=int(config_get(config, "nav_smooth_window", 7)),
        nav_smooth_passes=int(config_get(config, "nav_smooth_passes", 2)),
        compute_statistics=compute_statistics,
    )


def _make_observation(
    dataset: ShardedLeRobotSingleActionChunkDatasetInteriorGS,
    episode_id: int,
    step: int,
    length: int,
    video_key: str,
    state_key: str,
    language_key: str,
    num_frames: int,
) -> dict[str, Any]:
    history_indices, target_indices = _visible_history_target_indices(
        step=step,
        num_frames=num_frames,
        length=length,
    )
    indices = {
        video_key: np.concatenate([history_indices, target_indices]),
        state_key: np.array([step], dtype=np.int64),
        language_key: np.array([step], dtype=np.int64),
    }
    observation = dataset.get_step_data(episode_id, indices)
    observation["target_prefix_frames"] = np.asarray(1, dtype=np.int64)
    return observation


def _get_ground_truth_action(
    dataset: ShardedLeRobotSingleActionChunkDatasetInteriorGS,
    episode_id: int,
    step: int,
    length: int,
    action_key: str,
    action_horizon: int,
) -> np.ndarray:
    indices = {
        action_key: _clipped_indices(
            step + np.arange(action_horizon, dtype=np.int64),
            length,
        )
    }
    data = dataset.get_step_data(episode_id, indices)
    return np.asarray(data[action_key], dtype=np.float32)


def _select_episode_ids(
    dataset: ShardedLeRobotSingleActionChunkDatasetInteriorGS,
    episodes: str | None,
    start_episode: int,
    num_episodes: int,
) -> list[int]:
    all_ids = [int(value) for value in dataset.trajectory_ids]
    if not episodes:
        return all_ids[start_episode : start_episode + num_episodes]

    requested = [int(value.strip()) for value in episodes.split(",") if value.strip()]
    available = set(all_ids)
    missing = [episode_id for episode_id in requested if episode_id not in available]
    if missing:
        raise ValueError(f"Requested episode ids not found: {missing[:10]}")
    return requested


def _evaluate_prediction(
    predicted_scaled: np.ndarray,
    ground_truth_scaled: np.ndarray,
    episode_id: int,
    step: int,
    action_horizon: int,
    nav_action_scale: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted_scaled = np.asarray(predicted_scaled)
    if predicted_scaled.ndim == 3 and predicted_scaled.shape[0] == 1:
        predicted_scaled = predicted_scaled[0]
    if predicted_scaled.ndim == 1:
        predicted_scaled = predicted_scaled.reshape(1, -1)

    ground_truth_scaled = np.asarray(ground_truth_scaled)
    count = min(
        predicted_scaled.shape[0],
        ground_truth_scaled.shape[0],
        action_horizon,
    )
    dimensions = min(
        predicted_scaled.shape[-1],
        ground_truth_scaled.shape[-1],
        3,
    )
    predicted_scaled = predicted_scaled[:count, :dimensions]
    ground_truth_scaled = ground_truth_scaled[:count, :dimensions]
    predicted_physical = predicted_scaled / nav_action_scale
    ground_truth_physical = ground_truth_scaled / nav_action_scale
    error = predicted_physical - ground_truth_physical

    metrics = {
        "episode_id": int(episode_id),
        "step": int(step),
        "num_actions": int(count),
        "mae_physical": float(np.mean(np.abs(error))),
        "mse_physical": float(np.mean(error**2)),
        "rmse_physical": float(np.sqrt(np.mean(error**2))),
        "mae_scaled": float(np.mean(np.abs(predicted_scaled - ground_truth_scaled))),
        "mse_scaled": float(np.mean((predicted_scaled - ground_truth_scaled) ** 2)),
    }
    return (
        metrics,
        predicted_scaled,
        ground_truth_scaled,
        predicted_physical,
        ground_truth_physical,
    )


def _write_results(
    output_dir: Path,
    rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    model_path: Path,
    dataset_path: Path,
    episode_ids: list[int],
    nav_action_scale: float,
    num_frames: int,
    action_horizon: int,
) -> None:
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")

    summary: dict[str, Any] = {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "episodes": episode_ids,
        "mode": "open_loop_block1",
        "num_predictions": len(metric_rows),
        "nav_action_scale": nav_action_scale,
        "target_frames": num_frames,
        "action_horizon": action_horizon,
    }
    if metric_rows:
        for key in (
            "mae_physical",
            "mse_physical",
            "rmse_physical",
            "mae_scaled",
            "mse_scaled",
        ):
            summary[key] = float(np.mean([row[key] for row in metric_rows]))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved predictions: %s", predictions_path)
    logger.info("Saved summary    : %s", summary_path)
    logger.info("Summary          : %s", summary)


def run(args: argparse.Namespace) -> None:
    if args.nav_action_scale is not None and (
        not np.isfinite(args.nav_action_scale) or args.nav_action_scale <= 0
    ):
        raise ValueError("--nav-action-scale must be a positive finite number")
    if args.start_episode < 0 or args.num_episodes <= 0 or args.start_step < 0:
        raise ValueError(
            "--start-episode and --start-step must be non-negative; "
            "--num-episodes must be positive"
        )

    configure_torch_dynamo()
    device_mesh = _init_device_mesh()
    rank = dist.get_rank()

    model_path = Path(args.model_path)
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir or (model_path / "inference" / "interiorgs_nav"))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Model path     : %s", model_path)
        logger.info("Dataset path   : %s", dataset_path)
        logger.info("Output dir     : %s", output_dir)

    try:
        policy = WNM3DInferencePolicy(
            embodiment_tag=EmbodimentTag.INTERIORGS,
            model_path=str(model_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            device_mesh=device_mesh,
        )
        policy.trained_model.eval()

        nav_action_scale = resolve_nav_action_scale(
            policy.train_cfg,
            override=args.nav_action_scale,
            default=1.0,
        )

        dataset = _make_dataset(
            policy=policy,
            dataset_path=str(dataset_path),
            compute_statistics=not args.skip_dataset_statistics,
            nav_action_scale=nav_action_scale,
        )

        video_key = policy.modality_configs.video.modality_keys[0]
        state_key = policy.modality_configs.state.modality_keys[0]
        action_key = policy.modality_configs.action.modality_keys[0]
        language_key = policy.modality_configs.language.modality_keys[0]
        checkpoint_metadata = load_checkpoint_metadata(
            model_path,
            EmbodimentTag.INTERIORGS,
        )
        reset_video_transforms = restore_policy_video_metadata(
            policy=policy,
            metadata=checkpoint_metadata,
            video_key=video_key,
        )
        state_transforms = set_causal_eval_state_horizon(policy)

        action_horizon = int(policy.trained_model.action_head.action_horizon)
        num_frames = int(policy.trained_model.action_head.num_frames)
        episode_ids = _select_episode_ids(
            dataset,
            episodes=args.episodes,
            start_episode=args.start_episode,
            num_episodes=args.num_episodes,
        )

        if rank == 0:
            logger.info("Video key       : %s", video_key)
            logger.info("State key       : %s", state_key)
            logger.info("Action key      : %s", action_key)
            logger.info("Language key    : %s", language_key)
            logger.info("Video metadata  : %s", checkpoint_metadata.modalities.video)
            logger.info("Reset transforms: %s", reset_video_transforms)
            logger.info("State transforms: %s", state_transforms)
            logger.info("Action horizon  : %d", action_horizon)
            logger.info("Nav action scale: %g", nav_action_scale)
            logger.info("Target frames   : %d", num_frames)
            logger.info("Mode            : open-loop block 1")
            logger.info("Episodes        : %s", episode_ids)

        rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        for episode_id in episode_ids:
            trajectory_index = dataset.get_trajectory_index(episode_id)
            length = int(dataset.trajectory_lengths[trajectory_index])
            if length < 2:
                if rank == 0:
                    logger.warning(
                        "Skipping episode %s with length %d",
                        episode_id,
                        length,
                    )
                continue

            reset_causal_state(policy)
            step = min(max(int(args.start_step), 0), length - 2)
            history_indices, target_indices = _visible_history_target_indices(
                step=step,
                num_frames=num_frames,
                length=length,
            )
            observation = _make_observation(
                dataset=dataset,
                episode_id=episode_id,
                step=step,
                length=length,
                video_key=video_key,
                state_key=state_key,
                language_key=language_key,
                num_frames=num_frames,
            )

            dist.barrier()
            with torch.no_grad():
                result_batch, _video_prediction = policy.lazy_joint_forward_causal(
                    PolicyBatch(obs=observation)
                )
            dist.barrier()

            predicted_scaled = as_numpy(batch_get(result_batch.act, action_key)).astype(
                np.float32
            )
            ground_truth_scaled = _get_ground_truth_action(
                dataset=dataset,
                episode_id=episode_id,
                step=step,
                length=length,
                action_key=action_key,
                action_horizon=action_horizon,
            )
            (
                metrics,
                predicted_scaled,
                ground_truth_scaled,
                predicted_physical,
                ground_truth_physical,
            ) = _evaluate_prediction(
                predicted_scaled=predicted_scaled,
                ground_truth_scaled=ground_truth_scaled,
                episode_id=episode_id,
                step=step,
                action_horizon=action_horizon,
                nav_action_scale=nav_action_scale,
            )
            metric_rows.append(metrics)
            rows.append(
                {
                    **metrics,
                    "history_indices": history_indices.tolist(),
                    "target_indices": target_indices.tolist(),
                    "target_prefix_frames": 1,
                    "pred_action_scaled": predicted_scaled.tolist(),
                    "gt_action_scaled": ground_truth_scaled.tolist(),
                    "pred_action_physical": predicted_physical.tolist(),
                    "gt_action_physical": ground_truth_physical.tolist(),
                }
            )

            if rank == 0:
                logger.info(
                    "episode=%s step=%s mae_phys=%.6f rmse_phys=%.6f",
                    episode_id,
                    step,
                    metrics["mae_physical"],
                    metrics["rmse_physical"],
                )

        if rank == 0:
            _write_results(
                output_dir=output_dir,
                rows=rows,
                metric_rows=metric_rows,
                model_path=model_path,
                dataset_path=dataset_path,
                episode_ids=episode_ids,
                nav_action_scale=nav_action_scale,
                num_frames=num_frames,
                action_horizon=action_horizon,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Checkpoint directory.")
    parser.add_argument(
        "--dataset-path",
        default="./data/interiorgs_lerobot_seen",
        help="LeRobot-format InteriorGS dataset path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save predictions.jsonl and summary.json.",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated episode ids; overrides episode range options.",
    )
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument(
        "--nav-action-scale",
        type=float,
        default=None,
        help=(
            "Override the checkpoint's train_dataset.dataset_kwargs."
            "nav_action_scale (missing checkpoint value defaults to 1.0)."
        ),
    )
    parser.add_argument(
        "--skip-dataset-statistics",
        action="store_true",
        help="Do not recompute InteriorGS statistics when constructing the dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
