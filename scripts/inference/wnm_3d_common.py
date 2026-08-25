"""Shared runtime helpers for offline and online WNM-3D inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gammanav.vln.data.schema import DatasetMetadata, EmbodimentTag
from gammanav.vln.model.wnm_3d.inference_policy import WNM3DInferencePolicy


logger = logging.getLogger(__name__)


def configure_torch_dynamo() -> None:
    """Raise Dynamo cache limits for the dynamic causal video shapes."""
    try:
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit),
            800,
        )
        torch._dynamo.config.cache_size_limit = max(
            int(torch._dynamo.config.cache_size_limit),
            800,
        )
    except Exception as exc:
        logger.warning("Could not configure torch dynamo: %s", exc)


def as_numpy(value: Any) -> np.ndarray:
    """Move a tensor to CPU or coerce an array-like value to NumPy."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def batch_get(container: Any, key: str) -> Any:
    """Read a key from dict-like policy outputs or attribute containers."""
    if isinstance(container, dict):
        return container[key]
    try:
        return container[key]
    except (KeyError, TypeError):
        return getattr(container, key)


def config_get(config: Any, key: str, default: Any) -> Any:
    """Read an optional key from an OmegaConf-compatible object."""
    return config[key] if key in config else default


def config_select(config: Any, path: str, default: Any) -> Any:
    """Read an optional dotted path from an OmegaConf-compatible object."""
    current = config
    for key in path.split("."):
        try:
            if key not in current:
                return default
            current = current[key]
        except (KeyError, TypeError):
            return default
    return current


def resolve_nav_action_scale(
    config: Any,
    *,
    override: float | None = None,
    default: float = 1.0,
) -> float:
    """Resolve the physical-action scale from CLI or checkpoint training config."""
    value = (
        override
        if override is not None
        else config_select(
            config,
            "train_dataset.dataset_kwargs.nav_action_scale",
            default,
        )
    )
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"nav_action_scale must be a positive finite number, got {scale}."
        )
    return scale


def reset_causal_state(policy: WNM3DInferencePolicy) -> None:
    """Clear all request-sequence state held by the causal action head."""
    action_head = policy.trained_model.action_head
    action_head.current_start_frame = 0
    for name in (
        "kv_cache1",
        "kv_cache_neg",
        "crossattn_cache",
        "crossattn_cache_neg",
        "ys",
        "clip_feas",
        "last_language",
        "language",
    ):
        if hasattr(action_head, name):
            setattr(action_head, name, None)


def load_checkpoint_metadata(
    model_path: Path,
    embodiment_tag: EmbodimentTag,
) -> DatasetMetadata:
    """Load one embodiment's dataset metadata from a WNM-3D checkpoint."""
    metadata_path = model_path / "experiment_cfg" / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as file:
        all_metadata = json.load(file)
    return DatasetMetadata.model_validate(all_metadata[embodiment_tag.value])


def restore_policy_video_metadata(
    policy: WNM3DInferencePolicy,
    metadata: DatasetMetadata,
    video_key: str,
) -> list[str]:
    """Restore raw video metadata without changing action normalization."""
    reset_transforms = []
    for transform in getattr(policy.eval_transform, "transforms", []):
        if video_key not in getattr(transform, "apply_to", []):
            continue
        if not hasattr(transform, "original_resolutions"):
            continue
        if transform.__class__.__name__ == "VideoCrop":
            transform.height = None
            transform.width = None
        transform.set_metadata(metadata)
        transform.eval()
        reset_transforms.append(transform.__class__.__name__)
    return reset_transforms


def set_causal_eval_state_horizon(policy: WNM3DInferencePolicy) -> list[str]:
    """Use one state register for each causal inference block."""
    updated_transforms = []
    action_head = policy.trained_model.action_head
    state_horizon = int(getattr(action_head, "num_state_per_block", 1))
    for transform in getattr(policy.eval_transform, "transforms", []):
        if not hasattr(transform, "state_horizon"):
            continue
        transform.state_horizon = state_horizon
        updated_transforms.append(
            f"{transform.__class__.__name__}.state_horizon={state_horizon}"
        )
    return updated_transforms
