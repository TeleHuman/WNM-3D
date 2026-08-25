"""Build the self-contained action-normalization manifest for WNM-3D checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping

from omegaconf import OmegaConf


ACTION_NORMALIZATION_FILENAME = "action_normalization.json"
ACTION_NORMALIZATION_SCHEMA = "wnm_3d_action_normalization_v1"
INTERIORGS_EMBODIMENT = "interiorgs"
INTERIORGS_ACTION_KEY = "action.nav_delta"
INTERIORGS_CHANNEL_ORDER = ("dx_m", "dy_m", "dyaw_rad")


def _required_config_value(config: Any, path: str) -> Any:
    value = OmegaConf.select(config, path, default=None)
    if value is None:
        raise ValueError(f"Training config is missing required value {path!r}")
    return value


def _normalization_mode(
    config: Any,
    *,
    embodiment: str,
    action_key: str,
) -> str:
    transforms = _required_config_value(config, f"transforms.{embodiment}.transforms")
    for transform in transforms:
        modes = transform.get("normalization_modes")
        if modes is not None and action_key in modes:
            return str(modes[action_key])
    raise ValueError(
        f"Training config has no normalization mode for {action_key!r} "
        f"under transforms.{embodiment}.transforms"
    )


def _finite_float_list(value: Any, *, name: str) -> list[float]:
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a sequence of numbers") from exc
    if not values or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite numbers")
    return values


def build_action_normalization_manifest(
    config: Any,
    metadata_by_embodiment: Mapping[str, Any],
    *,
    embodiment: str = INTERIORGS_EMBODIMENT,
    action_key: str = INTERIORGS_ACTION_KEY,
) -> dict[str, Any]:
    """Build a checkpoint manifest from the exact training config and statistics."""
    try:
        metadata = metadata_by_embodiment[embodiment]
        modality, subkey = action_key.split(".", 1)
        statistics = metadata["statistics"][modality][subkey]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint metadata is missing statistics for {embodiment}.{action_key}"
        ) from exc

    q01 = _finite_float_list(statistics.get("q01"), name=f"{action_key}.q01")
    q99 = _finite_float_list(statistics.get("q99"), name=f"{action_key}.q99")
    if len(q01) != len(q99):
        raise ValueError(f"{action_key} q01 and q99 dimensions do not match")
    if len(q01) != len(INTERIORGS_CHANNEL_ORDER):
        raise ValueError(
            f"{action_key} must have {len(INTERIORGS_CHANNEL_ORDER)} decoded "
            f"dimensions, got {len(q01)}"
        )

    nav_action_scale = float(
        _required_config_value(config, "train_dataset.dataset_kwargs.nav_action_scale")
    )
    if not math.isfinite(nav_action_scale) or nav_action_scale <= 0:
        raise ValueError(
            "train_dataset.dataset_kwargs.nav_action_scale must be positive "
            f"and finite, got {nav_action_scale}"
        )

    normalization_mode = _normalization_mode(
        config,
        embodiment=embodiment,
        action_key=action_key,
    )
    if normalization_mode != "q99":
        raise ValueError(
            f"Unsupported normalization mode {normalization_mode!r} for {action_key}"
        )

    return {
        "schema": ACTION_NORMALIZATION_SCHEMA,
        "embodiment": embodiment,
        "action_key": action_key,
        "channel_order": list(INTERIORGS_CHANNEL_ORDER),
        "model_action_dim": int(_required_config_value(config, "max_action_dim")),
        "action_horizon_per_chunk": int(
            _required_config_value(config, "action_horizon")
        ),
        "decoded_action_dims": len(q01),
        "normalization_mode": normalization_mode,
        "normalized_reference_range": [-1.0, 1.0],
        "q01": q01,
        "q99": q99,
        "nav_action_scale": nav_action_scale,
        "physical_q01_after_scale": [value / nav_action_scale for value in q01],
        "physical_q99_after_scale": [value / nav_action_scale for value in q99],
        "decode_formula": (
            "delta = (((action_norm[..., :3] + 1) * 0.5) * "
            "(q99 - q01) + q01) / nav_action_scale"
        ),
        "encode_formula": (
            "action_norm = 2 * ((delta * nav_action_scale) - q01) / (q99 - q01) - 1"
        ),
        "decode_clamps_normalized_action": False,
    }


def write_action_normalization_manifest(
    checkpoint_dir: str | Path,
    config: Any,
    metadata_by_embodiment: Mapping[str, Any],
    *,
    embodiment: str = INTERIORGS_EMBODIMENT,
    action_key: str = INTERIORGS_ACTION_KEY,
) -> Path:
    """Atomically write ``action_normalization.json`` in a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = checkpoint_dir / ACTION_NORMALIZATION_FILENAME
    manifest = build_action_normalization_manifest(
        config,
        metadata_by_embodiment,
        embodiment=embodiment,
        action_key=action_key,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=checkpoint_dir,
        prefix=f".{ACTION_NORMALIZATION_FILENAME}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
        temporary_path = Path(file.name)
    temporary_path.replace(output_path)
    return output_path


def write_action_normalization_manifest_from_checkpoint(
    checkpoint_dir: str | Path,
) -> Path:
    """Create a manifest for an existing checkpoint's saved config and metadata."""
    checkpoint_dir = Path(checkpoint_dir)
    exp_cfg_dir = checkpoint_dir / "experiment_cfg"
    config = OmegaConf.load(exp_cfg_dir / "conf.yaml")
    with (exp_cfg_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata_by_embodiment = json.load(file)
    return write_action_normalization_manifest(
        checkpoint_dir,
        config,
        metadata_by_embodiment,
    )
