"""Inference wrapper for the released WNM-3D causal video-action policy."""

import importlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from gammanav.vln.data.schema import DatasetMetadata, EmbodimentTag
from gammanav.vln.data.transform import ComposedModalityTransform


class PolicyBatch(SimpleNamespace):
    """Minimal attribute container used by the WNM-3D inference scripts."""


def _update_tokenizer_path_in_config(cfg, new_path: str) -> None:
    """Update tokenizer paths inside the transform configuration."""
    if isinstance(cfg, DictConfig):
        if "tokenizer_path" in cfg:
            cfg.tokenizer_path = new_path
        if "transforms" not in cfg:
            return
        transforms = cfg.transforms
        if isinstance(transforms, DictConfig):
            for transform in transforms.values():
                _update_tokenizer_path_in_config(transform, new_path)
        elif isinstance(transforms, ListConfig):
            for transform in transforms:
                _update_tokenizer_path_in_config(transform, new_path)
    elif isinstance(cfg, ListConfig):
        for value in cfg:
            _update_tokenizer_path_in_config(value, new_path)


class WNM3DInferencePolicy:
    """Load a WNM-3D checkpoint and run causal joint video-action inference."""

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        device: int | str,
        tokenizer_path_override: str | None = None,
        device_mesh: DeviceMesh | None = None,
    ):
        self.embodiment_tag = embodiment_tag
        self.model_path = model_path
        self.device = device
        self.rank = dist.get_rank()

        model_dir = Path(model_path)
        exp_cfg_dir = model_dir / "experiment_cfg"
        train_cfg = OmegaConf.load(exp_cfg_dir / "conf.yaml")
        if tokenizer_path_override is not None:
            _update_tokenizer_path_in_config(train_cfg, tokenizer_path_override)
        self.train_cfg = train_cfg

        model_target = train_cfg.model._target_
        module_name, class_name = model_target.rsplit(".", 1)
        model_class = getattr(importlib.import_module(module_name), class_name)
        if train_cfg.get("save_lora_only", False):
            print("Loading WNM-3D LoRA checkpoint")
            model = model_class.load_lora(model_path)
        else:
            print("Loading WNM-3D full checkpoint")
            model = model_class.from_pretrained(model_path)

        model.eval()
        model.requires_grad_(False)
        if model.action_head.train_architecture == "lora":
            print("Merging LoRA weights")
            model.action_head.model = model.action_head.model.merge_and_unload()

        self.eval_bf16 = bool(train_cfg.get("eval_bf16", False))
        if self.eval_bf16:
            model = model.to(dtype=torch.bfloat16)
        model.to(device=device)
        model.post_initialize()

        if device_mesh is not None:
            model.parallelize(device_mesh=device_mesh)
        torch.cuda.empty_cache()
        self.trained_model = model

        with (exp_cfg_dir / "metadata.json").open("r", encoding="utf-8") as file:
            all_metadata = json.load(file)
        metadata = DatasetMetadata.model_validate(
            all_metadata[self.embodiment_tag.value]
        )

        action_head_config = self.trained_model.action_head.config
        target_height = getattr(action_head_config, "target_video_height", None)
        target_width = getattr(action_head_config, "target_video_width", None)
        if target_height is not None and target_width is not None:
            for video_metadata in metadata.modalities.video.values():
                video_metadata.resolution = (int(target_width), int(target_height))

        if self.embodiment_tag.value not in train_cfg.transforms:
            raise KeyError(
                f"Missing transform for {self.embodiment_tag.value!r}; "
                f"available={list(train_cfg.transforms)}"
            )
        eval_transform = instantiate(train_cfg.transforms[self.embodiment_tag.value])
        if not isinstance(eval_transform, ComposedModalityTransform):
            raise TypeError(
                f"Expected ComposedModalityTransform, got {type(eval_transform)!r}"
            )
        eval_transform.set_metadata(metadata)
        eval_transform.eval()
        self.eval_transform = eval_transform

        if self.embodiment_tag.value in train_cfg.modality_configs:
            self.modality_configs = instantiate(
                train_cfg.modality_configs[self.embodiment_tag.value]
            )
        else:
            self.modality_configs = instantiate(train_cfg.modality_configs)

    def apply(self, batch: PolicyBatch) -> PolicyBatch:
        """Normalize and format one policy observation batch."""
        batch.normalized_obs = self.eval_transform(batch.obs)
        return batch

    def unapply(self, batch: PolicyBatch) -> PolicyBatch:
        """Convert normalized model actions back to navigation actions."""
        batch.act = self.eval_transform.unapply(
            dict(action=batch.normalized_action.cpu())
        )
        return batch

    def forward(self, batch: PolicyBatch, **kwargs):
        """Alias for the released causal inference path."""
        return self.lazy_joint_forward_causal(batch, **kwargs)

    def lazy_joint_forward_causal(
        self,
        batch: PolicyBatch,
        video: torch.Tensor | None = None,
        latent_video: torch.Tensor | None = None,
        video_only: bool = False,
        **kwargs,
    ):
        transform_start = time.perf_counter()
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        normalized_input = self.apply(batch).normalized_obs
        transform_time = time.perf_counter() - transform_start
        if video is not None:
            for key in normalized_input:
                if "images" in key:
                    normalized_input[key] = video

        for key, value in normalized_input.items():
            if (
                torch.is_tensor(value)
                and value.dtype == torch.float32
                and self.eval_bf16
            ):
                normalized_input[key] = value.to(dtype=torch.bfloat16)

        model_start = time.perf_counter()
        with torch.inference_mode():
            model_pred = self.trained_model.lazy_joint_video_action_causal(
                normalized_input, latent_video=latent_video
            )
        model_time = time.perf_counter() - model_start

        untransform_start = time.perf_counter()
        normalized_action = model_pred["action_pred"].float()
        if video_only:
            result = PolicyBatch(normalized_action=normalized_action)
        else:
            result = self.unapply(PolicyBatch(normalized_action=normalized_action))
        if not is_batched and hasattr(result, "act"):
            result.act = squeeze_dict_values(result.act)
        untransform_time = time.perf_counter() - untransform_start

        if self.rank == 0:
            total_time = transform_time + model_time + untransform_time
            print(
                f"Inference Time: Total {total_time:.3f} seconds, "
                f"Transform: {transform_time:.3f} seconds, "
                f"Model: {model_time:.3f} seconds, "
                f"Untransform: {untransform_time:.3f} seconds"
            )
        return result, model_pred["video_pred"]

    @staticmethod
    def _check_state_is_batched(obs: dict[str, Any]) -> bool:
        for key, value in obs.items():
            if "state" in key and len(value.shape) < 3:
                return False
        return True


def unsqueeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """Add a batch dimension to one WNM-3D observation."""
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            result[key] = np.expand_dims(value, axis=0)
        elif isinstance(value, list):
            result[key] = np.array(value)
        elif isinstance(value, torch.Tensor):
            result[key] = value.unsqueeze(0)
        elif isinstance(value, str):
            result[key] = np.array([value])
        else:
            result[key] = value
    return result


def squeeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """Remove the leading singleton batch dimension."""
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            result[key] = np.squeeze(value, axis=0)
        elif isinstance(value, torch.Tensor):
            result[key] = value.squeeze(0)
        elif isinstance(value, list) and len(value) == 1:
            result[key] = value[0]
        else:
            result[key] = value
    return result
