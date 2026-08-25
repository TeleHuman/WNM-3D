from dataclasses import dataclass, field
import logging
from typing import Tuple

from hydra.utils import instantiate
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3

logger = logging.getLogger(__name__)


@dataclass
class WNM3DConfig(PretrainedConfig):
    model_type = "wnm_3d"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        """Serialize full checkpoints as self-contained WNM-3D models."""
        output = super().to_dict()
        action_head_cfg = output.get("action_head_cfg")
        if isinstance(action_head_cfg, dict):
            action_head_inner = action_head_cfg.get("config", action_head_cfg)
            if isinstance(action_head_inner, dict):
                action_head_inner["skip_component_loading"] = True
        return output


class WNM3DModel(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = WNM3DConfig
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: WNM3DConfig,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)
        super().__init__(config)
        self.backbone = instantiate(config.backbone_cfg)
        self.action_head = instantiate(config.action_head_cfg)
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def validate_inputs(self, inputs):
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] % self.action_horizon == 0
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):

        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_head_outputs

    def lazy_joint_video_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action(
            backbone_outputs, action_inputs
        )
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def lazy_joint_video_action_causal(
        self,
        inputs: dict,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action(
            backbone_outputs, action_inputs, latent_video=latent_video
        )
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(
            to_device_with_maybe_dtype, backbone_inputs
        )
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def load_lora(cls, pretrained_model_name_or_path: str):
        from safetensors.torch import load_file
        import os
        import json

        logger.info("Loading LoRA checkpoint from %s", pretrained_model_name_or_path)

        # Check for different checkpoint formats
        safetensors_path = os.path.join(
            pretrained_model_name_or_path, "model.safetensors"
        )
        safetensors_index_path = os.path.join(
            pretrained_model_name_or_path, "model.safetensors.index.json"
        )

        state_dict = {}
        if os.path.exists(safetensors_index_path):
            # Handle sharded safetensors
            logger.info(
                "Loading sharded safetensors using index: %s",
                safetensors_index_path,
            )

            with open(safetensors_index_path, "r") as f:
                index = json.load(f)

            # Load each shard
            for shard_file in set(index["weight_map"].values()):
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                logger.info("Loading shard: %s", shard_path)
                shard_state_dict = load_file(shard_path)
                state_dict.update(shard_state_dict)

        elif os.path.exists(safetensors_path):
            # Handle single safetensors file
            logger.info("Loading weights from safetensors: %s", safetensors_path)
            state_dict.update(load_file(safetensors_path))

        # Load config
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        logger.info("Loading model configuration: %s", config_path)
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = WNM3DConfig(**config_dict)
        logger.info("Building %s", cls.__name__)

        # Disable defer_lora_injection so LoRA layers are created during init,
        # matching the PEFT key hierarchy (base_model.model.*) in the checkpoint.
        ah_cfg = config.action_head_cfg
        inner = (
            ah_cfg.get("config", ah_cfg)
            if isinstance(ah_cfg.get("config"), dict)
            else ah_cfg
        )
        if "defer_lora_injection" in inner:
            inner["defer_lora_injection"] = False
            logger.info("Disabled defer_lora_injection for LoRA loading")
        # Enable component loading so DiT base weights are loaded from pretrained
        if "skip_component_loading" in inner:
            inner["skip_component_loading"] = False
            logger.info("Disabled skip_component_loading for LoRA loading")

        # Instantiate model (LoRA layers now exist from init)
        model = cls(config)

        # Remove .base_layer from keys if present
        has_base_layer = any(".base_layer." in key for key in state_dict.keys())
        if has_base_layer:
            logger.info("Removing '.base_layer' from state-dict keys")
            state_dict = {
                k.replace(".base_layer.", "."): v for k, v in state_dict.items()
            }

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            logger.warning(
                "Missing keys when loading pretrained weights: count=%d sample=%s",
                len(missing_keys),
                missing_keys[:8],
            )
        if unexpected_keys:
            logger.warning(
                "Unexpected keys when loading pretrained weights: count=%d sample=%s",
                len(unexpected_keys),
                unexpected_keys[:8],
            )

        logger.info("Successfully loaded pretrained weights into %s", cls.__name__)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: WNM3DConfig | dict | None = None,
    ):
        import gc
        import json
        import os

        from safetensors.torch import load_file

        safetensors_path = os.path.join(
            pretrained_model_name_or_path, "model.safetensors"
        )
        safetensors_index_path = os.path.join(
            pretrained_model_name_or_path, "model.safetensors.index.json"
        )

        if config is None:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = WNM3DConfig(**json.load(config_file))
        elif isinstance(config, dict):
            config = WNM3DConfig(**config)

        action_head_cfg = config.action_head_cfg
        action_head_inner = action_head_cfg.get("config", action_head_cfg)
        if isinstance(action_head_inner, dict):
            action_head_inner["defer_lora_injection"] = False
            action_head_inner["skip_component_loading"] = True

        logger.info("Building %s from %s", cls.__name__, pretrained_model_name_or_path)
        model = cls(config)

        loaded_keys: set[str] = set()
        unexpected_keys: set[str] = set()

        def load_state_dict_chunk(state_dict: dict[str, torch.Tensor]) -> None:
            normalized_state_dict = {
                key.replace(".base_layer.", "."): value
                for key, value in state_dict.items()
            }
            result = model.load_state_dict(normalized_state_dict, strict=False)
            loaded_keys.update(normalized_state_dict)
            unexpected_keys.update(result.unexpected_keys)

        if os.path.exists(safetensors_index_path):
            with open(safetensors_index_path, "r", encoding="utf-8") as index_file:
                index = json.load(index_file)
            shard_files = sorted(set(index["weight_map"].values()))
            for shard_number, shard_file in enumerate(shard_files, start=1):
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                logger.info(
                    "Loading WNM-3D shard %d/%d: %s",
                    shard_number,
                    len(shard_files),
                    shard_file,
                )
                shard_state_dict = load_file(shard_path, device="cpu")
                load_state_dict_chunk(shard_state_dict)
                del shard_state_dict
                gc.collect()
        elif os.path.exists(safetensors_path):
            logger.info("Loading WNM-3D weights: %s", safetensors_path)
            state_dict = load_file(safetensors_path, device="cpu")
            load_state_dict_chunk(state_dict)
            del state_dict
            gc.collect()
        else:
            raise FileNotFoundError(
                f"No weights found at {pretrained_model_name_or_path!r}; expected "
                "model.safetensors or model.safetensors.index.json"
            )

        missing_keys = set(model.state_dict()) - loaded_keys
        if missing_keys:
            missing_sample = sorted(missing_keys)[:8]
            logger.warning(
                "Missing keys when loading WNM-3D weights: count=%d sample=%s",
                len(missing_keys),
                missing_sample,
            )
        if unexpected_keys:
            unexpected_sample = sorted(unexpected_keys)[:8]
            logger.warning(
                "Unexpected keys when loading WNM-3D weights: count=%d sample=%s",
                len(unexpected_keys),
                unexpected_sample,
            )

        logger.info(
            "Loaded %d tensor keys into %s from %s",
            len(loaded_keys),
            cls.__name__,
            pretrained_model_name_or_path,
        )
        return model

    def post_initialize(self):
        self.action_head.post_initialize()

    def parallelize(self, device_mesh: DeviceMesh):
        self.action_head.parallelize(device_mesh=device_mesh)


# register
AutoConfig.register("wnm_3d", WNM3DConfig)
AutoModel.register(WNM3DConfig, WNM3DModel)
