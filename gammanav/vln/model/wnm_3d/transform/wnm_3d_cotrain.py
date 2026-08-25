import ast
import html
import os
import random
from typing import Any, Dict, List, Optional

import ftfy
from einops import rearrange
import numpy as np
from pydantic import Field, PrivateAttr
import regex as re
import torch
import tree
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from gammanav.vln.data.schema import DatasetMetadata, EmbodimentTag
from gammanav.vln.data.transform.base import InvertibleModalityTransform
from gammanav.vln.model.wnm_3d.transform.common import formalize_language


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    return html.unescape(html.unescape(text)).strip()


def whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class HuggingfaceTokenizer:
    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        if clean not in (None, "whitespace"):
            raise ValueError(f"Unsupported text cleaning mode: {clean}")
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        load_kwargs = dict(kwargs)
        if os.path.isdir(name):
            load_kwargs.setdefault("local_files_only", True)
        self.tokenizer = AutoTokenizer.from_pretrained(name, **load_kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop("return_mask", False)
        tokenizer_kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            tokenizer_kwargs.update(
                padding="max_length", truncation=True, max_length=self.seq_len
            )
        tokenizer_kwargs.update(kwargs)

        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [whitespace_clean(basic_clean(text)) for text in sequence]
        tokenized = self.tokenizer(sequence, **tokenizer_kwargs)
        if return_mask:
            return tokenized.input_ids, tokenized.attention_mask
        return tokenized.input_ids


def _as_instruction(value: Any) -> str:
    """Normalize serialized single-instruction values emitted by LeRobot."""
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError, TypeError):
        parsed = value
    if isinstance(parsed, (list, tuple)):
        parsed = parsed[0]
    return str(parsed)


def _format_checkpoint_instruction(value: Any) -> str:
    """Reproduce the text condition used to train the released checkpoints.

    The original InteriorGS pipeline shared embodiment id 17 with OXE DROID,
    so its collator applied this three-view prefix before tokenization.  The
    released WNM-3D checkpoints therefore require the same text condition even
    though InteriorGS itself is monocular.
    """
    instruction = _as_instruction(value).lower()
    return (
        "A multi-view video shows that a robot "
        + instruction
        + " The video is split into three views: The top view shows the camera "
        "view from the robot's wrist, the bottom-left view shows the camera view "
        "from the left exterior camera, and the bottom-right view shows the camera "
        "view from the right exterior camera. During training, one of the two "
        "bottom exterior views may be a black screen (dropped view). The robot "
        + instruction
    )


def collate(features: List[dict], tokenizer: AutoTokenizer) -> dict:
    batch = {}
    for key in features[0]:
        values = [item[key] for item in features]
        if key == "text":
            ids, mask = tokenizer(
                [_format_checkpoint_instruction(value) for value in values],
                return_mask=True,
                add_special_tokens=True,
            )
            batch[key] = ids
            batch["text_attention_mask"] = mask
        elif key == "text_negative":
            ids, mask = tokenizer(values, return_mask=True, add_special_tokens=True)
            batch[key] = ids
            batch["text_attention_mask_negative"] = mask
        else:
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class DefaultDataCollator(DataCollatorMixin):
    def __init__(self, tokenizer_path: str = "google/umt5-xxl", max_length: int = 512):
        super().__init__()
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=max_length, clean="whitespace"
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.tokenizer)


class WNM3DTransform(InvertibleModalityTransform):
    """Convert InteriorGS trajectories into WNM-3D model inputs."""

    apply_to: list[str] = Field(default_factory=list)
    training: bool = True
    formalize_language: bool = False
    language_dropout_prob: float = 0.0
    always_use_default_instruction: bool = False

    default_instruction: str
    max_state_dim: int
    max_action_dim: int
    max_length: int = 512
    state_horizon: int
    action_horizon: int
    tokenizer_path: str = "google/umt5-xxl"

    embodiment_tag: EmbodimentTag | None = None
    _language_key: Optional[str] = PrivateAttr(default=None)
    _language_keys: Optional[list[str]] = PrivateAttr(default=None)
    _tokenizer: Optional[HuggingfaceTokenizer] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = HuggingfaceTokenizer(
            name=self.tokenizer_path,
            seq_len=self.max_length,
            clean="whitespace",
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        if dataset_metadata.embodiment_tag != EmbodimentTag.INTERIORGS:
            raise ValueError(
                "WNM-3D supports InteriorGS metadata only; received "
                f"{dataset_metadata.embodiment_tag.value!r}."
            )
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def check_keys_and_batch_size(self, data):
        language_keys = [key for key in data if "annotation" in key]
        self._language_keys = language_keys
        self._language_key = language_keys[0] if language_keys else None

        video_ndim = data["video"].ndim
        if video_ndim == 5:  # [T, V, H, W, C]
            return False, 1
        if video_ndim == 6:  # [B, T, V, H, W, C]
            return True, data["video"].shape[0]
        raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        images = rearrange(batch["images"], "v t c h w -> (t v) h w c")
        language = batch["language"]
        if isinstance(language, (list, np.ndarray)):
            language = language[0]
        return {"images": images, "text": language}

    @staticmethod
    def _flatten_images_for_model(images: np.ndarray) -> np.ndarray:
        return rearrange(images, "v t c h w -> (t v) h w c")

    @staticmethod
    def _prepare_video_array(video: np.ndarray) -> np.ndarray:
        images = rearrange(video, "t v h w c -> v t c h w")
        if images.shape[0] != 1:
            raise ValueError(
                f"WNM-3D expects one monocular InteriorGS view, got {images.shape[0]}."
            )
        return images

    def _split_past_target_video(
        self, data: dict
    ) -> tuple[np.ndarray, np.ndarray | None]:
        video = data["video"]
        if video.ndim != 5:
            return video, None
        if "action" in data:
            target_frames = int(data["action"].shape[0]) + 1
        elif "target_prefix_frames" in data and video.shape[0] % 2 == 0:
            target_frames = video.shape[0] // 2
        else:
            return video, None
        if video.shape[0] == 2 * target_frames:
            return video[-target_frames:], video[:target_frames]
        return video, None

    def _prepare_language(self, data: dict) -> str:
        if self._language_key is None:
            raw_language = self.default_instruction
        else:
            raw_language = data[self._language_key]
            if isinstance(raw_language, np.ndarray):
                raw_language = (
                    raw_language.item() if raw_language.size == 1 else raw_language[0]
                )
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

        if (
            self.training
            and self.language_dropout_prob > 1e-9
            and random.random() < self.language_dropout_prob
        ):
            raw_language = self.default_instruction
        if self.always_use_default_instruction:
            raw_language = self.default_instruction

        raw_language = str(raw_language)
        return (
            formalize_language(raw_language)
            if self.formalize_language
            else raw_language
        )

    def _prepare_state(self, data: dict):
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros_like(state, dtype=bool)
            return state, state_mask, self.state_horizon

        state = data["state"]
        if state.shape[0] % self.state_horizon != 0:
            raise ValueError(
                f"Invalid state horizon: {state.shape=}, {self.state_horizon=}"
            )
        n_state_dims = min(state.shape[-1], self.max_state_dim)
        state = state[:, : self.max_state_dim]
        if state.shape[-1] < self.max_state_dim:
            state = np.pad(
                state, ((0, 0), (0, self.max_state_dim - state.shape[-1])), "constant"
            )
        state_mask = np.zeros_like(state, dtype=bool)
        state_mask[:, :n_state_dims] = True
        return state, state_mask, state.shape[0]

    def _prepare_action(self, data: dict):
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            action_mask = np.zeros_like(actions, dtype=bool)
            return actions, action_mask, self.action_horizon

        actions = data["action"]
        if actions.shape[0] % self.action_horizon != 0:
            raise ValueError(
                f"Invalid action horizon: {actions.shape=}, {self.action_horizon=}"
            )
        n_action_dims = actions.shape[1]
        if n_action_dims > self.max_action_dim:
            raise ValueError(
                f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."
            )
        actions = np.pad(
            actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant"
        )
        action_mask = np.zeros_like(actions, dtype=bool)
        action_mask[:, :n_action_dims] = True
        return actions, action_mask, actions.shape[0]

    def apply_single(self, data: dict) -> dict:
        if self.embodiment_tag != EmbodimentTag.INTERIORGS:
            raise RuntimeError(
                "Call set_metadata() with InteriorGS metadata before apply()."
            )

        target_video, past_video = self._split_past_target_video(data)
        images = self._prepare_video_array(target_video).astype(np.uint8)
        vlm_outputs = self._apply_vlm_processing(
            {"images": images, "language": self._prepare_language(data)}
        )

        transformed_data = {}
        if past_video is not None:
            past_images = self._prepare_video_array(past_video).astype(np.uint8)
            transformed_data["past_images"] = self._flatten_images_for_model(
                past_images
            )

        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            actions, action_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = action_mask
            transformed_data["has_real_action"] = np.ones((), dtype=bool)

        transformed_data["text_negative"] = (
            "Vibrant colors, overexposed, static, blurry details, text, subtitles, "
            "style, artwork, painting, image, still, grayscale, dull, worst quality, "
            "low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, "
            "bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant "
            "image, cluttered background, three legs, many people in the background, "
            "walking backwards."
        )
        transformed_data.update(vlm_outputs)
        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        split_data = [
            tree.map_structure(lambda value: value[i], data) for i in range(batch_size)
        ]
        return collate([self.apply_single(item) for item in split_data], self.tokenizer)

    def apply(self, data: dict) -> dict:
        if not self.training and data["video"].ndim == 5:
            data = dict(data)
            data["video"] = data["video"][None, ...]
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        return (
            self.apply_batch(data, batch_size)
            if is_batched
            else self.apply_single(data)
        )

    def unapply(self, data: dict) -> dict:
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
