from typing import Any

import numpy as np
from pydantic import Field
import torch

from gammanav.vln.data.schema import DatasetMetadata, StateActionMetadata
from gammanav.vln.data.transform.base import InvertibleModalityTransform


class ConcatTransform(InvertibleModalityTransform):
    """Concatenate WNM-3D video, navigation state, and navigation action keys."""

    apply_to: list[str] = Field(default_factory=list)
    video_concat_order: list[str]
    state_concat_order: list[str] | None = None
    action_concat_order: list[str] | None = None
    action_dims: dict[str, int] = Field(default_factory=dict)
    state_dims: dict[str, int] = Field(default_factory=dict)

    def model_dump(self, *args, **kwargs):
        include = (
            {
                "apply_to",
                "video_concat_order",
                "state_concat_order",
                "action_concat_order",
            }
            if kwargs.get("mode", "python") == "json"
            else kwargs.pop("include", None)
        )
        return super().model_dump(*args, include=include, **kwargs)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if any(key in data for key in self.video_concat_order):
            missing = [key for key in self.video_concat_order if key not in data]
            if missing:
                raise KeyError(f"Missing video keys: {missing}")
            data["video"] = np.concatenate(
                [
                    np.expand_dims(data.pop(key), axis=-4)
                    for key in self.video_concat_order
                ],
                axis=-4,
            )

        if self.state_concat_order and any(
            key in data for key in self.state_concat_order
        ):
            self._validate_dims(data, self.state_concat_order, self.state_dims, "state")
            data["state"] = torch.cat(
                [data.pop(key) for key in self.state_concat_order], dim=-1
            )

        if self.action_concat_order and any(
            key in data for key in self.action_concat_order
        ):
            self._validate_dims(
                data, self.action_concat_order, self.action_dims, "action"
            )
            data["action"] = torch.cat(
                [data.pop(key) for key in self.action_concat_order], dim=-1
            )
        return data

    @staticmethod
    def _validate_dims(data, order, expected_dims, modality):
        missing = [key for key in order if key not in data]
        if missing:
            raise KeyError(f"Missing {modality} keys: {missing}")
        for key in order:
            if data[key].shape[-1] != expected_dims[key]:
                raise ValueError(
                    f"{key} has dimension {data[key].shape[-1]}, expected {expected_dims[key]}"
                )

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        if "action" in data:
            if not self.action_concat_order:
                raise ValueError("action_concat_order is required to unapply actions")
            action = data.pop("action")
            offset = 0
            for key in self.action_concat_order:
                width = self.action_dims[key]
                data[key] = action[..., offset : offset + width]
                offset += width

        if "state" in data:
            if not self.state_concat_order:
                raise ValueError("state_concat_order is required to unapply state")
            state = data.pop("state")
            offset = 0
            for key in self.state_concat_order:
                width = self.state_dims[key]
                data[key] = state[..., offset : offset + width]
                offset += width
        return data

    def get_modality_metadata(self, key: str) -> StateActionMetadata:
        modality, subkey = key.split(".", 1)
        if self.dataset_metadata is None:
            raise RuntimeError("Metadata is not set")
        metadata = getattr(self.dataset_metadata.modalities, modality)[subkey]
        return metadata

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        super().set_metadata(dataset_metadata)
        for key in self.action_concat_order or []:
            self.action_dims[key] = self.get_modality_metadata(key).shape[0]
        for key in self.state_concat_order or []:
            self.state_dims[key] = self.get_modality_metadata(key).shape[0]
