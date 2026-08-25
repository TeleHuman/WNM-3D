from abc import ABC, abstractmethod

from torch import nn
from transformers.feature_extraction_utils import BatchFeature


class ActionHead(ABC, nn.Module):
    def __init__(self):
        super(ActionHead, self).__init__()

    @abstractmethod
    def forward(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        pass

    def prepare_input(self, batch: dict) -> BatchFeature:
        pass
