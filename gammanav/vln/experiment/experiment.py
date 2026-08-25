import logging

import hydra
import torch.distributed as dist

from gammanav.vln.experiment.base import BaseExperiment, BaseTrainer
from gammanav.vln.utils.action_args_override_utils import apply_action_overrides

logger = logging.getLogger(__name__)


class VLNTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        import torch.distributed as dist

        self.rank = dist.get_rank()

        self.micro_global_step = 0

        super().__init__(**kwargs)

    def training_step(self, model, inputs, *args, **kwargs):
        self.micro_global_step += 1

        if hasattr(self.model.action_head, "global_step"):
            self.model.action_head.global_step = self.state.global_step
            self.model.action_head.metric_log_step = self.current_step
            self.model.action_head.metric_log_interval = self._metric_log_interval()

        loss_dict = super().training_step(model, inputs, *args, **kwargs)
        return loss_dict


class VLNExperiment(BaseExperiment):
    pass


@hydra.main(config_path="../configs", config_name="conf", version_base=None)
def main(cfg):
    # Automatically update action dim and action horizon keys if specified in the config
    cfg = apply_action_overrides(cfg)

    try:
        experiment = VLNExperiment(cfg)
        experiment.train()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
