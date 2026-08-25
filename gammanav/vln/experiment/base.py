# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/

from abc import ABC
import json
import os
from pathlib import Path
import shutil
from typing import Optional
import warnings

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import Dataset, Sampler
import transformers
from transformers import TrainerCallback, set_seed
from transformers.trainer import (
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)

from gammanav.vln.data.action_normalization import (
    ACTION_NORMALIZATION_FILENAME,
    write_action_normalization_manifest,
)
from gammanav.vln.experiment.utils import (
    compute_grad_accum_to_match_global_bs,
    dtype_from_string,
    get_checkpoint_path,
    mprint,
    safe_save_model_for_hf_trainer,
)
from gammanav.vln.utils.timer import ContextTimer

# Fix resume: https://github.com/huggingface/transformers/pull/34632/files
np_core = np.core
allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
# numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
# all versions of numpy
allowlist += [type(np.dtype(np.uint32))]
torch.serialization.add_safe_globals(allowlist)

LAYERNORM_LAYERS = [
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LocalResponseNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
]


class LossLoggerCallback(TrainerCallback):
    """Callback that writes per-step loss metrics to a JSONL file for offline analysis."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        entry = {"step": state.global_step}
        for key in ("loss", "dynamics_loss_avg", "action_loss_avg", "learning_rate"):
            if key in logs:
                entry[key] = logs[key]
        for key, value in logs.items():
            if key.startswith(("timing_", "vggt_timing_")):
                entry[key] = value
        if len(entry) > 1:  # more than just "step"
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - metadata.json

    It also copies checkpoint-level metadata files such as
    ``action_normalization.json`` into /checkpoint-{step}/.
    """

    def __init__(
        self,
        exp_cfg_dir: Path | None = None,
        checkpoint_files: tuple[Path, ...] = (),
    ):
        """
        Args:
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.checkpoint_files = checkpoint_files

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            for source_path in self.checkpoint_files:
                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"Checkpoint metadata file does not exist: {source_path}"
                    )
                destination_path = checkpoint_dir / source_path.name
                print(
                    f"Copying checkpoint metadata file {source_path} "
                    f"to {destination_path}"
                )
                shutil.copy2(source_path, destination_path)


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class BaseTrainer(transformers.Trainer):
    def __init__(self, **kwargs):
        # Increase the cache size limit for torch._dynamo to
        # accommodate videos with different numbers of frames.
        torch._dynamo.config.cache_size_limit = 1000

        self.compute_dtype = kwargs.pop("compute_dtype")
        self.output_dir = kwargs.pop("output_dir")
        self.timer = ContextTimer(self)

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.node_rank = int(os.environ.get("NODE_RANK", "0"))

        # Get distributed info
        self.current_step = 0

        super().__init__(**kwargs)

        self.loss_queues = {}
        self.loss_queue_size = 10

    def _metric_log_interval(self):
        return max(
            1,
            int(
                getattr(self.args, "logging_steps", self.loss_queue_size)
                or self.loss_queue_size
            ),
        )

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self.timer.with_label("training_step"):
            output = super().training_step(model, inputs)

        if self.current_step % self._metric_log_interval() == 0:
            grad_metrics = self._collect_vggt_grad_metrics(model)
            if grad_metrics:
                self.log(grad_metrics)

        self.current_step += 1
        return output

    def _collect_vggt_grad_metrics(self, model):
        metrics = {}
        for root in (model, getattr(model, "module", None)):
            if root is None or not hasattr(root, "modules"):
                continue
            for module in root.modules():
                hook_metrics = getattr(module, "_last_vggt_grad_metrics", None)
                if hook_metrics:
                    metrics.update(hook_metrics)
        if metrics:
            return metrics

        named_parameters = getattr(model, "named_parameters", None)
        if named_parameters is None and hasattr(model, "module"):
            named_parameters = getattr(model.module, "named_parameters", None)
        if named_parameters is None:
            return metrics

        adapter_grad_sq = 0.0
        has_adapter_grad = False
        for name, param in named_parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            if "vggt_geometry_adapter" in name:
                grad_norm = grad.detach().float().norm().item()
                adapter_grad_sq += grad_norm * grad_norm
                has_adapter_grad = True

        if has_adapter_grad:
            metrics["vggt_adapter_grad_norm"] = adapter_grad_sq**0.5
        return metrics

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        with self.timer.with_label("model_forward"):
            outputs = model(inputs)
        ### For additional losses, track and log their moving averages
        for key, value in outputs.items():
            if key.endswith("_loss") and key != "loss":
                # Initialize queue if not exists
                if key not in self.loss_queues:
                    self.loss_queues[key] = []

                # Add current loss value to queue
                current_value = value.item() if torch.is_tensor(value) else value
                self.loss_queues[key].append(current_value)

                # Keep only last N values
                if len(self.loss_queues[key]) > self.loss_queue_size:
                    self.loss_queues[key].pop(0)

                # Log average every 10 steps
                if self.current_step % self._metric_log_interval() == 0:
                    avg_loss = sum(self.loss_queues[key]) / len(self.loss_queues[key])
                    self.log({f"{key}_avg": avg_loss})
            elif key.endswith("_metric"):
                if self.current_step % self._metric_log_interval() == 0:
                    current_value = (
                        value.detach().float().mean().item()
                        if torch.is_tensor(value)
                        else value
                    )
                    self.log({key[: -len("_metric")]: current_value})

        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = (
                transformers.Trainer.get_optimizer_cls_and_kwargs(self.args)
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            # DeepSpeed CPU Adam (ZeRO offload) expects 'bias_correction' in each param group.
            # HuggingFace Trainer's AdamW does not set it, causing KeyError in cpu_adam.step().
            if getattr(self.args, "deepspeed", None):
                for group in self.optimizer.param_groups:
                    group.setdefault("bias_correction", True)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):

        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.base_cfg.save_lora_only:
            # Save only the trainable parameters
            train_key = [k for k, v in self.model.named_parameters() if v.requires_grad]
            lora_state_dict = {
                k: v for k, v in self.model.state_dict().items() if k in train_key
            }
            state_dict = lora_state_dict

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(
            resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs
        )


class BaseExperiment(ABC):
    def __init__(self, cfg: DictConfig):
        assert cfg.max_steps > 0, "max_steps must be > 0"

        if cfg.load_from_yaml is not None:
            # Override the default config with the loaded config.
            loaded_cfg = OmegaConf.load(cfg.load_from_yaml)
            cfg = loaded_cfg  # overwrite

        # Instantiate the training arguments.
        cfg.training_args.output_dir = cfg.training_args.output_dir.rstrip("/")
        cfg.training_args.run_name = cfg.training_args.output_dir.split("/")[-1]
        print(f"Run name: {cfg.training_args.run_name}")
        training_args = instantiate(cfg.training_args)
        set_seed(training_args.seed)

        # Set the environment variables for wandb.
        if "WANDB_PROJECT" not in os.environ:
            os.environ["WANDB_PROJECT"] = cfg.wandb_project
        os.environ["WANDB_DIR"] = training_args.output_dir

        # Create the experiment config directory.
        output_dir = Path(training_args.output_dir)
        exp_cfg_dir = output_dir / "experiment_cfg"
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        # Check if we are resuming training.
        resume_path, continue_training = get_checkpoint_path(training_args.output_dir)
        if not continue_training:
            print(f"Models is ready under {training_args.output_dir}. Skip training.")
            exit(0)
        if resume_path:
            print(f"Resuming training from {resume_path}")
            resume_from_checkpoint = True
        else:
            # First time training.
            resume_from_checkpoint = False

        # Instantiate the model.
        model = self.create_model(cfg, training_args)

        if hasattr(model.action_head, "max_steps"):
            model.action_head.max_steps = cfg.max_steps

        # Make sure model_dtype and training_args dtype are compatible.
        compute_dtype = dtype_from_string(model.config.model_dtype)

        # Create the train dataset.
        # Dump the metadata; necessary for policy to normalize the input and unnormalize the output
        train_dataset = self.create_train_dataset(cfg, model)
        print("Using dataset:")
        print(train_dataset)
        assert train_dataset.merged_metadata is not None, (
            "You must set metadata_config.merge=true in order to save the metadata."
        )

        metadata_by_embodiment = {
            k: v.model_dump(mode="json")
            for k, v in train_dataset.merged_metadata.items()
        }
        metadata_save_path = exp_cfg_dir / "metadata.json"
        with metadata_save_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata_by_embodiment, metadata_file, indent=4)
        print("Successfully dumped metadata")

        action_normalization_path = None
        if (
            "interiorgs" in metadata_by_embodiment
            and OmegaConf.select(
                cfg,
                "train_dataset.dataset_kwargs.nav_action_scale",
                default=None,
            )
            is not None
        ):
            action_normalization_path = output_dir / ACTION_NORMALIZATION_FILENAME
            if training_args.should_save:
                write_action_normalization_manifest(
                    output_dir,
                    cfg,
                    metadata_by_embodiment,
                )
                print(
                    "Successfully dumped action normalization manifest to "
                    f"{action_normalization_path}"
                )

        data_collator = self.create_data_collator(cfg, model)
        trainer = self.create_trainer(
            cfg=cfg,
            exp_cfg_dir=exp_cfg_dir,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
            checkpoint_files=(
                (action_normalization_path,)
                if action_normalization_path is not None
                else ()
            ),
        )
        self.cfg = cfg
        self.exp_cfg_dir = exp_cfg_dir
        self.training_args = training_args
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset
        self.trainer = trainer

    def create_model(self, cfg, training_args):
        model = instantiate(cfg.model)

        if cfg.pretrained_model_path is not None:
            mprint(f"Loading pretrained weights from: {cfg.pretrained_model_path}")
            import gc
            import json
            from safetensors.torch import load_file

            ckpt_dir = cfg.pretrained_model_path
            safetensors_index_path = os.path.join(
                ckpt_dir, "model.safetensors.index.json"
            )
            safetensors_path = os.path.join(ckpt_dir, "model.safetensors")

            if os.path.exists(safetensors_index_path):
                with open(safetensors_index_path, "r") as f:
                    index = json.load(f)
                for shard_file in sorted(set(index["weight_map"].values())):
                    shard_path = os.path.join(ckpt_dir, shard_file)
                    mprint(f"Loading shard: {shard_path}")
                    shard_state_dict = load_file(shard_path)
                    model.load_state_dict(shard_state_dict, strict=False)
                    del shard_state_dict
                    gc.collect()
            elif os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(
                    f"No weights found at '{ckpt_dir}'. "
                    "Expected 'model.safetensors' or 'model.safetensors.index.json'."
                )

            if (
                hasattr(model, "action_head")
                and hasattr(model.action_head, "inject_lora_after_loading")
                and model.action_head.config.defer_lora_injection
            ):
                model.action_head.inject_lora_after_loading()

            mprint("Successfully loaded pretrained weights")

        model.config._name_or_path = training_args.output_dir
        mprint(f"{model}\n")
        return model

    def create_train_dataset(self, cfg, model):
        assert torch.distributed.is_initialized()
        train_dataset = instantiate(cfg.train_dataset)
        return train_dataset

    def create_data_collator(self, cfg, model):
        return instantiate(cfg.data_collator)

    def create_trainer(
        self,
        cfg,
        exp_cfg_dir,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
        checkpoint_files: tuple[Path, ...] = (),
    ):
        # Set the gradient accumulation steps.
        if cfg.global_batch_size is not None:
            global_bs = cfg.global_batch_size
            bs = training_args.per_device_train_batch_size
            grad_acc = compute_grad_accum_to_match_global_bs(global_bs, bs)
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_bs}, set gradient accumulation steps to {grad_acc}"
            )
        elif cfg.raise_error_if_global_batch_size_not_set:
            raise ValueError(
                "global_batch_size is not set. To ensure the scripts can be reproduced regardless of the number of nodes used, please set this."
            )
        else:
            warnings.warn(
                "global_batch_size is not set. This is fine for debugging, but please set this for real experiments."
            )

        # Instantiate the partial trainer.
        trainer_partial = instantiate(
            cfg.trainer,
            model=model,
            output_dir=training_args.output_dir,
            train_dataset=train_dataset,
            compute_dtype=compute_dtype,
        )

        # Fully instantiate the trainer with dataclasses instances.
        trainer = trainer_partial(data_collator=data_collator, args=training_args)
        trainer.base_cfg = cfg
        train_dl_len = len(trainer.get_train_dataloader())

        # Save the total training steps in the config.
        with open_dict(cfg):
            cfg.total_training_steps = train_dl_len * cfg.training_args.num_train_epochs

        # Save config.
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        ckpt_format_callback = CheckpointFormatCallback(
            exp_cfg_dir=exp_cfg_dir,
            checkpoint_files=checkpoint_files,
        )
        trainer.add_callback(ckpt_format_callback)

        loss_log_path = str(Path(training_args.output_dir) / "loss_log.jsonl")
        trainer.add_callback(LossLoggerCallback(output_path=loss_log_path))

        mprint(
            f"train dataloader length: {train_dl_len}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

    def train(self):
        # Start training.
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_state()
        safe_save_model_for_hf_trainer(
            trainer=self.trainer, output_dir=self.training_args.output_dir
        )
