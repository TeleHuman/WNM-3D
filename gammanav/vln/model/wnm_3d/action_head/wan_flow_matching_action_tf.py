from dataclasses import dataclass, field
import json
import logging
import math
import os
import time
from typing import TypeAlias, cast

from einops import rearrange
from huggingface_hub import hf_hub_download
from hydra.utils import instantiate
from peft import LoraConfig, get_peft_model
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributions import Beta
from torchvision.transforms import v2
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from safetensors.torch import load_file

from gammanav.vln.model.wnm_3d.action_head.base_action_head import ActionHead
from gammanav.vln.model.wnm_3d.modules.flow_match_scheduler import FlowMatchScheduler
from gammanav.vln.model.wnm_3d.modules.flow_unipc_multistep_scheduler import (
    FlowUniPCMultistepScheduler,
)
from gammanav.vln.model.wnm_3d.modules.vggt_geometry_adapter import (
    VGGTOmegaGeometryAdapter,
)
from gammanav.vln.model.wnm_3d.modules.vggt_omega.models.aggregator import (
    Aggregator as VGGTOmegaAggregator,
)


logger = logging.getLogger(__name__)

WAN_HF_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN22_HF_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"


def hf_download(filename: str, repo_id: str = WAN_HF_REPO_ID) -> str:
    """Download a file from the specified HuggingFace repo to HF cache."""
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return path


def ensure_file(
    path: str | None, hf_filename: str, repo_id: str = WAN_HF_REPO_ID
) -> str:
    """Return a valid local path: use `path` if it exists, otherwise download from HuggingFace."""
    if path is not None and os.path.exists(path):
        return path
    return hf_download(hf_filename, repo_id)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_num_inference_steps(config) -> int:
    override = os.getenv("NUM_INFERENCE_STEPS")
    value = (
        override
        if override is not None
        else getattr(config, "num_inference_timesteps", None)
    )
    if value is None or int(value) <= 0:
        raise ValueError(
            "num_inference_timesteps must be a positive integer for WNM3D inference, "
            f"got {value!r}"
        )
    return int(value)


DIT_STEP_MASK = (
    True,
    True,
    True,
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    True,
    False,
    False,
    True,
    True,
    True,
)


def _resolve_dit_step_mask(num_inference_steps: int) -> list[bool]:
    if num_inference_steps <= 0:
        raise ValueError(
            f"num_inference_steps must be positive, got {num_inference_steps}."
        )
    if num_inference_steps > len(DIT_STEP_MASK):
        raise ValueError(
            f"num_inference_steps={num_inference_steps} exceeds the fixed "
            f"{len(DIT_STEP_MASK)}-step DiT mask."
        )
    return list(DIT_STEP_MASK[:num_inference_steps])


KVCacheType: TypeAlias = torch.Tensor


@dataclass
class WANPolicyHeadConfig(PretrainedConfig):
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    tiled: bool = field(default=True, metadata={"help": "Whether to use tiled input."})
    tile_size_height: int = field(default=34, metadata={"help": "Tile size height."})
    tile_size_width: int = field(default=34, metadata={"help": "Tile size width."})
    tile_stride_height: int = field(
        default=18, metadata={"help": "Tile stride height."}
    )
    tile_stride_width: int = field(default=16, metadata={"help": "Tile stride width."})
    num_frame_per_block: int = field(
        default=1, metadata={"help": "Number of frames per block."}
    )
    # Target video (H, W) for Wan22 resize. When set, videos are resized to this before VAE so latent
    # spatial size matches. Use height/width divisible by 32 for WanVideoVAE38 (16x) so latent H,W are even.
    target_video_height: int | None = field(
        default=None,
        metadata={
            "help": "Target video height for resize (e.g. 160 for even latent with VAE38)."
        },
    )
    target_video_width: int | None = field(
        default=None, metadata={"help": "Target video width for resize (e.g. 320)."}
    )
    use_vggt_geometry_adapter: bool = field(
        default=False,
        metadata={
            "help": "Use frozen VGGT-Omega tokens as a geometry residual for clean video tokens."
        },
    )
    vggt_checkpoint_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to a VGGT-Omega checkpoint. The checkpoint may be a full VGGTOmega state dict."
        },
    )
    vggt_image_resolution: int = field(
        default=512, metadata={"help": "VGGT-Omega preprocessing resolution."}
    )
    vggt_resize_mode: str = field(
        default="square",
        metadata={
            "help": "VGGT-Omega tensor resize mode: square, balanced, or max_size."
        },
    )
    vggt_patch_size: int = field(
        default=16, metadata={"help": "VGGT-Omega image patch size."}
    )
    vggt_adapter_dim: int = field(
        default=512, metadata={"help": "Hidden size of the VGGT geometry adapter."}
    )
    vggt_adapter_blocks: int = field(
        default=2, metadata={"help": "Number of temporal geometry adapter blocks."}
    )
    vggt_adapter_heads: int = field(
        default=8, metadata={"help": "Attention heads in the VGGT geometry adapter."}
    )
    vggt_encoder_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Storage/compute dtype for the frozen VGGT-Omega encoder."},
    )

    lora_rank: int = field(default=4, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=4, metadata={"help": "LoRA alpha."})
    lora_target_modules: str = field(default="q,k,v,o,ffn.0,ffn.2")
    init_lora_weights: str = field(
        default="kaiming", metadata={"help": "LoRA initialization method."}
    )
    train_architecture: str = field(
        default="lora", metadata={"help": "Train architecture."}
    )
    skip_component_loading: bool = field(
        default=False,
        metadata={
            "help": "Skip loading individual component weights (used when loading from full pretrained model)."
        },
    )

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    # High noise emphasis for BASE (coupled) training - applies Beta distribution to BOTH video and action together
    use_high_noise_emphasis: bool = field(
        default=False,
        metadata={
            "help": "Use Beta distribution for noise sampling (biases BOTH video and action towards high noise levels together)."
        },
    )
    high_noise_beta_alpha: float = field(
        default=3.0,
        metadata={
            "help": "Beta alpha for high noise emphasis. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher = more high noise bias."
        },
    )
    # Decoupled noise sampling config for training-inference alignment
    # When enabled: video uses Beta(alpha,beta) biased towards high noise, action uses independent uniform
    decouple_video_action_noise: bool = field(
        default=False,
        metadata={
            "help": "Decouple video/action noise: video uses Beta distribution (high noise bias), action uses independent uniform."
        },
    )
    video_noise_beta_alpha: float = field(
        default=3.0,
        metadata={
            "help": "Beta alpha for video noise. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher alpha = more bias to high noise."
        },
    )
    video_noise_beta_beta: float = field(
        default=1.0, metadata={"help": "Beta beta for video noise. Keep at 1.0."}
    )
    # Decoupled inference config - allows video to stay noisy while action fully denoises
    decouple_inference_noise: bool = field(
        default=False,
        metadata={
            "help": "Use decoupled noise schedules during inference (video stays noisy, action fully denoises)."
        },
    )
    video_inference_final_noise: float = field(
        default=0.8,
        metadata={
            "help": "Final noise level for video during decoupled inference (0.0-1.0). E.g., 0.8 means video ends at 80% noise."
        },
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    tune_projector: bool = field(
        default=True, metadata={"help": "Whether to tune the projector."}
    )
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    defer_lora_injection: bool = field(
        default=False,
        metadata={
            "help": "Defer LoRA injection until after loading pretrained weights."
        },
    )

    text_encoder_cfg: dict = field(default=None)
    image_encoder_cfg: dict = field(default=None)
    vae_cfg: dict = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class WANPolicyHead(ActionHead):
    config_class = WANPolicyHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: WANPolicyHeadConfig,
    ):
        super().__init__()
        self.tiled = config.tiled
        self.tile_size_height = config.tile_size_height
        self.tile_size_width = config.tile_size_width
        self.tile_stride_height = config.tile_stride_height
        self.tile_stride_width = config.tile_stride_width
        self.num_frame_per_block = config.num_frame_per_block
        self.num_frames = config.num_frames
        self.text_encoder = instantiate(config.text_encoder_cfg)
        self.image_encoder = instantiate(config.image_encoder_cfg)
        self.vae = instantiate(config.vae_cfg)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)

        self.num_inference_steps = _resolve_num_inference_steps(config)
        self.seed = 1140
        self.enable_cfg = _env_flag("ENABLE_CFG")
        self.cfg_scale = float(os.getenv("CFG_SCALE", "5.0"))
        if not math.isfinite(self.cfg_scale) or self.cfg_scale <= 0:
            raise ValueError(
                f"CFG_SCALE must be a positive finite number, got {self.cfg_scale!r}."
            )
        self.sigma_shift = 5.0
        self.kv_cache1: KVCacheType | None = None
        self.kv_cache_neg: KVCacheType | None = None
        self.crossattn_cache: KVCacheType | None = None
        self.crossattn_cache_neg: KVCacheType | None = None

        self.global_step = 0
        self.max_steps = 0
        self.metric_log_step = 0
        self.metric_log_interval = max(
            1, int(os.getenv("WNM3D_METRIC_LOG_INTERVAL", "10"))
        )
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = config.lora_target_modules
        self.init_lora_weights = config.init_lora_weights
        self.train_architecture = config.train_architecture
        self.clip_feas = None
        self.ys = None
        self.current_start_frame = 0
        self.language = None

        self.ip_rank = 0
        self.ip_size = 1
        self.ip_group = None

        self._device = "cuda"
        self.dynamic_cache_schedule = (
            os.getenv("DYNAMIC_CACHE_SCHEDULE", "False").lower() == "true"
        )
        self.enable_dit_cache = _env_flag("ENABLE_DIT_CACHE")
        self.profile_module_times = _env_flag("WNM3D_PROFILE_MODULE_TIMES")

        self.dit_step_mask = _resolve_dit_step_mask(self.num_inference_steps)
        assert self.dit_step_mask[0], "first step must be True"

        self.normalize_video = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if self.training:
            self.scheduler.set_timesteps(1000, training=True)

        self.model = instantiate(config.diffusion_model_cfg)
        self._init_vggt_geometry_modules(config)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

        if not config.skip_component_loading:
            text_enc_path = ensure_file(
                self.text_encoder.text_encoder_pretrained_path,
                "models_t5_umt5-xxl-enc-bf16.pth",
            )
            self.text_encoder.load_state_dict(
                torch.load(text_enc_path, map_location="cpu")
            )

            img_enc_path = ensure_file(
                self.image_encoder.image_encoder_pretrained_path,
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            )
            self.image_encoder.model.load_state_dict(
                torch.load(img_enc_path, map_location="cpu"), strict=False
            )

            # Wan2.2 (WanVideoVAE38, z_dim=48) uses Wan2.2_VAE.pth; Wan2.1 uses Wan2.1_VAE.pth.
            vae_hf_filename = (
                "Wan2.2_VAE.pth"
                if getattr(self.vae, "z_dim", 16) == 48
                else "Wan2.1_VAE.pth"
            )
            vae_repo_id = (
                WAN22_HF_REPO_ID
                if getattr(self.vae, "z_dim", 16) == 48
                else WAN_HF_REPO_ID
            )
            vae_path = ensure_file(
                self.vae.vae_pretrained_path,
                vae_hf_filename,
                repo_id=vae_repo_id,
            )
            self.vae.model.load_state_dict(torch.load(vae_path, map_location="cpu"))

            dit_dir = self.model.diffusion_model_pretrained_path
            # Wan2.2 (in_dim=48) uses Wan2.2-TI2V-5B repo; Wan2.1 uses Wan2.1-I2V-14B-480P
            dit_repo_id = (
                WAN22_HF_REPO_ID
                if getattr(self.model, "in_dim", 16) == 48
                else WAN_HF_REPO_ID
            )
            if dit_dir is None or not os.path.isdir(dit_dir):
                index_path = hf_hub_download(
                    repo_id=dit_repo_id,
                    filename="diffusion_pytorch_model.safetensors.index.json",
                )
                dit_dir = os.path.dirname(index_path)
                with open(index_path, "r") as f:
                    index = json.load(f)
                for shard_file in set(index["weight_map"].values()):
                    hf_hub_download(repo_id=dit_repo_id, filename=shard_file)

            if dit_dir is not None:
                safetensors_path = os.path.join(
                    dit_dir, "diffusion_pytorch_model.safetensors"
                )
                safetensors_index_path = os.path.join(
                    dit_dir, "diffusion_pytorch_model.safetensors.index.json"
                )
                state_dict = {}

                if os.path.exists(safetensors_index_path):
                    # Handle sharded safetensors
                    print(
                        f"Loading sharded safetensors using index: {safetensors_index_path}"
                    )

                    with open(safetensors_index_path, "r") as f:
                        index = json.load(f)

                    # Load each shard
                    for shard_file in set(index["weight_map"].values()):
                        shard_path = os.path.join(dit_dir, shard_file)
                        print(f"Loading shard: {shard_path}")
                        shard_state_dict = load_file(shard_path)
                        state_dict.update(shard_state_dict)

                elif os.path.exists(safetensors_path):
                    # Handle single safetensors file
                    print(f"Loading weights from safetensors: {safetensors_path}")
                    state_dict = load_file(safetensors_path)

                else:
                    raise ValueError(
                        f"No safetensors file found at {safetensors_path} or {safetensors_index_path}"
                    )

                missing_keys, unexpected_keys = self.model.load_state_dict(
                    state_dict, strict=False
                )

                if missing_keys:
                    print(
                        "Missing keys when loading pretrained weights: "
                        f"count={len(missing_keys)} sample={missing_keys[:8]}"
                    )
                if unexpected_keys:
                    print(
                        "Unexpected keys when loading pretrained weights: "
                        f"count={len(unexpected_keys)} sample={unexpected_keys[:8]}"
                    )

                print("Successfully loaded pretrained weights")
        else:
            print(
                "Skipping base component files; all WNM-3D weights will be loaded "
                "from the full checkpoint"
            )
        # Video noise Beta distribution (biased towards high noise levels when enabled)
        self.video_beta_dist = Beta(
            config.video_noise_beta_alpha, config.video_noise_beta_beta
        )
        # High noise emphasis Beta distribution for coupled training (applies to both video and action)
        self.high_noise_beta_dist = Beta(config.high_noise_beta_alpha, 1.0)
        self.config = config
        self._noise_logged = False
        self.defer_lora_injection = config.defer_lora_injection
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model
        )

    def _profile_sync(self, device: torch.device | str) -> None:
        if not self.profile_module_times:
            return
        device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _profile_start(self, device: torch.device | str) -> float | None:
        if not self.profile_module_times:
            return None
        self._profile_sync(device)
        return time.perf_counter()

    def _record_profile_time(
        self,
        metrics: dict[str, torch.Tensor],
        key: str,
        start: float | None,
        device: torch.device | str,
    ) -> None:
        if not self.profile_module_times or start is None:
            return
        self._profile_sync(device)
        metrics[key] = torch.tensor(
            time.perf_counter() - start,
            device=torch.device(device),
            dtype=torch.float32,
        )

    def _init_vggt_geometry_modules(self, config: WANPolicyHeadConfig):
        self.use_vggt_geometry_adapter = bool(
            getattr(config, "use_vggt_geometry_adapter", False)
        )
        self.vggt_aggregator = None
        self.vggt_geometry_adapter = None
        self._last_vggt_geometry_metrics = {}
        self._last_vggt_grad_metrics = {}
        self._vggt_adapter_grad_sq = 0.0
        self._vggt_grad_hook_handles = []
        self.vggt_encoder_dtype = self._dtype_from_name(
            getattr(config, "vggt_encoder_dtype", "bfloat16")
        )
        if not self.use_vggt_geometry_adapter:
            return

        self.vggt_aggregator = VGGTOmegaAggregator(
            patch_size=getattr(config, "vggt_patch_size", 16),
        )
        ckpt_path = getattr(config, "vggt_checkpoint_path", None) or os.getenv(
            "VGGT_OMEGA_CKPT"
        )
        if not config.skip_component_loading:
            if ckpt_path is None or not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    "VGGT geometry adapter is enabled, but "
                    "vggt_checkpoint_path/VGGT_OMEGA_CKPT is missing. "
                    f"Got: {ckpt_path}"
                )
            self._load_vggt_aggregator_checkpoint(ckpt_path)
        else:
            print(
                "Skipping the standalone VGGT-Omega checkpoint; its parameters "
                "will be loaded from the full WNM-3D checkpoint"
            )
        self.vggt_aggregator.requires_grad_(False)
        self.vggt_aggregator.eval()

        self.vggt_geometry_adapter = VGGTOmegaGeometryAdapter(
            output_dim=self.model.dim,
            adapter_dim=getattr(config, "vggt_adapter_dim", 512),
            num_heads=getattr(config, "vggt_adapter_heads", 8),
            num_blocks=getattr(config, "vggt_adapter_blocks", 2),
        )
        self._register_vggt_grad_hooks()
        checkpoint_label = (
            ckpt_path if not config.skip_component_loading else "full-checkpoint"
        )
        print(
            "Enabled VGGT-Omega past-observation adapter "
            f"checkpoint={checkpoint_label} image_resolution={getattr(config, 'vggt_image_resolution', 512)} "
            f"resize_mode={getattr(config, 'vggt_resize_mode', 'square')}"
        )

    def _register_vggt_grad_hooks(self):
        for handle in getattr(self, "_vggt_grad_hook_handles", []):
            handle.remove()
        self._vggt_grad_hook_handles = []
        if self.vggt_geometry_adapter is not None:
            for param in self.vggt_geometry_adapter.parameters():
                if param.requires_grad:
                    self._vggt_grad_hook_handles.append(
                        param.register_hook(self._record_vggt_adapter_grad)
                    )

    def _reset_vggt_grad_metrics(self):
        self._last_vggt_grad_metrics = {}
        self._vggt_adapter_grad_sq = 0.0

    def _record_vggt_adapter_grad(self, grad: torch.Tensor):
        if not self._should_record_vggt_metrics():
            return grad
        grad_norm = grad.detach().float().norm().item()
        self._vggt_adapter_grad_sq += grad_norm * grad_norm
        self._last_vggt_grad_metrics["vggt_adapter_grad_norm"] = (
            self._vggt_adapter_grad_sq**0.5
        )
        return grad

    def _should_record_vggt_metrics(self) -> bool:
        if not self.training:
            return True
        interval = max(1, int(getattr(self, "metric_log_interval", 10) or 10))
        step = int(
            getattr(self, "metric_log_step", getattr(self, "global_step", 0)) or 0
        )
        return step % interval == 0

    @staticmethod
    def _dtype_from_name(dtype_name: str) -> torch.dtype:
        dtype_name = str(dtype_name).lower()
        if dtype_name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype_name in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype_name in {"fp32", "float32", "float"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype name: {dtype_name}")

    def _load_vggt_aggregator_checkpoint(self, ckpt_path: str):
        assert self.vggt_aggregator is not None
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported VGGT checkpoint type: {type(state_dict)}")
        aggregator_state = {}
        own_state = self.vggt_aggregator.state_dict()
        for key, value in state_dict.items():
            if key.startswith("module.aggregator."):
                aggregator_state[key[len("module.aggregator.") :]] = value
            elif key.startswith("aggregator."):
                aggregator_state[key[len("aggregator.") :]] = value
            elif key.startswith("module.") and key[len("module.") :] in own_state:
                aggregator_state[key[len("module.") :]] = value
            elif key in own_state:
                aggregator_state[key] = value
        missing_keys, unexpected_keys = self.vggt_aggregator.load_state_dict(
            aggregator_state, strict=False
        )
        print(
            "Loaded VGGT-Omega aggregator "
            f"from {ckpt_path}; missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
        )
        if missing_keys:
            print(f"VGGT aggregator missing sample: {missing_keys[:8]}")
        if unexpected_keys:
            print(f"VGGT aggregator unexpected sample: {unexpected_keys[:8]}")

    def _set_vggt_geometry_trainable(self):
        if not self.use_vggt_geometry_adapter:
            return
        assert self.vggt_aggregator is not None
        assert self.vggt_geometry_adapter is not None
        self.vggt_aggregator.requires_grad_(False)
        self.vggt_aggregator.eval()
        self.vggt_geometry_adapter.requires_grad_(True)

    def _prepare_vggt_images(
        self,
        images: torch.Tensor,
        target_frames: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,T,H,W,C], got {tuple(images.shape)}")
        if images.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB images in the last dimension, got {tuple(images.shape)}"
            )

        x = images.to(device=device)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            if x.max() > 2.0:
                x = x / 255.0
            elif x.min() < -0.1:
                x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)

        batch_size, num_frames, height, width, _ = x.shape
        if target_frames <= 0:
            raise ValueError(f"target_frames must be positive, got {target_frames}")
        if target_frames != num_frames:
            frame_indices = (
                torch.linspace(
                    0,
                    num_frames - 1,
                    target_frames,
                    device=x.device,
                )
                .round()
                .long()
            )
            x = x.index_select(1, frame_indices)

        x = x.permute(0, 1, 4, 2, 3).contiguous()
        resize_mode = str(getattr(self.config, "vggt_resize_mode", "square")).lower()
        if resize_mode != "square":
            x = self._center_crop_to_vggt_supported_aspect(x)
        _, _, _, height, width = x.shape
        target_h, target_w = self._vggt_target_shape(height, width)
        x = torch.nn.functional.interpolate(
            x.reshape(batch_size * target_frames, 3, height, width),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )
        x = x.clamp(0.0, 1.0).reshape(batch_size, target_frames, 3, target_h, target_w)
        return x

    def _center_crop_to_vggt_supported_aspect(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        _, _, _, height, width = images.shape
        aspect_ratio = height / max(width, 1)
        if aspect_ratio < 0.5:
            crop_width = min(width, max(1, int(round(height / 0.5))))
            left = max((width - crop_width) // 2, 0)
            return images[..., left : left + crop_width]
        if aspect_ratio > 2.0:
            crop_height = min(height, max(1, int(round(width * 2.0))))
            top = max((height - crop_height) // 2, 0)
            return images[..., top : top + crop_height, :]
        return images

    def _vggt_target_shape(self, height: int, width: int) -> tuple[int, int]:
        image_resolution = int(getattr(self.config, "vggt_image_resolution", 512))
        patch_size = int(getattr(self.config, "vggt_patch_size", 16))
        resize_mode = str(getattr(self.config, "vggt_resize_mode", "square")).lower()
        if image_resolution <= 0 or image_resolution % patch_size != 0:
            raise ValueError(
                f"vggt_image_resolution={image_resolution} must be positive and divisible by patch_size={patch_size}."
            )
        aspect_ratio = height / max(width, 1)
        if resize_mode == "square":
            h_patches = image_resolution // patch_size
            w_patches = image_resolution // patch_size
        elif resize_mode == "balanced":
            token_number = (image_resolution // patch_size) ** 2
            w_patches = math.sqrt(token_number / aspect_ratio)
            h_patches = token_number / w_patches
            w_patches = max(1, int(round(w_patches)))
            h_patches = max(1, int(round(h_patches)))
        elif resize_mode == "max_size":
            if aspect_ratio >= 1.0:
                h_patches = image_resolution // patch_size
                w_patches = max(1, int(round(h_patches / aspect_ratio)))
            else:
                w_patches = image_resolution // patch_size
                h_patches = max(1, int(round(w_patches * aspect_ratio)))
        else:
            raise ValueError(f"Unsupported vggt_resize_mode: {resize_mode}")
        return h_patches * patch_size, w_patches * patch_size

    def _build_vggt_past_obs_tokens(
        self,
        images: torch.Tensor,
        target_frames: int,
        target_grid_size: tuple[int, int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.use_vggt_geometry_adapter:
            return None
        assert self.vggt_aggregator is not None
        assert self.vggt_geometry_adapter is not None

        self._reset_vggt_grad_metrics()
        metrics: dict[str, torch.Tensor] = {}
        record_metrics = self._should_record_vggt_metrics()
        total_start = (
            self._profile_start(device)
            if self.profile_module_times
            else time.perf_counter()
        )
        self.vggt_aggregator.eval()
        source_frames = int(
            getattr(self.vggt_geometry_adapter, "source_t", target_frames)
        )
        prepare_start = self._profile_start(device)
        vggt_images = self._prepare_vggt_images(
            images, target_frames=source_frames, device=device
        )
        self._record_profile_time(
            metrics,
            "vggt_timing_prepare_images_metric",
            prepare_start,
            vggt_images.device,
        )
        device_type = torch.device(device).type
        use_amp = device_type == "cuda" and self.vggt_encoder_dtype in {
            torch.bfloat16,
            torch.float16,
        }
        aggregator_start = self._profile_start(vggt_images.device)
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type=device_type,
                dtype=self.vggt_encoder_dtype,
                enabled=use_amp,
            ),
        ):
            aggregated_tokens_list, patch_token_start = self.vggt_aggregator(
                vggt_images
            )
        self._record_profile_time(
            metrics,
            "vggt_timing_aggregator_metric",
            aggregator_start,
            vggt_images.device,
        )
        aggregated_tokens_list = [
            tokens.detach() if tokens is not None else None
            for tokens in aggregated_tokens_list
        ]

        cached_layers = [
            tokens for tokens in aggregated_tokens_list if tokens is not None
        ]
        num_vggt_taps = int(getattr(self.vggt_geometry_adapter, "num_vggt_taps", 4))
        source_patch_tokens = int(
            getattr(self.vggt_geometry_adapter, "source_patch_tokens", 0)
        )
        if record_metrics:
            selected_layers = cached_layers[-num_vggt_taps:]
            for layer_idx, tokens in enumerate(selected_layers):
                patch_end = (
                    patch_token_start + source_patch_tokens
                    if source_patch_tokens > 0
                    else None
                )
                patch_tokens = tokens[:, :, patch_token_start:patch_end]
                metrics[f"vggt_token_norm_layer_{layer_idx}_metric"] = (
                    patch_tokens.float().norm(dim=-1).mean().detach()
                )
            metrics["vggt_source_frames_metric"] = torch.tensor(
                float(vggt_images.shape[1]),
                device=device,
                dtype=torch.float32,
            )
            metrics["vggt_source_patch_tokens_metric"] = torch.tensor(
                float(source_patch_tokens),
                device=device,
                dtype=torch.float32,
            )

        adapter_start = self._profile_start(device)
        self.vggt_geometry_adapter.record_metrics = record_metrics
        with torch.amp.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            past_obs_tokens = self.vggt_geometry_adapter(
                aggregated_tokens_list=aggregated_tokens_list,
                patch_token_start=patch_token_start,
                target_frames=target_frames,
                target_grid_size=target_grid_size,
            )
        self._record_profile_time(
            metrics,
            "vggt_timing_adapter_metric",
            adapter_start,
            past_obs_tokens.device,
        )
        layer_weights = getattr(
            self.vggt_geometry_adapter, "last_layer_weight_means", None
        )
        if record_metrics and layer_weights is not None:
            for layer_idx, weight in enumerate(layer_weights):
                metrics[f"vggt_layer_weight_{layer_idx}_metric"] = weight.to(
                    device=past_obs_tokens.device,
                    dtype=torch.float32,
                )
        if record_metrics:
            metrics["vggt_past_obs_token_norm_metric"] = (
                past_obs_tokens.detach().float().norm(dim=-1).mean()
            )
            if self.profile_module_times:
                self._record_profile_time(
                    metrics,
                    "vggt_timing_total_metric",
                    total_start,
                    past_obs_tokens.device,
                )
                metrics["vggt_forward_time_metric"] = metrics[
                    "vggt_timing_total_metric"
                ]
            else:
                metrics["vggt_forward_time_metric"] = torch.tensor(
                    time.perf_counter() - total_start,
                    device=past_obs_tokens.device,
                    dtype=torch.float32,
                )
        else:
            self._last_vggt_geometry_metrics = {}
            return past_obs_tokens.to(device=device, dtype=dtype)
        self._last_vggt_geometry_metrics = metrics
        return past_obs_tokens.to(device=device, dtype=dtype)

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

        if self.train_architecture == "lora" and not self.defer_lora_injection:
            print("Adding LoRA to model")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
        elif self.train_architecture == "lora" and self.defer_lora_injection:
            print("Deferring LoRA injection until after pretrained weights are loaded")
        else:
            self.print_trainable_params()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self._set_vggt_geometry_trainable()
        if not self.defer_lora_injection:
            self.print_trainable_params()

    def print_trainable_params(self):
        """Print trainable parameters of the diffusion model."""
        trainable_params = []
        total_params = 0
        trainable_total = 0

        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params.append(name)
                trainable_total += param.numel()

        print(f"Total parameters in diffusion model: {total_params:,}")
        print(f"Trainable parameters in diffusion model: {trainable_total:,}")

    def inject_lora_after_loading(self):
        """
        Inject LoRA adapters after pretrained weights have been loaded.
        This should be called when defer_lora_injection=True.
        """
        if self.train_architecture == "lora":
            print("Injecting LoRA after loading pretrained weights")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)

            self.text_encoder.requires_grad_(False)
            self.image_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            self._set_vggt_geometry_trainable()
            self.print_trainable_params()
        else:
            print("LoRA injection not needed (train_architecture != 'lora')")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_diffusion_model:
                self.model.eval()
            self.text_encoder.eval()
            self.image_encoder.eval()
            self.vae.eval()
            if self.use_vggt_geometry_adapter and self.vggt_aggregator is not None:
                self.vggt_aggregator.eval()

    def _create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seqlen: int,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU KV cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        kv_cache1: KVCacheType = []
        kv_cache_neg: KVCacheType = []
        for _ in range(self.model.num_layers):
            kv_cache1.append(
                torch.zeros(
                    [2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device
                ),
            )
            kv_cache_neg.append(
                torch.zeros(
                    [2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device
                ),
            )

        return kv_cache1, kv_cache_neg

    def _create_crossattn_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        crossattn_cache: KVCacheType = []
        crossattn_cache_neg: KVCacheType = []

        for _ in range(self.model.num_layers):
            crossattn_cache.append(
                torch.zeros(
                    [2, batch_size, 512, num_heads, head_dim],
                    dtype=dtype,
                    device=device,
                ),
            )
            crossattn_cache_neg.append(
                torch.zeros(
                    [2, batch_size, 512, num_heads, head_dim],
                    dtype=dtype,
                    device=device,
                ),
            )

        return crossattn_cache, crossattn_cache_neg

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def encode_prompt(self, input_ids, attention_mask):
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
        valid_token_mask = (
            attention_mask.to(device=prompt_emb.device).gt(0).unsqueeze(-1)
        )
        prompt_emb = prompt_emb.masked_fill(~valid_token_mask, 0)
        return prompt_emb

    def _ensure_vae_on_device(self, ref_tensor):
        """Lazily move the VAE to the correct device/dtype on first use."""
        if not getattr(self, "_vae_device_ready", False):
            self.vae.to(device=ref_tensor.device, dtype=torch.bfloat16)
            self.vae.eval()
            self._vae_device_ready = True

    def encode_video(
        self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        self._ensure_vae_on_device(input_video)
        with torch.no_grad():
            latents = self.vae.encode(
                input_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            )
        return latents

    def encode_image(self, image, num_frames, height, width):
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self._device).type
        ):
            batch_size = image.shape[0]
            clip_context = self.image_encoder.encode_image(image)
            image_input = image.transpose(1, 2)
            image_zeros = torch.zeros(
                batch_size,
                3,
                num_frames - 1,
                height,
                width,
                dtype=torch.bfloat16,
                device=self._device,
            )
            self._ensure_vae_on_device(image_input)
            with torch.no_grad():
                y = self.vae.encode(torch.concat([image_input, image_zeros], dim=2))
            # Build mask to match VAE output shape (VAE may use different spatial downsampling, e.g. WanVideoVAE38 uses patch_size=2 -> height/16)
            # y shape is B * 16 * (1+(T-1)/4) * H_latent * W_latent
            num_t = y.shape[2]
            h_latent, w_latent = y.shape[3], y.shape[4]
            msk = torch.zeros(
                batch_size,
                4,
                num_t,
                h_latent,
                w_latent,
                dtype=y.dtype,
                device=self._device,
            )
            msk[:, :, 0:1, :, :] = 1
            new_image = y[:, :, 0:1]
            # concat: B * (4+16) * (1+(T-1)/4) * H_latent * W_latent
            y = torch.concat([msk, y], dim=1)
        return clip_context, y, new_image

    def add_lora_to_model(
        self,
        model,
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
    ) -> nn.Module:
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = get_peft_model(model, lora_config)
        for param in model.parameters():
            param.data = param.to(torch.float32)
        return model

    def forward(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        profile_metrics: dict[str, torch.Tensor] = {}
        forward_start = self._profile_start(self._device)
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        data = action_input
        has_real_action = action_input.has_real_action
        action_mask = action_input.action_mask

        state_features = action_input.state

        actions = action_input.action
        if actions.numel() > 0:
            assert actions.min() >= -1.0 and actions.max() <= 1.0, (
                "actions must be in [-1,1] range"
            )
        raw_past_videos_for_vggt = data.get("past_images", None)
        if self.use_vggt_geometry_adapter and raw_past_videos_for_vggt is None:
            raise ValueError(
                "WNM-3D training requires past_images from the InteriorGS transform."
            )
        videos = data["images"]

        videos = rearrange(videos, "b t h w c -> b c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(
                0, 2, 1, 3, 4
            )  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, (
                "videos must be in [-1,1] range"
            )
            videos = videos.to(dtype=self.dtype)

        # shape of B * max_length * dim
        prompt_start = self._profile_start(videos.device)
        prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])
        self._record_profile_time(
            profile_metrics,
            "timing_prompt_encode_metric",
            prompt_start,
            prompt_embs.device,
        )

        # Wan 5B: resize to target resolution so latent tokens/frame matches DiT. Use config target when set
        # (e.g. 160x320 so latent is 10x20 with VAE38 16x → even H,W, no crop in dynamics loss); else 176x320.
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        vae_start = self._profile_start(videos.device)
        latents = self.encode_video(
            videos,
            self.tiled,
            (self.tile_size_height, self.tile_size_width),
            (self.tile_stride_height, self.tile_stride_width),
        )
        self._record_profile_time(
            profile_metrics,
            "timing_wan_vae_target_encode_metric",
            vae_start,
            latents.device,
        )
        if latents.shape[2] <= 1:
            raise ValueError(
                f"WNM-3D training expects at least 2 target latents, got {latents.shape}"
            )
        vggt_start = self._profile_start(latents.device)
        past_obs_tokens = self._build_vggt_past_obs_tokens(
            raw_past_videos_for_vggt,
            target_frames=latents.shape[2],
            target_grid_size=(latents.shape[3] // 2, latents.shape[4] // 2),
            device=latents.device,
            dtype=latents.dtype,
        )
        self._record_profile_time(
            profile_metrics,
            "timing_vggt_condition_metric",
            vggt_start,
            latents.device,
        )

        _, _, num_frames, height, width = videos.shape
        image = videos[:, :, :1].transpose(1, 2)

        image_condition_start = self._profile_start(image.device)
        clip_feas, ys, _ = self.encode_image(image, num_frames, height, width)
        self._record_profile_time(
            profile_metrics,
            "timing_image_condition_encode_metric",
            image_condition_start,
            clip_feas.device,
        )

        latents = latents.to(self._device)
        if past_obs_tokens is not None:
            past_obs_tokens = past_obs_tokens.to(self._device)
        clip_feas = clip_feas.to(self._device)
        ys = ys.to(self._device)
        prompt_embs = prompt_embs.to(self._device)

        # Loss
        noise_schedule_start = self._profile_start(latents.device)
        noise = torch.randn_like(latents)

        # specific to autoregressive
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)

        # ============ VIDEO TIMESTEP SAMPLING ============
        if self.config.decouple_video_action_noise:
            # Decoupled mode: sample video from Beta distribution biased towards HIGH noise
            video_noise_ratio = self.video_beta_dist.sample(
                [noise.shape[0], noise.shape[1]]
            )
            timestep_id = (
                (1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps
            ).long()
            timestep_id = torch.clamp(
                timestep_id, 0, self.scheduler.num_train_timesteps - 1
            )
            noise_mode = "DECOUPLED"
        elif self.config.use_high_noise_emphasis:
            # High noise emphasis mode (coupled): BOTH video and action use Beta distribution
            noise_ratio = self.high_noise_beta_dist.sample(
                [noise.shape[0], noise.shape[1]]
            )
            timestep_id = (
                (1.0 - noise_ratio) * self.scheduler.num_train_timesteps
            ).long()
            timestep_id = torch.clamp(
                timestep_id, 0, self.scheduler.num_train_timesteps - 1
            )
            noise_mode = "HIGH_NOISE_EMPHASIS"
        else:
            # Original: uniform sampling over full range
            timestep_id = torch.randint(
                0, self.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1])
            )
            noise_mode = "STANDARD"

        timestep_id_block = timestep_id[:, 1:].reshape(
            timestep_id.shape[0], -1, self.num_frame_per_block
        )
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]

        if actions.numel() > 0:
            noise_action = torch.randn_like(actions)
            assert actions.shape[1] / (noise.shape[1] - 1) == (
                self.model.num_action_per_block // self.num_frame_per_block
            ), (
                f"actions.shape, {actions.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            )
            assert (noise.shape[1] - 1) / state_features.shape[1] == (
                self.num_frame_per_block // self.model.num_state_per_block
            ), (
                f"state_features.shape, {state_features.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            )

            # ============ ACTION TIMESTEP SAMPLING ============
            if self.config.decouple_video_action_noise:
                # Decoupled: sample action timestep independently with full range
                timestep_action_id = torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (actions.shape[0], actions.shape[1]),
                )
                action_mode = "INDEPENDENT"
            else:
                # Original coupled: action timestep derived from video timestep
                timestep_action_id = timestep_id_block.repeat(
                    1, 1, actions.shape[1] // (noise.shape[1] - 1)
                )
                timestep_action_id = timestep_action_id.reshape(
                    timestep_action_id.shape[0], -1
                )
                action_mode = "COUPLED"

            # Log noise mode once
            if not self._noise_logged:
                video_mean = timestep_id.float().mean().item()
                action_mean = timestep_action_id.float().mean().item()
                if noise_mode == "DECOUPLED":
                    print(
                        f"[NOISE] Mode={noise_mode} | Video: Beta({self.config.video_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode} Uniform mean_t={action_mean:.0f}"
                    )
                elif noise_mode == "HIGH_NOISE_EMPHASIS":
                    print(
                        f"[NOISE] Mode={noise_mode} | Video+Action: Beta({self.config.high_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode}"
                    )
                else:
                    print(
                        f"[NOISE] Mode={noise_mode} | Video+Action: Uniform mean_t={video_mean:.0f} | Action: {action_mode}"
                    )
                self._noise_logged = True
        else:
            noise_action = None
            timestep_action_id = None

        timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
        _, num_frames, num_channels, height, width = noise.shape
        # DiT patch_embedding uses stride (1,2,2), so sequence length is num_frames * (H//2) * (W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        seq_len = num_frames * tokens_per_frame

        timestep = self.scheduler.timesteps[timestep_id].to(self._device)
        noisy_latents = self.scheduler.add_noise(
            latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(
            latents, noise, timestep
        ).transpose(1, 2)

        if actions.numel() > 0:
            timestep_action = self.scheduler.timesteps[timestep_action_id].to(
                self._device
            )
            noisy_actions = self.scheduler.add_noise(
                actions.flatten(0, 1),
                noise_action.flatten(0, 1),
                timestep_action.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
            training_target_action = self.scheduler.training_target(
                actions, noise_action, timestep_action
            )
        else:
            timestep_action = None
            noisy_actions = None
            training_target_action = None
        self._record_profile_time(
            profile_metrics,
            "timing_noise_schedule_metric",
            noise_schedule_start,
            latents.device,
        )

        # Compute loss
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self._device).type
        ):
            dit_start = self._profile_start(noisy_latents.device)
            if actions.numel() > 0:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2),
                    timestep=timestep,
                    clip_feature=clip_feas,
                    y=ys,
                    context=prompt_embs,
                    seq_len=seq_len,
                    state=state_features,
                    action=noisy_actions,
                    timestep_action=timestep_action,
                    past_obs_tokens=past_obs_tokens,
                )
            else:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2),
                    timestep=timestep,
                    timestep_action=timestep_action,
                    clip_feature=clip_feas,
                    y=ys,
                    context=prompt_embs,
                    seq_len=seq_len,
                    state=state_features,
                    past_obs_tokens=past_obs_tokens,
                )
            self._record_profile_time(
                profile_metrics,
                "timing_dit_forward_metric",
                dit_start,
                video_noise_pred.device,
            )

            # Per-sample dynamics loss
            loss_compute_start = self._profile_start(video_noise_pred.device)
            # DiT patch_embedding uses stride (1,2,2), so output spatial size can be smaller than
            # latent when H or W is odd (e.g. latent 11x20 -> model output 10x20). Crop target to match.
            if training_target.shape != video_noise_pred.shape:
                training_target = training_target[
                    ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
                ]
            dynamics_loss_per_sample = torch.nn.functional.mse_loss(
                video_noise_pred.float(), training_target.float(), reduction="none"
            ).mean(dim=(1, 3, 4))  # shape: [B, ...]

            weight_dynamics = dynamics_loss_per_sample * self.scheduler.training_weight(
                timestep.flatten(0, 1)
            ).unflatten(0, (noise.shape[0], noise.shape[1])).to(self._device)
            weighted_dynamics_loss = weight_dynamics.mean()

            if actions.numel() > 0:
                action_loss_per_sample = (
                    torch.nn.functional.mse_loss(
                        action_noise_pred.float(),
                        training_target_action.float(),
                        reduction="none",
                    )
                    * action_mask
                )  # shape: [B, ...]
                has_real_action_mask = has_real_action.float().view(
                    -1, *([1] * (action_loss_per_sample.ndim - 1))
                )
                action_loss_per_sample = has_real_action_mask * action_loss_per_sample
                weight_action = action_loss_per_sample.mean(
                    dim=2
                ) * self.scheduler.training_weight(
                    timestep_action.flatten(0, 1),
                ).unflatten(0, (noise_action.shape[0], noise_action.shape[1])).to(
                    self._device
                )
                weighted_action_loss = weight_action.mean()
                loss = weighted_dynamics_loss + weighted_action_loss
            else:
                weighted_action_loss = torch.tensor(0.0, device=self._device)
                loss = weighted_dynamics_loss
            self._record_profile_time(
                profile_metrics,
                "timing_loss_compute_metric",
                loss_compute_start,
                loss.device,
            )

        # Record log
        self._record_profile_time(
            profile_metrics,
            "timing_total_forward_metric",
            forward_start,
            loss.device,
        )
        output_dict = {
            "loss": loss,
            "dynamics_loss": weighted_dynamics_loss,
            "action_loss": weighted_action_loss,
        }
        if self.use_vggt_geometry_adapter:
            output_dict.update(getattr(self, "_last_vggt_geometry_metrics", {}))
        output_dict.update(profile_metrics)

        return BatchFeature(data=output_dict)

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise

    def _get_caches(
        self,
        kv_caches_input: list[KVCacheType],
    ) -> list[KVCacheType]:
        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            assert len(kv_caches_input) == 2
            if self.ip_rank == 0:
                kv_caches = [kv_caches_input[0]]
            else:
                kv_caches = [kv_caches_input[1]]
        else:
            assert len(kv_caches_input) <= 2
            kv_caches = [kv_caches_input[0]]
            if self.cfg_scale != 1.0:
                kv_caches.append(kv_caches_input[1])
        return kv_caches

    def _prepare_text_inputs(
        self, data: BatchFeature
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:

        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            if self.ip_rank == 0:
                text_inputs = [(data["text"], data["text_attention_mask"])]
            else:
                text_inputs = [
                    (data["text_negative"], data["text_attention_mask_negative"])
                ]
        else:
            text_inputs = [(data["text"], data["text_attention_mask"])]
            if self.cfg_scale != 1.0:
                text_inputs.append(
                    (data["text_negative"], data["text_attention_mask_negative"])
                )
        return text_inputs

    def _run_diffusion_steps(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        kv_caches: list[KVCacheType],
        crossattn_caches: list[KVCacheType],
        kv_cache_metadata: dict[str, bool | int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        predictions = []
        for index, prompt_emb in enumerate(context):
            kv_cache = kv_caches[index]
            crossattn_cache = crossattn_caches[index]
            obs_noise_pred, action_noise_pred, updated_kv_caches = self.model(
                noisy_input,
                timestep,
                action=action,
                timestep_action=timestep_action,
                state=state,
                context=prompt_emb,
                seq_len=seq_len,
                y=y,
                clip_feature=clip_feature,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_frame=kv_cache_metadata["start_frame"],
            )
            if kv_cache_metadata["update_kv_cache"]:
                for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                    kv_cache[block_index] = updated_kv_cache.clone()
            obs_noise_pred = obs_noise_pred.clone()
            if action_noise_pred is not None:
                action_noise_pred = action_noise_pred.clone()
            else:
                action_noise_pred = torch.tensor(
                    0.0, device=obs_noise_pred.device
                )  # dummy action noise prediction
            predictions.append((obs_noise_pred, action_noise_pred))
        return self._exchange_predictions(predictions)

    def _exchange_predictions(
        self,
        predictions: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.ip_size == 1:
            return predictions

        assert len(predictions) == 1
        my_predictions = list(predictions[0])

        other_predictions = [torch.empty_like(pred) for pred in my_predictions]

        send_ops = [
            dist.P2POp(
                op=dist.isend,
                tensor=pred,
                group_peer=(self.ip_rank + 1) % self.ip_size,
                group=self.ip_group,
            )
            for pred in my_predictions
        ]
        recv_ops = [
            dist.P2POp(
                op=dist.irecv,
                tensor=other_pred,
                group_peer=(self.ip_rank + 1) % self.ip_size,
                group=self.ip_group,
            )
            for other_pred in other_predictions
        ]
        ops = send_ops + recv_ops

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        output_predictions: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in range(self.ip_size)
        ]
        output_predictions[self.ip_rank] = tuple(my_predictions)
        output_predictions[(self.ip_rank + 1) % self.ip_size] = tuple(other_predictions)
        assert all(isinstance(pred, tuple) for pred in output_predictions)
        return cast(list[tuple[torch.Tensor, torch.Tensor]], output_predictions)

    def should_run_model(self, index, current_timestep, prev_predictions):

        if not self.dynamic_cache_schedule:
            return self.dit_step_mask[index]

        # Always run first 2 steps to establish history
        if len(prev_predictions) < 2:
            return True

        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            return False
        elif self.skip_countdown == 1:
            self.skip_countdown = 0
            return True

        v_last = prev_predictions[-1][1].flatten(1).float()
        v_prev = prev_predictions[-2][1].flatten(1).float()
        sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()

        thresholds = [0.95, 0.93]
        countdowns = [4, 2]

        for threshold, countdown in zip(thresholds, countdowns):
            if sim > threshold:
                self.skip_countdown = countdown
                return False

        return True

    def _lazy_joint_video_action_vggt_full(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:
        start_time = time.perf_counter()
        profile_metrics: dict[str, torch.Tensor] = {}
        preprocess_start = self._profile_start(self._device)
        self.set_frozen_modules_to_eval_mode()
        data = action_input

        raw_past_videos_for_vggt = data.get("past_images", None)
        if raw_past_videos_for_vggt is None:
            raise ValueError(
                "WNM-3D inference requires past_images from the InteriorGS transform."
            )

        videos = data["images"]
        state_features = action_input.state

        videos = rearrange(videos, "b t h w c -> b c t h w")
        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            assert videos.min() >= -1.0 and videos.max() <= 1.0, (
                "videos must be in [-1,1] range"
            )

        videos = videos.to(device=self._device, dtype=torch.bfloat16)
        state_features = state_features.to(device=self._device, dtype=torch.bfloat16)

        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)
        self._record_profile_time(
            profile_metrics,
            "preprocess",
            preprocess_start,
            self._device,
        )

        text_start = self._profile_start(self._device)
        if self.enable_cfg and self.ip_size > 1:
            if self.ip_rank == 0:
                text_inputs = [(data["text"], data["text_attention_mask"])]
            else:
                text_inputs = [
                    (data["text_negative"], data["text_attention_mask_negative"])
                ]
        else:
            text_inputs = [(data["text"], data["text_attention_mask"])]
            if self.enable_cfg:
                text_inputs.append(
                    (data["text_negative"], data["text_attention_mask_negative"])
                )
        prompt_embs = [
            self.encode_prompt(text, attention_mask).to(self._device)
            for text, attention_mask in text_inputs
        ]
        self._record_profile_time(
            profile_metrics,
            "text",
            text_start,
            self._device,
        )

        _, _, num_frames, height, width = videos.shape
        image = videos[:, :, :1].transpose(1, 2)
        clip_start = self._profile_start(self._device)
        clip_feas, ys, _ = self.encode_image(image, num_frames, height, width)
        self._record_profile_time(
            profile_metrics,
            "clip",
            clip_start,
            self._device,
        )

        vae_start = self._profile_start(self._device)
        latents = self.encode_video(
            videos,
            self.tiled,
            (self.tile_size_height, self.tile_size_width),
            (self.tile_stride_height, self.tile_stride_width),
        ).to(self._device)
        self._record_profile_time(
            profile_metrics,
            "vae",
            vae_start,
            self._device,
        )
        if latents.shape[2] <= 1:
            raise ValueError(
                f"WNM-3D inference expects at least 2 target latents, got {latents.shape}"
            )

        vggt_start = self._profile_start(self._device)
        past_obs_tokens = self._build_vggt_past_obs_tokens(
            raw_past_videos_for_vggt,
            target_frames=latents.shape[2],
            target_grid_size=(latents.shape[3] // 2, latents.shape[4] // 2),
            device=latents.device,
            dtype=latents.dtype,
        )
        self._record_profile_time(
            profile_metrics,
            "vggt",
            vggt_start,
            self._device,
        )
        if past_obs_tokens is None:
            raise ValueError("WNM-3D inference expected non-null past_obs_tokens.")
        past_obs_tokens = past_obs_tokens.to(self._device)
        clip_feas = clip_feas.to(self._device)
        ys = ys.to(self._device)

        batch_size, _, latent_frames, latent_h, latent_w = latents.shape
        num_blocks = (latent_frames - 1) // self.num_frame_per_block
        if num_blocks <= 0:
            raise ValueError(
                f"Invalid latent frame count for WNM-3D inference: {latent_frames=} "
                f"{self.num_frame_per_block=}."
            )
        total_action_horizon = num_blocks * self.model.num_action_per_block
        total_state_horizon = num_blocks * self.model.num_state_per_block
        if state_features.shape[1] != total_state_horizon:
            if state_features.shape[1] <= 0:
                state_dim = (
                    state_features.shape[-1]
                    if state_features.ndim >= 3
                    else int(getattr(self.model, "max_state_dim", 64))
                )
                state_features = torch.zeros(
                    batch_size,
                    total_state_horizon,
                    state_dim,
                    device=self._device,
                    dtype=torch.bfloat16,
                )
            elif state_features.shape[1] < total_state_horizon:
                pad = state_features[:, -1:].expand(
                    -1,
                    total_state_horizon - state_features.shape[1],
                    -1,
                )
                state_features = torch.cat([state_features, pad], dim=1)
            else:
                state_features = state_features[:, :total_state_horizon]

        noise_obs = self.generate_noise(
            latents.shape,
            seed=self.seed,
            device=self._device,
            dtype=torch.bfloat16,
        ).transpose(1, 2)
        noise_action = self.generate_noise(
            (batch_size, total_action_horizon, self.model.action_dim),
            seed=self.seed,
            device=self._device,
            dtype=torch.bfloat16,
        )

        tokens_per_frame = (latent_h // 2) * (latent_w // 2)
        seq_len = latent_frames * tokens_per_frame
        noisy_input = noise_obs
        noisy_input_action = noise_action

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(
            self.num_inference_steps,
            device=latents.device,
            shift=self.sigma_shift,
        )
        sample_scheduler_action.set_timesteps(
            self.num_inference_steps,
            device=latents.device,
            shift=self.sigma_shift,
        )

        if getattr(self.config, "decouple_inference_noise", False):
            video_final_noise = float(
                getattr(self.config, "video_inference_final_noise", 0.0)
            )
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = (
                sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max
                + video_final_noise
            )
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(
                torch.int64
            )
            if self.ip_rank == 0:
                print(
                    "WNM-3D full inference: video sigmas "
                    f"{sigma_max:.3f} -> {sample_scheduler.sigmas[-1].item():.3f}"
                )

        device_type = torch.device(self._device).type
        prev_predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.skip_countdown = 0
        dit_compute_steps = 0
        local_model_forwards = 0
        dit_start = self._profile_start(self._device)
        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            video_timestep = current_timestep
            action_timestep = sample_scheduler_action.timesteps[index]
            timestep = (
                torch.ones(
                    [batch_size, latent_frames],
                    device=latents.device,
                    dtype=torch.int64,
                )
                * video_timestep
            )
            timestep_action = (
                torch.ones(
                    [batch_size, total_action_horizon],
                    device=latents.device,
                    dtype=torch.int64,
                )
                * action_timestep
            )

            should_run_model = not self.enable_dit_cache or self.should_run_model(
                index, current_timestep, prev_predictions
            )
            if should_run_model:
                predictions = []
                for prompt_emb in prompt_embs:
                    with torch.amp.autocast(
                        dtype=torch.bfloat16, device_type=device_type
                    ):
                        obs_noise_pred, action_noise_pred = self.model(
                            noisy_input.transpose(1, 2),
                            timestep=timestep,
                            action=noisy_input_action,
                            timestep_action=timestep_action,
                            state=state_features,
                            context=prompt_emb,
                            seq_len=seq_len,
                            y=ys,
                            clip_feature=clip_feas,
                            past_obs_tokens=past_obs_tokens,
                        )
                    if action_noise_pred is None:
                        raise ValueError(
                            "WNM-3D inference requires action noise prediction."
                        )
                    predictions.append(
                        (obs_noise_pred.clone(), action_noise_pred.clone())
                    )
                    local_model_forwards += 1

                if self.enable_cfg:
                    if self.ip_size > 1:
                        predictions = self._exchange_predictions(predictions)
                    flow_pred_cond, flow_pred_cond_action = predictions[0]
                    flow_pred_uncond, _ = predictions[1]
                    flow_pred = flow_pred_uncond + self.cfg_scale * (
                        flow_pred_cond - flow_pred_uncond
                    )
                    flow_pred_action = flow_pred_cond_action
                else:
                    flow_pred, flow_pred_action = predictions[0]
                dit_compute_steps += 1

                prev_predictions.append((current_timestep, flow_pred, flow_pred_action))
                if len(prev_predictions) > 2:
                    prev_predictions.pop(0)
            else:
                if not prev_predictions:
                    raise RuntimeError(
                        "DiT cache requested before any WNM-3D prediction was computed."
                    )
                _, flow_pred, flow_pred_action = prev_predictions[-1]

            noisy_input = sample_scheduler.step(
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep,
                sample=noisy_input,
                step_index=index,
                return_dict=False,
            )[0]
            noisy_input_action = sample_scheduler_action.step(
                model_output=flow_pred_action,
                timestep=action_timestep,
                sample=noisy_input_action,
                step_index=index,
                return_dict=False,
            )[0]
        self._record_profile_time(
            profile_metrics,
            "dit",
            dit_start,
            self._device,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.ip_rank == 0:
            model_forwards = dit_compute_steps * (2 if self.enable_cfg else 1)
            cfg_status = f"on(scale={self.cfg_scale:g})" if self.enable_cfg else "off"
            print(
                "WNM-3D full inference: "
                f"video={tuple(noisy_input.shape)} action={tuple(noisy_input_action.shape)} "
                f"steps={dit_compute_steps}/{len(sample_scheduler.timesteps)} "
                f"model_forwards={model_forwards} "
                f"model_forwards_per_rank={local_model_forwards} cfg={cfg_status} "
                f"dit_cache={'on' if self.enable_dit_cache else 'off'} "
                f"time={time.perf_counter() - start_time:.2f}s"
            )
            if self.profile_module_times:
                profile_parts = [
                    f"{name}={float(profile_metrics[name].item()):.3f}s"
                    for name in ("preprocess", "text", "clip", "vae", "vggt", "dit")
                ]
                vggt_metrics = getattr(self, "_last_vggt_geometry_metrics", {})
                for key, label in (
                    ("vggt_timing_prepare_images_metric", "vggt_prepare"),
                    ("vggt_timing_aggregator_metric", "vggt_encoder"),
                    ("vggt_timing_adapter_metric", "vggt_adapter"),
                ):
                    value = vggt_metrics.get(key)
                    if torch.is_tensor(value):
                        profile_parts.append(f"{label}={float(value.item()):.3f}s")
                print("WNM-3D profile: " + " ".join(profile_parts))

        action_pred = noisy_input_action[:, : self.action_horizon]
        return BatchFeature(
            data={
                "action_pred": action_pred,
                "video_pred": noisy_input.transpose(1, 2),
            }
        )

    def lazy_joint_video_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        if (
            self.use_vggt_geometry_adapter
            and action_input.get("past_images", None) is not None
        ):
            return self._lazy_joint_video_action_vggt_full(
                backbone_output, action_input
            )

        start_time = time.perf_counter()

        # Tracking time taken on GPU for various operations.
        start_text_encoder_event = torch.cuda.Event(enable_timing=True)
        end_text_encoder_event = torch.cuda.Event(enable_timing=True)
        start_image_encoder_event = torch.cuda.Event(enable_timing=True)
        end_image_encoder_event = torch.cuda.Event(enable_timing=True)
        start_vae_event = torch.cuda.Event(enable_timing=True)
        end_vae_event = torch.cuda.Event(enable_timing=True)
        start_kv_event = torch.cuda.Event(enable_timing=True)
        end_kv_event = torch.cuda.Event(enable_timing=True)
        start_diffusion_events = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(self.num_inference_steps)
        ]
        end_diffusion_events = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(self.num_inference_steps)
        ]

        self.set_frozen_modules_to_eval_mode()
        data = action_input

        videos = data["images"]

        state_features = action_input.state

        videos = rearrange(videos, "b t h w c -> b c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            videos = videos.to(dtype=self.dtype)
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(
                0, 2, 1, 3, 4
            )  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, (
                "videos must be in [-1,1] range"
            )
            videos = videos.to(dtype=self.dtype)

        state_features = state_features.to(dtype=torch.bfloat16)
        videos = videos.to(dtype=torch.bfloat16)

        # Wan 5B: same as training — resize to target resolution so latent matches DiT
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        if self.language is None:
            print("language is None, reset current_start_frame to 0")
            self.language = data["text"]
            self.current_start_frame = 0
        elif not torch.equal(self.language, data["text"]):
            print("language changed, reset current_start_frame to 0")
            self.current_start_frame = 0
            self.language = data["text"]
        elif videos.shape[2] == 1:
            print("videos.shape[2] == 1, reset current_start_frame to 0")
            self.current_start_frame = 0
        elif self.current_start_frame >= self.model.local_attn_size:
            print(
                "current_start_frame >= local_attn_size, reset current_start_frame to 0"
            )
            self.current_start_frame = 0

        if self.ip_rank == 0:
            print("videos shape", videos.shape, self.num_frames)

        start_text_encoder_event.record()

        text_inputs = self._prepare_text_inputs(data)
        prompt_embs = [
            self.encode_prompt(text, attention_mask)
            for text, attention_mask in text_inputs
        ]

        end_text_encoder_event.record()

        start_image_encoder_event.record()

        _, _, num_frames, height, width = videos.shape
        if videos.shape[2] == 4 or videos.shape[2] == 9:
            # special case for real-world eval where language is updated
            image = videos[:, :, -1:].transpose(1, 2)
        else:
            image = videos[:, :, :1].transpose(1, 2)

        if self.current_start_frame == 0:
            clip_feas, ys, image = self.encode_image(
                image, self.num_frames, height, width
            )
            self.clip_feas = clip_feas.to(dtype=image.dtype)
            self.ys = ys.to(dtype=image.dtype)

        assert self.clip_feas is not None and self.ys is not None, (
            "clip_feas and ys must be set"
        )

        end_image_encoder_event.record()

        start_vae_event.record()

        if latent_video is not None and self.current_start_frame != 0:
            image = latent_video
            if self.ip_rank == 0:
                logger.debug("Reusing latent video with shape %s", tuple(image.shape))
        elif self.current_start_frame != 0:
            # this is for real world execution
            if (videos.shape[2] - 1) // 4 == self.num_frame_per_block:
                logger.debug("No additional video block is required")
            elif videos.shape[2] // 4 != self.num_frame_per_block:
                # Repeating videos along dim 2.
                repeat_factor = self.num_frame_per_block // (videos.shape[2] // 4)
                videos = torch.repeat_interleave(videos, repeat_factor, dim=2)

                first_frame = videos[:, :, 0:1]  # Extract first frame
                videos = torch.cat([first_frame, videos], dim=2)
            else:
                first_frame = videos[:, :, 0:1]  # Extract first frame
                videos = torch.cat([first_frame, videos], dim=2)

            image = self.vae.encode(
                videos,
                tiled=self.tiled,
                tile_size=(self.tile_size_height, self.tile_size_width),
                tile_stride=(self.tile_stride_height, self.tile_stride_width),
            )

        end_vae_event.record()

        noise_obs = self.generate_noise(
            (
                image.shape[0],
                image.shape[1],
                self.num_frame_per_block,
                image.shape[3],
                image.shape[4],
            ),
            seed=self.seed,
            device="cuda",
            dtype=torch.bfloat16,
        )
        noise_action = self.generate_noise(
            (image.shape[0], self.action_horizon, self.model.action_dim),
            seed=self.seed,
            device="cuda",
            dtype=torch.bfloat16,
        )
        batch_size, num_channels, num_frames, height, width = noise_obs.shape
        ######### Generate video #########
        # DiT patch_embedding uses stride (1,2,2), so tokens per frame = (H//2)*(W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        frame_seqlen = tokens_per_frame
        seq_len = num_frames * frame_seqlen

        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)

        if self.current_start_frame == 0:
            # Reinitialize KV cache and crossattn cache for each new sequence.
            self.kv_cache1, self.kv_cache_neg = self._create_kv_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
                frame_seqlen=frame_seqlen,
            )
            self.crossattn_cache, self.crossattn_cache_neg = (
                self._create_crossattn_caches(
                    batch_size=batch_size,
                    dtype=noise_obs.dtype,
                    device=noise_obs.device,
                )
            )

        assert self.kv_cache1 is not None
        assert self.kv_cache_neg is not None
        assert self.crossattn_cache is not None
        assert self.crossattn_cache_neg is not None
        kv_caches = self._get_caches(
            [self.kv_cache1, self.kv_cache_neg],
        )
        crossattn_caches = self._get_caches(
            [self.crossattn_cache, self.crossattn_cache_neg],
        )

        start_kv_event.record()

        if self.current_start_frame == 0:
            timestep = (
                torch.ones([batch_size, 1], device=noise_obs.device, dtype=torch.int64)
                * 0
            )
            self._run_diffusion_steps(
                noisy_input=image.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                context=prompt_embs,
                seq_len=frame_seqlen,
                y=self.ys[:, :, 0:1],
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=0,
                    update_kv_cache=True,
                ),
            )
            self.current_start_frame += 1

        timestep = (
            torch.ones(
                [batch_size, self.num_frame_per_block],
                device=noise_obs.device,
                dtype=torch.int64,
            )
            * 0
        )

        if self.current_start_frame != 1:
            current_ref_latents = image[:, -self.num_frame_per_block :]
            if self.current_start_frame <= self.ys.shape[2]:
                y = self.ys[
                    :,
                    :,
                    self.current_start_frame
                    - self.num_frame_per_block : self.current_start_frame,
                ]
            else:
                y = self.ys[:, :, -self.num_frame_per_block :]
            self._run_diffusion_steps(
                noisy_input=current_ref_latents.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                context=prompt_embs,
                seq_len=seq_len,
                y=y,
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=self.current_start_frame - self.num_frame_per_block,
                    update_kv_cache=True,
                ),
            )

        end_kv_event.record()

        noisy_input = noise_obs
        noisy_input_action = noise_action

        # Step 3.1: Spatial denoising loop

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift
        )
        sample_scheduler_action.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift
        )

        # Decoupled inference: video sigmas end at video_final_noise instead of 0
        # This rescales the schedule so video still takes all denoising steps,
        # but ends at a higher noise level (e.g., 1.0 → 0.9 → 0.8 instead of 1.0 → 0.5 → 0.0)
        if self.config.decouple_inference_noise:
            video_final_noise = self.config.video_inference_final_noise
            # Rescale video sigmas: map [sigma_max, 0] -> [sigma_max, video_final_noise]
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = (
                sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max
                + video_final_noise
            )
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(
                torch.int64
            )
            if self.ip_rank == 0:
                print(
                    f"Decoupled inference: video sigmas {sigma_max:.3f} -> {sample_scheduler.sigmas[-1].item():.3f}"
                )

        start_diffusion_events = [
            torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps
        ]
        end_diffusion_events = [
            torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps
        ]
        prev_predictions = []
        self.skip_countdown = 0
        dit_compute_steps = 0
        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            start_diffusion_events[index].record()

            # Get timesteps from respective schedulers
            action_timestep = sample_scheduler_action.timesteps[index]
            video_timestep = sample_scheduler.timesteps[
                index
            ]  # Already rescaled if decoupled

            # set current timestep
            timestep = (
                torch.ones(
                    [batch_size, self.num_frame_per_block],
                    device=noise_obs.device,
                    dtype=torch.int64,
                )
                * video_timestep
            )
            timestep_action = (
                torch.ones(
                    [batch_size, self.action_horizon],
                    device=noise_obs.device,
                    dtype=torch.int64,
                )
                * action_timestep
            )

            # check if we need to run the DIT step
            should_run_model = self.should_run_model(
                index, current_timestep, prev_predictions
            )
            if should_run_model:
                dit_compute_steps += 1
                if (
                    self.current_start_frame + self.num_frame_per_block
                    <= self.ys.shape[2]
                ):
                    y = self.ys[
                        :,
                        :,
                        self.current_start_frame : self.current_start_frame
                        + self.num_frame_per_block,
                    ]
                else:
                    y = self.ys[:, :, -self.num_frame_per_block :]
                predictions = self._run_diffusion_steps(
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    state=state_features,
                    context=prompt_embs,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=self.clip_feas,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(
                        start_frame=self.current_start_frame,
                        update_kv_cache=False,
                    ),
                )
                flow_pred_cond, flow_pred_cond_action = predictions[0]
                flow_pred_uncond, flow_pred_uncond_action = predictions[1]

                flow_pred = flow_pred_uncond + self.cfg_scale * (
                    flow_pred_cond - flow_pred_uncond
                )
                prev_predictions.append(
                    (current_timestep, flow_pred, flow_pred_cond_action)
                )
                max_cache_size = 2
                if len(prev_predictions) > max_cache_size:
                    prev_predictions.pop(0)

            else:
                assert len(prev_predictions) > 0, (
                    "prev_predictions must be set when skipping"
                )
                _, flow_pred, flow_pred_cond_action = prev_predictions[-1]

            end_diffusion_events[index].record()

            # Video: denoising step (uses rescaled schedule if decoupled)
            noisy_input = sample_scheduler.step(
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep,
                sample=noisy_input,
                step_index=index,
                return_dict=False,
            )[0]

            # Action: always fully denoises with standard schedule (1000->0)
            noisy_input_action = sample_scheduler_action.step(
                model_output=flow_pred_cond_action,
                timestep=action_timestep,
                sample=noisy_input_action,
                step_index=index,
                return_dict=False,
            )[0]

        latents = noisy_input
        latents_action = noisy_input_action
        output = latents

        if self.current_start_frame == 1:
            output = torch.cat([image, output], dim=1)
        self.current_start_frame += self.num_frame_per_block

        # Do torch.cuda.synchronize() to ensure all operations are completed before timing.
        # This isn't expected to affect inference performance since it's at the end of an inference step.
        torch.cuda.synchronize()

        total_time = time.perf_counter() - start_time
        text_encoder_time = (
            start_text_encoder_event.elapsed_time(end_text_encoder_event) / 1000
        )
        image_encoder_time = (
            start_image_encoder_event.elapsed_time(end_image_encoder_event) / 1000
        )
        vae_time = start_vae_event.elapsed_time(end_vae_event) / 1000
        kv_creation_time = start_kv_event.elapsed_time(end_kv_event) / 1000
        diffusion_times = [
            s.elapsed_time(e)
            for s, e in zip(start_diffusion_events, end_diffusion_events)
        ]
        diffusion_time = sum(diffusion_times) / 1000
        scheduler_time = (
            total_time
            - kv_creation_time
            - diffusion_time
            - text_encoder_time
            - image_encoder_time
            - vae_time
        )

        if self.ip_rank == 0:
            print(
                f"Time taken: Total {total_time:.2f} seconds, "
                f"Text Encoder {text_encoder_time:.2f} seconds, "
                f"Image Encoder {image_encoder_time:.2f} seconds, "
                f"VAE {vae_time:.2f} seconds, "
                f"KV Cache Creation {kv_creation_time:.2f} seconds, "
                f"Diffusion {diffusion_time:.2f} seconds, "
                f"DIT Compute Steps {dit_compute_steps} steps, "
                f"Scheduler {scheduler_time:.2f} seconds"
            )

        return BatchFeature(
            data={"action_pred": latents_action, "video_pred": output.transpose(1, 2)}
        )

    def post_initialize(self):
        # Move models to the cuda device and set the dtype to bfloat16.
        print("Moving models to the cuda device and setting the dtype to bfloat16.")
        self.model.to(device=self._device, dtype=torch.bfloat16)
        self.text_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.image_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.vae.to(device=self._device, dtype=torch.bfloat16)
        disable_torch_compile = (
            os.getenv("DISABLE_TORCH_COMPILE", "False").lower() == "true"
        )

        # Torch compile the modules. Skip _forward_blocks: Dynamo with fullgraph can fail on
        # shape variation (e.g. x [1,50,C] vs e [1,200,C]); the block aligns e to x at runtime.
        if not disable_torch_compile:
            print(
                "Torch compiling the TextEncoder, ImageEncoder, and VAE modules (Wan _forward_blocks not compiled)."
            )

            self.text_encoder.forward = torch.compile(
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            )(self.text_encoder.forward)

            self.image_encoder.model.visual.forward = torch.compile(
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            )(self.image_encoder.model.visual.forward)

            self.vae.model.encode = torch.compile(
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            )(self.vae.model.encode)
        else:
            print(
                "Skipping torch.compile for TextEncoder, ImageEncoder, and VAE modules."
            )

    def parallelize(self, device_mesh: DeviceMesh) -> None:
        ip_mesh = device_mesh["ip"]
        self.ip_rank = ip_mesh.get_local_rank()
        self.ip_size = ip_mesh.size()
        self.ip_group = ip_mesh.get_group()

        assert self.ip_size == 1 or self.ip_size == 2, "ip_size must be 1 or 2"
        assert self.ip_rank >= 0 and self.ip_rank < self.ip_size, (
            "ip_rank must be in [0, ip_size)"
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
