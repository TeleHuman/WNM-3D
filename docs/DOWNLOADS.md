# Downloads

This guide defines the reference artifact preparation workflow for reproducing
WNM-3D evaluation results and, optionally, initializing Stage-I training.

> [!IMPORTANT]
> The released **WNM-3D checkpoint** and **benchmark datasets** are the core
> artifacts required for reproduction. The component initialization weights
> in Section 3 are required only when training WNM-3D from Stage-I.

## Artifact Overview

| Priority | Artifact | Purpose | Required for reproduction |
| --- | --- | --- | :---: |
| 1 | WNM-3D Stage-III checkpoint | Released policy for evaluation and inference | Yes |
| 2 | InteriorGS and GN-Matrix | Simulator scenes and benchmark annotations | Yes |
| 3 | Wan, UMT5-XXL, and VGGT-Ω | Stage-I model initialization | Training only |

All commands use the canonical paths shown in the
[project layout](../README.md#-layout). The launchers also accept
path overrides for non-default storage configurations.

## Prerequisites

Complete the [installation guide](INSTALLATION.md) before downloading project
artifacts. WNM-3D and GN0 should be available as sibling repositories, as
described in the [project layout](../README.md#-layout).

Install the Hugging Face Hub CLI with Xet support and authenticate:

```bash
python -m pip install --upgrade huggingface_hub hf_xet
hf auth login
```

Some artifacts are access-controlled. Review and accept the applicable terms
on each Hugging Face repository before starting a download.

## 1. Download the WNM-3D Checkpoint

The released Stage-III checkpoint is the authoritative model artifact for
evaluation and inference. From the WNM-3D repository root, run:

```bash
conda activate wnm_3d
mkdir -p checkpoints

hf download TeleEmbodied/WNM-3D \
  --include "wnm_3d_stage3_release/**" \
  --local-dir checkpoints
```

The full checkpoint is self-contained: evaluation and inference do not require
the standalone Wan, UMT5-XXL, or VGGT-Ω initialization weights.


## 2. Download the Benchmark Datasets

Official GN-Bench reproduction requires both datasets:

- [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS)
  provides the simulator-ready 3D scenes.
- [GN-Matrix](https://huggingface.co/datasets/TeleEmbodied/GN-Matrix)
  provides benchmark episodes and navigation annotations.

Accept the InteriorGS access conditions before proceeding. Then run the
following commands from the GN0 repository root with the `bae` environment
active:

```bash
conda activate bae
python -m pip install --upgrade huggingface_hub hf_xet
hf auth login

hf download spatialverse/InteriorGS \
  --repo-type dataset \
  --local-dir data/scene_datasets/InteriorGS

hf download TeleEmbodied/GN-Matrix \
  --repo-type dataset \
  --local-dir data/datasets/GN_Matrix
```

Keep the downloaded directory names unchanged. GN0 resolves benchmark scenes
and annotations from these default locations. For the expected on-disk
structure, refer to the [project layout](../README.md#-layout).

At this point, all artifacts required to evaluate the released WNM-3D policy
on GN-Bench are in place.

## 3. Download Initialization Weights for Training (Optional)

> [!NOTE]
> Skip this section when reproducing results with the released checkpoint.
> These weights are needed only to initialize Stage-I training.

From the WNM-3D repository root with the `wnm_3d` environment active, run:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir checkpoints/Wan2.2-TI2V-5B

hf download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir checkpoints/Wan2.1-I2V-14B-480P

hf download google/umt5-xxl \
  --local-dir checkpoints/umt5-xxl

hf download facebook/VGGT-Omega vggt_omega_1b_512.pt \
  --local-dir checkpoints
```

The Stage-I launcher automatically downloads the Wan and UMT5-XXL resources
when they are absent. VGGT-Ω is gated and must be authorized and downloaded
separately from the
[official model repository](https://huggingface.co/facebook/VGGT-Omega).

Training also requires InteriorGS in LeRobot format. Place the prepared dataset
at `WNM-3D/data/interiorgs_lerobot_seen`, or set `INTERIORGS_DATA_ROOT` to a
custom location.
