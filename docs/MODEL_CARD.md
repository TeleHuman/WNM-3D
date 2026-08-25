---
license: fair-noncommercial-research-license
tags:
  - robotics
  - vision-language-navigation
  - embodied-ai
  - world-model
  - video-generation
  - pytorch
---

# WNM-3D

WNM-3D is a generative world navigation model for continuous
vision-language navigation. It converts monocular egocentric RGB history into
persistent geometry-aware scene tokens and jointly generates future views and
navigation actions for closed-loop control.

> **License notice:** The released checkpoints embed VGGT-Ω/DINOv3-derived
> parameters. Use and redistribution are subject to the FAIR Noncommercial
> Research License, the DINOv3 License, and other applicable third-party terms.

- [Project page](https://wnm-3d.github.io/)
- [Paper](https://arxiv.org/abs/2608.07267)
- [Source code and documentation](https://github.com/TeleHuman/WNM-3D)
- [GN0 / GN-Bench](https://github.com/TeleHuman/GN0)
- [GN-Matrix dataset](https://huggingface.co/datasets/TeleEmbodied/GN-Matrix)

## Released Checkpoints

| Directory | Training stage | Recommended use |
| --- | --- | --- |
| `wnm_3d_stage1_release` | Offline A\* SFT | Stage-I analysis and initialization |
| `wnm_3d_stage2_release` | Closed-loop DAgger-SFT | Stage-II analysis and initialization |
| `wnm_3d_stage3_release` | Counterfactual DanceGRPO | Evaluation and inference |

The Stage-III checkpoint is the primary released policy. Each directory is a
self-contained inference checkpoint; standalone Wan, UMT5-XXL, and VGGT-Ω
initialization weights are not required for evaluation.

## Download

```bash
hf download TeleEmbodied/WNM-3D \
  --include "wnm_3d_stage3_release/**" \
  --local-dir checkpoints
```

## Inference

Install WNM-3D by following the
[installation guide](https://github.com/TeleHuman/WNM-3D/blob/main/docs/INSTALLATION.md),
then launch a single policy replica:

```bash
bash scripts/inference/wnm_3d_server.sh \
  --model-path checkpoints/wnm_3d_stage3_release \
  --cuda-devices 0 \
  --num-replicas 1 \
  --base-port 8000
```

Run the GN-Bench client from a sibling GN0 checkout:

```bash
cd ../GN0
bash scripts/evaluation/eval_remote.sh \
  --exp-config configs/gn_bench/interiorgs/test_unseen.yaml \
  --enable-stall-recovery \
  --num-gpus 1 \
  --result-dir tmp/eval/wnm_3d_test_unseen
```

The reference single-replica deployment was validated on an NVIDIA H100 80 GB
GPU and used approximately 27 GiB of GPU memory after loading. A released
checkpoint occupies approximately 26 GiB. These figures are observations from
the reference configuration, not strict minimum requirements.

## Intended Use and Limitations

WNM-3D is intended for research on embodied navigation, world models, and
closed-loop vision-language navigation. The released policy was developed and
evaluated with the InteriorGS scenes and GN-Matrix task annotations used by
GN-Bench. Performance may not transfer to new simulators, sensors, scene
distributions, languages, or physical robots without additional validation.

Generated actions can fail or behave unexpectedly. Do not use the model as a
safety-critical controller or deploy it around people, property, or physical
systems without appropriate safeguards and human oversight.

## License

The released checkpoints contain original WNM-3D parameters together with
VGGT-Ω/DINOv3-derived parameters. They are not licensed solely under
Apache-2.0. Use and redistribution are subject to all applicable terms,
including the FAIR Noncommercial Research License and DINOv3 License.

The model repository provides the complete
[Apache-2.0](https://huggingface.co/TeleEmbodied/WNM-3D/blob/main/LICENSES/APACHE-2.0),
[FAIR Noncommercial Research](https://huggingface.co/TeleEmbodied/WNM-3D/blob/main/LICENSES/FAIR_NONCOMMERCIAL_RESEARCH_LICENSE),
and [DINOv3](https://huggingface.co/TeleEmbodied/WNM-3D/blob/main/LICENSES/DINOV3_LICENSE.md)
license texts, together with its
[Third-Party Notices](https://huggingface.co/TeleEmbodied/WNM-3D/blob/main/THIRD_PARTY_NOTICES.md).
The source repository documents the complete
[component-level license boundaries](https://github.com/TeleHuman/WNM-3D/blob/main/docs/THIRD_PARTY_NOTICES.md).
Datasets and separately downloaded initialization weights remain subject to
their providers' terms.

## Citation

```bibtex
@article{huang2026wnm_3d,
  title={WNM-3D: A World Navigation Model with 3D Scene Conditioning for Closed-Loop VLN},
  author={Huang, Yuehao and Wu, Yunzi and Zhang, Xiaotao and Li, Xinhai and Dong, Jiankun and Lv, Jiajun and Zhang, Chi and Bai, Chenjia and Liu, Yong and Li, Xuelong},
  journal={arXiv preprint arXiv:2608.07267},
  year={2026}
}
```
