# Third-Party Notices

WNM-3D includes source code derived from third-party projects. The repository-
level Apache License 2.0 applies only where a file or directory does not state
different terms. Copyright notices in the source files remain in effect.

## DreamZero

WNM-3D is derived in part from
[DreamZero](https://github.com/dreamzero0/dreamzero), Copyright 2025 NVIDIA
Corporation. Those portions are distributed under the Apache License 2.0; see
[`LICENSE`](../LICENSE) and [`COPYRIGHT`](../COPYRIGHT).

## Wan

The video-model components under
`gammanav/vln/model/wnm_3d/modules/` include code from the Alibaba Wan Team.
The relevant source files retain their original copyright notices and are
distributed under the Apache License 2.0 included in [`LICENSE`](../LICENSE).

## VGGT-Ω and DINOv3

The vendored source under
`gammanav/vln/model/wnm_3d/modules/vggt_omega/` includes code from VGGT-Ω and
DINOv3, Copyright Meta Platforms, Inc. and affiliates. This source is not
covered by the repository-level Apache License 2.0:

- VGGT-Ω portions are distributed under the
  [FAIR Noncommercial Research License](../LICENSES/FAIR_NONCOMMERCIAL_RESEARCH_LICENSE).
- DINOv3-derived portions are distributed under the
  [DINOv3 License](../LICENSES/DINOV3_LICENSE.md).

The applicable terms are identified at file level as follows:

| Vendored path | Origin | Applicable license |
| --- | --- | --- |
| `vggt_omega/__init__.py` | VGGT-Ω | FAIR Noncommercial Research License |
| `vggt_omega/models/__init__.py` | VGGT-Ω | FAIR Noncommercial Research License |
| `vggt_omega/models/aggregator.py` | VGGT-Ω | FAIR Noncommercial Research License |
| `vggt_omega/models/layers/__init__.py` | VGGT-Ω | FAIR Noncommercial Research License |
| `vggt_omega/models/layers/{attention,block,ffn_layers,layer_scale,patch_embed,rms_norm,rope_position_encoding,utils,vision_transformer}.py` | VGGT-Ω and DINOv3 | FAIR Noncommercial Research License **and** DINOv3 License |

Here, `vggt_omega/` is relative to
`gammanav/vln/model/wnm_3d/modules/`. The source files retain the upstream
copyright and license notices and include SPDX identifiers that resolve to the
license texts distributed in [`LICENSES/`](../LICENSES/).

The released WNM-3D checkpoints embed VGGT-Ω/DINOv3-derived parameters and are
therefore also subject to the applicable third-party terms. The model
repository provides copies of these terms alongside the checkpoints.

Review the applicable license before using, modifying, or redistributing these
components. The third-party licenses may impose restrictions beyond the
Apache License 2.0, including use and redistribution conditions.
