# Stage 6.1 Literature Selector Provenance

Branch: `literature-frame-selection` · Base: `701c938` (Stage 6.0 frozen HEAD)
Date: 2026-09-15 · Stage 6.1 Formal-readiness update

This file is the version-tracked record of the literature code imported for the
Stage 6.1 Frame-Selection arms. The full audit documents live in
`实验记录/Stage6_对VHiCraft的优化/01_Frame_Membership/`.

## PGL-SUM

| Field | Value |
| --- | --- |
| upstream URL | https://github.com/e-apostolidis/PGL-SUM |
| upstream SSH | git@github.com:e-apostolidis/PGL-SUM.git |
| commit SHA | `81d0d6d0ee0470775ad759087deebbce1ceffec3` |
| commit date | 2023-01-30 |
| clone date | 2026-09-14 |
| clone mode | SSH, `--depth 1 --filter=blob:none --sparse` (data/*.rar not fetched) |
| paper | Apostolidis, Balaouras, Mezaris, Patras, "Combining Global and Local Attention with Positional Encoding for Video Summarization", IEEE ISM 2021 |
| license | academic / non-commercial, CERTH-ITI; notice retained |
| weights | Zenodo DOI 10.5281/zenodo.5635735 (`pretrained_models.zip`, 697,584,856 B), SHA256 `652db7763e9573e8d95d242b2e2f23072d4db30ffb73e31ac63990ba0f40c6ef` |
| released checkpoints | 20 total: Table III and Table IV, each SumMe/TVSum × split0..4 |
| Formal policy | Table IV 10 checkpoints, equal-weight arithmetic mean; no Dev-based selection |

Ported / rewritten:

| Selected source | Destination | Changed | Why |
| --- | --- | --- | --- |
| `inference/layers/attention.py` | `composition/pgl_sum_selector.py` | `from layers.attention import SelfAttention` inlined | make one self-contained module |
| `inference/layers/summarizer.py` | `composition/pgl_sum_selector.py` | same inlining | same |
| `inference/knapsack_implementation.py` | `composition/literature_frame_selection.py` | none (`knapSack`) | reuse author DP |
| `inference/generate_summary.py` | `composition/literature_frame_selection.py` | re-expressed as an FS-0 subset adapter | enforce `selected ⊆ FS0` and KEEP/DROP-only |

Not ported: `data/`, `evaluation/`, `model/` (all training), `model/layers/`
(duplicates of `inference/layers/`).

## VASNet

| Field | Value |
| --- | --- |
| upstream URL | https://github.com/ok1zjf/VASNet |
| upstream SSH | git@github.com:ok1zjf/VASNet.git |
| commit SHA | `c3787531486f74789dc5e92758edf51e24f56e6d` |
| commit date | 2019-03-04 |
| clone date | 2026-09-14 |
| clone mode | SSH, full |
| paper | Fajtl, Sadeghi Sokeh, Argyriou, Monekosso, Remagnino, "Summarizing Videos with Attention", ACCV 2018 (arXiv:1812.01969) |
| license | MIT (Jiri Fajtl); `vsum_tools.py`/`knapsack.py`/`cpd_*.py` courtesy KaiyangZhou (MIT) |
| original weights | `vasnet_models.zip` from the repo's `datasets_models_urls.txt` (Box) — unavailable; provenance cannot be recovered |
| adopted secondary weights | 10 VASNet checkpoints committed by the XAI-SUM authors at `e-apostolidis/XAI-SUM` commit `375bb1c8fc7bbd0afdd8933bb89888e5ad811234` |
| disclosure | `NOT_ORIGINAL_VASNET_BOX_CHECKPOINT`; `UNKNOWN_PROVENANCE`; secondary baseline only |
| Formal policy | SumMe/TVSum × split0..4, all 10 checkpoints, equal-weight arithmetic mean; no Dev-based selection |

Ported / rewritten:

| Selected source | Destination | Changed | Why |
| --- | --- | --- | --- |
| `vasnet_model.py` | `composition/vasnet_selector.py` | `from config import *` / `from layer_norm import *` replaced by inlined `LayerNorm` | remove training-only imports; keep exact state_dict keys |
| `layer_norm.py` | `composition/vasnet_selector.py` | inlined, unchanged | dependency of the model |
| `vsum_tools.py` (`generate_summary`) | `composition/literature_frame_selection.py` | re-expressed as FS-0 subset adapter | enforce KEEP/DROP-only |
| `knapsack.py` | `composition/literature_frame_selection.py` | `ortools` path replaced by the author DP; 1000× value scaling preserved | avoid a heavy/old dependency without changing the objective |
| `cpd_auto.py` | `composition/literature_frame_selection.py` | unchanged | KTS change-point detection |
| `cpd_nonlin.py` | `composition/literature_frame_selection.py` | unchanged | KTS DP core |

Not ported: `main.py` (training/eval loop), `sys_utils.py`, `create_split.py`,
`splits/`, `config.py`, the `ortools` import path.

XAI-SUM names the architecture submodule `attention.*`, while the original
VASNet implementation and this port name it `att.*`.  The loader performs only
the deterministic `attention.` → `att.` prefix remap and then uses strict
state-dict loading.  Parameter tensors, architecture, forward path and score
mathematics are unchanged.

## New AIC-side files (no new `src/` subpackage, no `third_party/`)

* `src/aic_video_highlight/composition/pgl_sum_selector.py`
* `src/aic_video_highlight/composition/vasnet_selector.py`
* `src/aic_video_highlight/composition/literature_frame_selection.py`
* `src/aic_video_highlight/composition/literature_features.py` (GoogleNet pool5 adapter)
* `tests/test_literature_frame_selection.py`

## Data policy

Training data, benchmark splits, h5 datasets and pretrained weights were **not**
copied into the repository. No weight file is committed. Audited weights live
only under the external AutoDL models root.

## Feature compatibility boundary

Both upstream inference paths consume the same precomputed
`eccv16_dataset_*_google_pool5.h5` format: 1024-D GoogleNet pool5 features and
original-frame `picks`.  PGL-SUM's paper states that frames are sampled at 2
FPS and represented by 1024-D GoogleNet pool5 ImageNet features.  Neither
PGL-SUM, VASNet nor XAI-SUM ships the original feature-extraction program or a
weight/preprocessing checksum for those H5 tensors.  The AIC adapter therefore
uses torchvision GoogLeNet ImageNet-1K V1 in evaluation mode, removes `fc`, and
labels the result `FEATURE_EXTRACTOR_COMPATIBLE_REPRODUCTION`; it does not
claim bitwise identity with the unavailable original Caffe features.

Both author inference implementations also consume precomputed
`change_points`, `n_frame_per_seg` and `picks`.  They do not publish the
`ncp`/`vmax` values used to create the benchmark H5 files.  Stage 6.1 therefore
freezes explicit AIC-side KTS compatibility parameters in its external Formal
protocol and does not attribute those values to the original authors.
