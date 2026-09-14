# Stage 6.1 Literature Selector Provenance

Branch: `literature-frame-selection` · Base: `701c938` (Stage 6.0 frozen HEAD)
Date: 2026-09-14 · Author: DeepSeek (Stage 6.1 preparation)

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
| weights | Zenodo DOI 10.5281/zenodo.5635735 (`pretrained_models.zip`, 697,584,856 B) — **not downloaded (BLOCKED)** |

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
| weights | `vasnet_models.zip` from the repo's `datasets_models_urls.txt` (Box) — **BLOCKED (host unreachable)** |

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

## New AIC-side files (no new `src/` subpackage, no `third_party/`)

* `src/aic_video_highlight/composition/pgl_sum_selector.py`
* `src/aic_video_highlight/composition/vasnet_selector.py`
* `src/aic_video_highlight/composition/literature_frame_selection.py`
* `src/aic_video_highlight/composition/literature_features.py` (GoogleNet pool5 adapter)
* `tests/test_literature_frame_selection.py`

## Data policy

Training data, benchmark splits, h5 datasets and pretrained weights were **not**
copied into the repository. No weight file is committed. Weights, if obtained,
belong under the AIC unified models root (`configs/environments/*.json`).
