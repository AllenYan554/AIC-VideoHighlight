"""VASNet frame-importance model ported for Stage 6.1 literature frame selection.

Upstream provenance
-------------------
* repository : https://github.com/ok1zjf/VASNet
* commit     : c3787531486f74789dc5e92758edf51e24f56e6d (master, 2019-03-04)
* paper      : J. Fajtl, H. Sadeghi Sokeh, V. Argyriou, D. Monekosso,
               P. Remagnino, "Summarizing Videos with Attention", ACCV 2018
               AIU Workshop (arXiv:1812.01969).
* taken from : ``vasnet_model.py`` and ``layer_norm.py``
* changed    : ``from config import *`` / ``from layer_norm import *`` are
               replaced by the explicit :class:`LayerNorm` definition inlined
               here; the model mathematics is unchanged.  The unused ``kb`` /
               ``kc`` layers are kept so upstream checkpoints load with an
               exact key match.
* license    : MIT (Copyright (c) 2018 Jiri Fajtl). Retain the notice.

As with the PGL-SUM port, this file only produces per-frame importance scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer norm variant from upstream ``layer_norm.py`` (courtesy jekbradbury)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SelfAttention(nn.Module):
    """VASNet self-attention cell (upstream ``vasnet_model.py``)."""

    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)

    def forward(self, x):
        n = x.shape[0]

        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1, 0))

        if self.ignore_itself:
            logits[torch.eye(n).byte()] = -float("Inf")

        if self.apperture > 0:
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")

        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1, 0), weights).transpose(1, 0)
        y = self.output_linear(y)

        return y, att_weights_


class VASNet(nn.Module):
    """VASNet regressor; architecture is byte-faithful to upstream."""

    def __init__(self):
        super(VASNet, self).__init__()

        self.m = 1024
        self.hidden_size = 1024

        self.att = SelfAttention(input_size=self.m, output_size=self.m)
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)

    def forward(self, x, seq_len):
        m = x.shape[2]

        x = x.view(-1, m)
        y, att_weights_ = self.att(x)

        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)

        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)

        return y, att_weights_


def weights_init(m: nn.Module) -> None:
    """Upstream ``main.py`` initializer (only used when a fresh model is built)."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


def load_vasnet_model(
    state_dict_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> VASNet:
    """Instantiate VASNet and load an upstream ``*.tar.pth`` checkpoint."""
    model = VASNet()
    state_dict = torch.load(Path(state_dict_path), map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError("VASNet checkpoint did not yield a state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def vasnet_frame_scores(model: VASNet, features: torch.Tensor, *, device: str | torch.device = "cpu") -> Any:
    """Return upstream frame importance scores for a ``[T, 1024]`` feature sequence."""
    if features.ndim != 2:
        raise ValueError("VASNet expects a 2-D [T, input_size] feature tensor")
    seq = features.to(device).float().unsqueeze(0)
    with torch.no_grad():
        scores, _ = model(seq, seq.shape[1])
    return scores.squeeze(0).detach().cpu().numpy()
