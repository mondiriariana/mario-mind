"""Hand-written LoRA adapters for Glimmer.

``add_lora(root, r)`` wraps every ``nn.Linear`` under ``root`` in a ``LoRA`` module: the base layer plus a
low-rank update B @ A (B starts at zero, so a fresh adapter leaves the model unchanged). ``find_layers`` returns
the language model's decoder layers.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        return y + ((self.drop(x).to(self.A.dtype) @ self.A.t() @ self.B.t()) * self.scale).to(y.dtype)


def add_lora(root, r):
    """LoRA (rank r, alpha 2r) on every nn.Linear under root; returns the number of wrapped modules."""
    n = 0
    for name, m in list(root.named_modules()):
        for cname, c in list(m.named_children()):
            if isinstance(c, nn.Linear) and not isinstance(m, LoRA):
                setattr(m, cname, LoRA(c, r, 2 * r))
                n += 1
    return n


def find_layers(model, n_layers):
    """The decoder ModuleList (len == n_layers) of the language model."""
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) == n_layers and "vision" not in name:
            return name, m
    raise RuntimeError("decoder layers not found")
