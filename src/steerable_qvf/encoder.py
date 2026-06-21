"""Steerable C8/D8 CNN encoder + QVF neural amplitude/angle encoding.

Building blocks mirror GSoC-23/models/C8SteerableCNN.py (e2cnn R2Conv /
InnerBatchNorm / ReLU / PointwiseAvgPoolAntialiased / GroupPooling),
generalized to N rotations (+ optional reflections) and ending in one of
three encoding heads (amplitude / angle / reupload).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn as e2nn


class SteerableAmplitudeEncoder(nn.Module):
    """Rotation-equivariant CNN -> invariant vector -> quantum input.

    Depending on ``encoding``, the head emits either:
      - "amplitude": a 2**n_qubits statevector (softmax -> sqrt, ||a||_2 = 1),
      - "angle":     n_qubits tanh-bounded rotation angles,
      - "reupload":  reupload_layers * n_qubits tanh-bounded angles, shaped (B, L, n_qubits).
    """

    def __init__(self, img_size=64, in_channels=1, N=8, reflections=False,
                 n_qubits=8, base_width=8, temperature=1.0, dropout=0.3,
                 encoding="amplitude", reupload_layers=2):
        super().__init__()
        self.encoding = encoding
        self.n_qubits = n_qubits
        self.reupload_layers = reupload_layers
        self.temperature = temperature

        if encoding == "amplitude":
            out_dim = 2 ** n_qubits
        elif encoding == "angle":
            out_dim = n_qubits
        elif encoding == "reupload":
            out_dim = reupload_layers * n_qubits
        else:
            raise ValueError(f"Unknown encoding {encoding!r}")
        self.out_dim = out_dim

        # Symmetry group: C_N (rotations) or D_N (rotations + mirrors).
        self.r2_act = gspaces.FlipRot2dOnR2(N=N) if reflections else gspaces.Rot2dOnR2(N=N)

        in_type = e2nn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        self.input_type = in_type

        # block1: lift trivial input -> regular-rep feature fields
        out_type = e2nn.FieldType(self.r2_act, base_width * [self.r2_act.regular_repr])
        self.block1 = e2nn.SequentialModule(
            e2nn.MaskModule(in_type, img_size, margin=1),
            e2nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True),
        )

        in_type = out_type
        out_type = e2nn.FieldType(self.r2_act, (2 * base_width) * [self.r2_act.regular_repr])
        self.block2 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True),
        )
        self.pool1 = e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)

        in_type = out_type
        out_type = e2nn.FieldType(self.r2_act, (4 * base_width) * [self.r2_act.regular_repr])
        self.block3 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True),
        )
        self.pool2 = e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)

        # GroupPooling collapses each regular field to a single invariant scalar.
        self.gpool = e2nn.GroupPooling(out_type)
        inv_channels = self.gpool.out_type.size
        self.inv_channels = inv_channels

        # QVF amplitude head: out_dim logits, then softmax -> sqrt = valid statevector.
        # LayerNorm (not BatchNorm1d) -> identical behavior in train/eval.
        self.to_logits = nn.Sequential(
            nn.Linear(inv_channels, 128),
            nn.LayerNorm(128),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.pool2(x)
        x = self.gpool(x).tensor
        inv = F.adaptive_avg_pool2d(x, 1).flatten(1)
        # Force fp32: under AMP the head may be fp16, but the softmax norm / angle
        # bounds feeding the quantum bottleneck must be exact.
        raw = self.to_logits(inv).float()

        if self.encoding == "amplitude":
            probs = F.softmax(raw / self.temperature, dim=-1)
            q_in = torch.sqrt(probs.clamp_min(1e-12))
        else:
            q_in = torch.tanh(raw) * (np.pi / 2.0)
            if self.encoding == "reupload":
                q_in = q_in.view(-1, self.reupload_layers, self.n_qubits)
        return q_in, inv
