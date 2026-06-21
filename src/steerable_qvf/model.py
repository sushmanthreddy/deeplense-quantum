"""Full model: steerable encoder -> (amplitude/angle) -> equivariant QCNN -> head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config
from .encoder import SteerableAmplitudeEncoder
from .quantum import EquivQCNN_TQ


class SteerableQVFQuantumModel(nn.Module):
    def __init__(self, img_size=64, in_channels=1, n_qubits=8, num_classes=3,
                 group_n=8, reflections=False, base_width=8, temperature=1.0, dropout=0.3,
                 encoding="amplitude", reupload_layers=2,
                 readout_paulis=("Z", "X", "Y"), readout_zz=True,
                 hybrid_residual=True, residual_dim=16):
        super().__init__()
        self.encoder = SteerableAmplitudeEncoder(
            img_size=img_size, in_channels=in_channels, N=group_n, reflections=reflections,
            n_qubits=n_qubits, base_width=base_width, temperature=temperature, dropout=dropout,
            encoding=encoding, reupload_layers=reupload_layers,
        )
        zz_edges = EquivQCNN_TQ.CONV1_EDGES if readout_zz else None
        self.qcnn = EquivQCNN_TQ(
            n_qubits=n_qubits, encoding=encoding, reupload_layers=reupload_layers,
            paulis=readout_paulis, zz_edges=zz_edges,
        )

        # Optional hybrid residual: project the invariant CNN features and concat with the
        # quantum readout so the classifier is not starved through the quantum bottleneck.
        self.hybrid_residual = hybrid_residual
        head_in = self.qcnn.readout_dim
        if hybrid_residual:
            self.residual_proj = nn.Sequential(
                nn.Linear(self.encoder.inv_channels, residual_dim),
                nn.ELU(inplace=True),
            )
            head_in += residual_dim

        # Classifier head (shared across all encodings: angle / reupload / amplitude).
        # Mirrors equiqnn's post_net: Linear -> ReLU -> Dropout(0.2) -> Linear.
        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] > 1 and self.encoder.input_type.size == 1:
            x = x.mean(dim=1, keepdim=True)
        q_in, inv = self.encoder(x)
        z = self.qcnn(q_in)
        if self.hybrid_residual:
            z = torch.cat([z, self.residual_proj(inv)], dim=-1)
        return self.head(z)


def build_model(cfg: Config, num_classes: int, device: torch.device) -> SteerableQVFQuantumModel:
    """Construct the model from a Config and move it to ``device``."""
    model = SteerableQVFQuantumModel(
        img_size=cfg.load_img_size, in_channels=cfg.in_channels, n_qubits=cfg.n_qubits,
        num_classes=num_classes, group_n=cfg.group_n, reflections=cfg.use_reflections,
        base_width=cfg.base_width, temperature=cfg.softmax_temperature, dropout=cfg.encoder_dropout,
        encoding=cfg.encoding, reupload_layers=cfg.reupload_layers,
        readout_paulis=cfg.readout_paulis, readout_zz=cfg.readout_zz,
        hybrid_residual=cfg.hybrid_residual, residual_dim=cfg.residual_dim,
    ).to(device)
    return model


def parameter_summary(model: SteerableQVFQuantumModel) -> dict:
    """Return a dict of trainable parameter counts per component."""
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_quantum = sum(p.numel() for p in model.qcnn.parameters() if p.requires_grad)
    n_encoder = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    n_residual = (
        sum(p.numel() for p in model.residual_proj.parameters())
        if model.hybrid_residual else 0
    )
    return {
        "total": n_total, "encoder": n_encoder, "quantum": n_quantum,
        "head": n_head, "residual": n_residual,
    }
