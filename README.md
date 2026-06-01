# Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure Classification

This project explores hybrid quantum-classical representation learning for classifying dark matter substructure in strong gravitational lensing images. Strong lensing images contain subtle signatures that can help distinguish between competing dark matter models, including Cold Dark Matter, Axion/Fuzzy Dark Matter, and no-substructure cases.

The goal is to benchmark hybrid quantum-classical neural networks against classical deep learning baselines and study whether variational quantum circuits can improve learned representations for multi-class lensing classification.

This work is ongoing. The current repository covers two stages of experiments:

- **Stage 1–2 (Pretrained backbones + VQC):** Hybrid models that combine pretrained classical CNN backbones (ResNet18, ResNet34, ConvNeXt-Tiny) with variational quantum circuits used as quantum representation layers.
- **Stage 3 (Equivariant network + quantum):** Physics-aware extension that replaces ImageNet-pretrained backbones with an `e2cnn` D₄ rotation/reflection-equivariant CNN and replaces the generic VQC with a quantum circuit that is itself equivariant under the dihedral group D₄ (the p4m point group). This is now the active research focus.

## Project Description

Strong gravitational lensing is a powerful probe of dark matter and large-scale structure. The visual morphology of lensing images can encode small perturbations caused by dark matter substructure, but these signals are often subtle and high-dimensional.

This project implements hybrid quantum-classical models where classical CNN / equivariant CNN backbones extract image features and trainable variational quantum circuits (or, in Stage 3, a p4m-equivariant quantum convolutional network) act as quantum representation layers. Classical backbone-only models (Stage 1–2) are included only as baselines for comparison and do not use a VQC.

## Models Evaluated

**Stage 1–2 — Pretrained backbone + VQC:**

- Hybrid ResNet18 + VQC
- Hybrid ResNet34 + VQC
- Hybrid ConvNeXt-Tiny + VQC
- Classical ResNet18, ResNet34, and ConvNeXt-Tiny baselines without VQC

**Stage 3 — Equivariant backbone + equivariant quantum circuit:**

- Hybrid D4-Equivariant CNN (`e2cnn FlipRot2dOnR2`) + **Strict p4m QCNN** — fully p4m-equivariant variant: SWAP-symmetric U₂ blocks with tied RX parameters, edges grouped by D₄-orbits with parameter sharing per orbit, no symmetry-breaking pooling. Output is invariant under D₄ via a 4-dim polynomial-invariant readout. 24 trainable quantum parameters.

Training notebooks and checkpoints are included for the main experiments.

## Results Summary

All reported models were evaluated on a held-out test set of 750 samples. The hybrid quantum-classical models are compared against classical backbone-only baselines trained without a variational quantum circuit.

| Backbone | Hybrid VQC Test Accuracy | Classical Backbone Test Accuracy | Improvement |
| --- | ---: | ---: | ---: |
| ResNet18 | 96.80% | 94.40% | +2.40% |
| ResNet34 | 96.00% | 95.60% | +0.40% |
| ConvNeXt-Tiny | 96.67% | 96.00% | +0.67% |

## Best Validation Results

| Model | Best Validation Accuracy | Test Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| ResNet18 + VQC | 96.25% | 96.80% | 96.76% |
| ResNet34 + VQC | 95.84% | 96.00% | 95.98% |
| ConvNeXt-Tiny + VQC | 96.22% | 96.67% | 96.60% |

## Key Findings

The strongest result was obtained by the ResNet18 + VQC hybrid model, which achieved 96.80% test accuracy and improved over the classical ResNet18 baseline by 2.40 percentage points.

ConvNeXt-Tiny also performed strongly, achieving 96.67% test accuracy with the hybrid quantum-classical setup. This was slightly higher than the pure classical ConvNeXt-Tiny baseline.

Overall, the experiments suggest that hybrid quantum-classical representation layers can provide competitive performance for dark matter substructure classification, especially when paired with strong classical image backbones.

## Stage 3 — Equivariant Networks

This stage extends the dressed-quantum pipeline by encoding the rotational and reflectional symmetries of strong-lensing images directly into both the classical backbone and the quantum circuit.

### Motivation

Gravitational lensing physics is invariant under the dihedral group `D4 = p4m point group` (rotations by 90° and reflections). Standard CNNs (and pretrained ImageNet backbones) only learn this from data augmentation, which burns capacity on a property that can be encoded structurally. Stage 3 replaces:

- the pretrained backbone with an `e2cnn` D4 group-equivariant CNN, and
- the generic VQC with a quantum circuit whose gates and parameter-sharing pattern make every layer commute with every group element of D4.

The result: the network's outputs are exactly invariant under 90° rotations and reflections by construction, not by augmentation.

### Equivariant Models

| Notebook | Classical backbone | Quantum circuit | Quantum params | Equivariance |
| --- | --- | --- | ---: | --- |
| `strict_p4m_qcnn.ipynb` | D4 Steerable CNN (`FlipRot2dOnR2(N=4)`) + equivariant 1×1 conv + spatial AvgPool to 8 channels (no bridge MLP) | Strict p4m QCNN: 6 layers of SWAP-symmetric U₂ blocks with tied RX, edges grouped by D₄-orbits with parameter sharing per orbit, **no pooling**, D₄-orbit polynomial-invariant readout | 24 | Strictly p4m by construction at every component; outputs invariant under D₄ |

### Stage 3 Results

| Model | Best Validation Accuracy | Test Accuracy | Macro F1 | Test ROC AUC | Total params |
| --- | ---: | ---: | ---: | ---: | ---: |
| D4-equivariant CNN + Strict p4m QCNN (`strict_p4m_qcnn.ipynb`) | 95.47% | 95.60% | 0.9555 | 0.9947 | 172,379 |

### Stage 3 Findings

- The Strict p4m model achieves **95.60% test accuracy** with only 172,379 total parameters (24 trainable quantum parameters) and provable end-to-end p4m equivariance. It is the most parameter-efficient model in the repository and is the right starting point for studies that need mathematical symmetry guarantees (e.g. out-of-distribution rotated/reflected test sets, few-shot transfer to new lensing datasets).
- The notebook ships with extensive **QuTiP visualizations** (Bloch spheres with confidence colorbar, layer-wise Bloch animations following the QuTiP `bloch-sphere-animation` tutorial, Hinton diagrams of reduced ρ per class, Qubism plots of the 8-qubit pure state, computational-basis probability distributions, per-layer D₄-orbit graphs, and full circuit diagrams) — making this stage the most interpretable in the repository.

### Stage 3 Notebooks

- [`notebooks/equivariant/strict_p4m_qcnn.ipynb`](notebooks/equivariant/strict_p4m_qcnn.ipynb) — Strictly D4 / p4m-equivariant hybrid model with the smallest parameter count (172k total / 24 quantum) and an explicit empirical equivariance test of every D₄ generator.

## Extension Direction

This repository previously represented only the pretrained-backbone + VQC phase. With Stage 3 added (see the "Stage 3 — Equivariant Networks" section above), it now covers both pretrained transfer-learning hybrids and a physics-aware equivariant hybrid. The current research focus is on extending the equivariant direction further (continuous-rotation backbones, attention, quantum kernels, noise robustness) — see "Future Work" below.

The broader Quantum ML test and extension plan that originated these equivariant notebooks lives here: [Specific Test III Quantum ML](https://github.com/sushmanthreddy/Task_2026/tree/main/Specific_Test_III_Quantum_ML).

The detailed project plan and upcoming implementation roadmap are tracked in this document: [GSoC 2026 project plan](https://docs.google.com/document/d/1weWicZdYN34_GT0637SVESl5sWR9RA6lcXsXWxuc5GA/edit?usp=sharing).

## Repository Structure

```text
checkpoints/
  classical/
    convnext_tiny.pth
    resnet18.pth
    resnet34.pth
  stage1/
    convnext_tiny.pth
    resnet18.pth
    resnet34.pth
  stage2/
    convnext_tiny.pth
    resnet18.pth
    resnet34.pth
  equivariant/
    best_strict_p4m_qcnn.pth

notebooks/
  susmered_convnext_tiny.ipynb
  susmered_resnet18.ipynb
  susmered_resnet34.ipynb
  amplitude_testing/
    datavisualization_amplitude.ipynb
  equivariant/
    strict_p4m_qcnn.ipynb        # Stage 3: D4 ECNN + Strict p4m QCNN
```

## Technologies Used

- Python
- PyTorch
- PennyLane / Qiskit-style quantum circuit simulation workflows (Stage 1–2)
- **TorchQuantum** — GPU-native PyTorch-autograd quantum simulator used in Stage 3 (batched 8-qubit circuits, end-to-end gradient flow with `loss.backward()`)
- **e2cnn** — group-equivariant steerable CNNs (`FlipRot2dOnR2` for D₄ / p4m)
- **QuTiP** (with optional `qutip-qip`) — Bloch sphere visualizations, Hinton diagrams, Qubism plots, and circuit rendering for the Stage 3 notebook
- Torchvision and pretrained image backbones
- Scikit-learn metrics
- Jupyter notebooks
- Git LFS for model checkpoints

## Mentors

- Michael Toomey, Massachusetts Institute of Technology
- Sergei Gleyzer, University of Alabama
- Pranath Reddy, Independent Researcher
- Rajat Shinde, University of Alabama in Huntsville

## Project Context

Project: Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure Classification

Duration: 175/350 hours

Difficulty: Advanced

## Future Work

The Stage 3 notebook added in this iteration covers the **equivariant networks** direction that was previously listed under Future Work. The remaining open directions are:

- **C8 / continuous-rotation backbones + QCNN** — extend the D₄ ECNN front-end to `Rot2dOnR2(N=8)`, harmonic networks (continuous SO(2) equivariance), and equivariant wide ResNets.
- **Equivariant Vision Transformers + QCNN** — combine attention with rotation equivariance feeding into the p4m QCNN.
- **Quantum kernel methods** — replace the variational QCNN with a quantum kernel estimator on the equivariant features.
- **Noise robustness under realistic NISQ conditions** — add depolarizing / amplitude-damping / phase-damping noise channels to the QCNN and evaluate degradation. TorchQuantum supports all of these natively.
- **Hardware-aware simulations** — compile the strict p4m circuit to IBM / Google / IonQ native gate sets and re-run.
- **Qubit-count and depth ablations** — re-train at 4, 8, 12, 16 qubits and 2, 4, 6, 8 conv layers to find the parameter / accuracy frontier of the equivariant QCNN.
- **Cross-dataset evaluation** — benchmark all stages on DeepLense Model I (150×150 Gaussian PSF), Model II (64×64 Euclid-like), Model III (64×64 HST-like), and Model IV (multi-channel real galaxies).
