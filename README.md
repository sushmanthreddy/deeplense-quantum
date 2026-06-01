# Equivariant Quantum-Classical Networks for Dark Matter Substructure Classification

This project studies a **symmetry-aware hybrid quantum-classical network** for classifying dark matter substructure in strong gravitational lensing images. Strong lensing images contain subtle signatures that help distinguish between competing dark matter models, including Cold Dark Matter, Axion/Fuzzy Dark Matter, and no-substructure cases.

Rather than relying on large pretrained backbones and data augmentation, the model encodes the rotational and reflectional symmetries of lensing physics **directly into the architecture** — both in the classical feature extractor and in the quantum circuit. The result is a compact, physics-aware model whose predictions are invariant under 90° rotations and reflections by construction.

This work is ongoing and is the active research focus of the project.

## Project Description

Strong gravitational lensing is a powerful probe of dark matter and large-scale structure. The visual morphology of lensing images can encode small perturbations caused by dark matter substructure, but these signals are often subtle and high-dimensional.

Gravitational lensing physics is invariant under the dihedral group `D4 = p4m point group` (rotations by 90° and reflections). Standard CNNs and pretrained ImageNet backbones only learn this property from data augmentation, which burns model capacity on a symmetry that can instead be encoded structurally. This project replaces:

- the classical backbone with an `e2cnn` **D₄ group-equivariant CNN**, and
- the generic variational quantum circuit (VQC) with a quantum circuit whose gates and parameter-sharing pattern make every layer commute with every group element of D₄.

A trainable quantum convolutional network then acts as a symmetry-preserving quantum representation layer on top of the equivariant classical features.

## Model

**D4-Equivariant CNN (`e2cnn FlipRot2dOnR2`) + Strict p4m QCNN**

A fully p4m-equivariant hybrid model:

- **Classical backbone:** D4 steerable CNN (`FlipRot2dOnR2(N=4)`) + an equivariant 1×1 convolution + spatial average pooling down to 8 channels (no bridge MLP).
- **Quantum circuit:** Strict p4m QCNN on **8 qubits** (`|D4| = 8`, one qubit per group element) — 6 layers of SWAP-symmetric U₂ blocks with tied RX parameters, edges grouped by D₄-orbits with parameter sharing per orbit, **no symmetry-breaking pooling**, and a D₄-orbit polynomial-invariant readout.
- **Equivariance:** strictly p4m by construction at every component; outputs are invariant under D₄.
- **Parameters:** 24 trainable quantum parameters (6 layers × 4), **172,379 total trainable parameters**.

## Results

Evaluated on a held-out test set of 750 samples.

| Metric | Value |
| --- | ---: |
| Best Validation Accuracy | 95.47% |
| Test Accuracy | 95.60% |
| Macro F1 | 0.9555 |
| Test ROC AUC | 0.9947 |
| Total trainable parameters | 172,379 |
| Trainable quantum parameters | 24 |

## Key Findings

- The Strict p4m model reaches **95.60% test accuracy** with only 172,379 total parameters (24 trainable quantum parameters) and provable end-to-end p4m equivariance — encoding the right inductive bias matters more than model size or ImageNet pretraining.
- Because equivariance is built in rather than learned from augmentation, the model is a strong starting point for studies that need mathematical symmetry guarantees, e.g. out-of-distribution rotated/reflected test sets and few-shot transfer to new lensing datasets.
- The notebook ships with extensive **QuTiP visualizations** (Bloch spheres with confidence colorbar, layer-wise Bloch animations following the QuTiP `bloch-sphere-animation` tutorial, Hinton diagrams of the reduced density matrix per class, Qubism plots of the 8-qubit pure state, computational-basis probability distributions, per-layer D₄-orbit graphs, and full circuit diagrams), making the model highly interpretable.

## Notebooks

- [`notebooks/equivariant/strict_p4m_qcnn.ipynb`](notebooks/equivariant/strict_p4m_qcnn.ipynb) — End-to-end training and evaluation of the D4-equivariant CNN + Strict p4m QCNN, including an explicit empirical equivariance test of every D₄ generator.
- [`notebooks/amplitude_testing/datavisualization_amplitude.ipynb`](notebooks/amplitude_testing/datavisualization_amplitude.ipynb) — Data visualization and amplitude-encoding exploration for the lensing dataset.

## Repository Structure

```text
checkpoints/
  equivariant/
    best_strict_p4m_qcnn.pth

notebooks/
  amplitude_testing/
    datavisualization_amplitude.ipynb
  equivariant/
    strict_p4m_qcnn.ipynb        # D4 ECNN + Strict p4m QCNN
```

## Technologies Used

- Python
- PyTorch
- **TorchQuantum** — GPU-native PyTorch-autograd quantum simulator (batched 8-qubit circuits, end-to-end gradient flow with `loss.backward()`)
- **e2cnn** — group-equivariant steerable CNNs (`FlipRot2dOnR2` for D₄ / p4m)
- **QuTiP** (with optional `qutip-qip`) — Bloch sphere visualizations, Hinton diagrams, Qubism plots, and circuit rendering
- Scikit-learn metrics
- Jupyter notebooks
- Git LFS for model checkpoints

## Extension Direction

The broader Quantum ML test and extension plan that originated this equivariant notebook lives here: [Specific Test III Quantum ML](https://github.com/sushmanthreddy/Task_2026/tree/main/Specific_Test_III_Quantum_ML).

The detailed project plan and upcoming implementation roadmap are tracked in this document: [GSoC 2026 project plan](https://docs.google.com/document/d/1weWicZdYN34_GT0637SVESl5sWR9RA6lcXsXWxuc5GA/edit?usp=sharing).

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

- **C8 / continuous-rotation backbones + QCNN** — extend the D₄ ECNN front-end to `Rot2dOnR2(N=8)`, harmonic networks (continuous SO(2) equivariance), and equivariant wide ResNets.
- **Equivariant Vision Transformers + QCNN** — combine attention with rotation equivariance feeding into the p4m QCNN.
- **Quantum kernel methods** — replace the variational QCNN with a quantum kernel estimator on the equivariant features.
- **Noise robustness under realistic NISQ conditions** — add depolarizing / amplitude-damping / phase-damping noise channels to the QCNN and evaluate degradation. TorchQuantum supports all of these natively.
- **Hardware-aware simulations** — compile the strict p4m circuit to IBM / Google / IonQ native gate sets and re-run.
- **Qubit-count and depth ablations** — re-train at 4, 8, 12, 16 qubits and 2, 4, 6, 8 conv layers to find the parameter / accuracy frontier of the equivariant QCNN.
- **Cross-dataset evaluation** — benchmark on DeepLense Model I (150×150 Gaussian PSF), Model II (64×64 Euclid-like), Model III (64×64 HST-like), and Model IV (multi-channel real galaxies).
