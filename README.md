# Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure Classification

This project explores hybrid quantum-classical representation learning for classifying dark matter substructure in strong gravitational lensing images. Strong lensing images contain subtle signatures that can help distinguish between competing dark matter models, including Cold Dark Matter, Axion/Fuzzy Dark Matter, and no-substructure cases.

The goal is to benchmark hybrid quantum-classical neural networks against classical deep learning baselines and study whether variational quantum circuits can improve learned representations for multi-class lensing classification.

This work is ongoing. The current repository focuses on pretrained classical backbones combined with VQC-based quantum representation layers. Future extensions will explore more physics-aware and equivariant network architectures.

## Project Description

Strong gravitational lensing is a powerful probe of dark matter and large-scale structure. The visual morphology of lensing images can encode small perturbations caused by dark matter substructure, but these signals are often subtle and high-dimensional.

This project implements hybrid quantum-classical models where pretrained classical CNN and vision backbones extract image features and trainable variational quantum circuits act as quantum representation layers. Classical backbone-only models are included only as baselines for comparison and do not use a VQC.

## Models Evaluated

- Hybrid ResNet18 + VQC
- Hybrid ResNet34 + VQC
- Hybrid ConvNeXt-Tiny + VQC
- Hybrid ViT-Small Patch16-224 + VQC
- Classical ResNet18, ResNet34, and ConvNeXt-Tiny baselines without VQC

Training notebooks and checkpoints are included for the main experiments.

## Results Summary

All reported models were evaluated on a held-out test set of 750 samples. The hybrid quantum-classical models are compared against classical backbone-only baselines trained without a variational quantum circuit.

| Backbone | Hybrid VQC Test Accuracy | Classical Backbone Test Accuracy | Improvement |
| --- | ---: | ---: | ---: |
| ResNet18 | 96.80% | 94.40% | +2.40% |
| ResNet34 | 96.00% | 95.60% | +0.40% |
| ConvNeXt-Tiny | 96.67% | 96.00% | +0.67% |
| ViT-Small Patch16-224 | 92.93% | - | - |

## Best Validation Results

| Model | Best Validation Accuracy | Test Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| ResNet18 + VQC | 96.25% | 96.80% | 96.76% |
| ResNet34 + VQC | 95.84% | 96.00% | 95.98% |
| ConvNeXt-Tiny + VQC | 96.22% | 96.67% | 96.60% |
| ViT-Small Patch16-224 + VQC | 92.27% | 92.93% | 92.89% |

## Key Findings

The strongest result was obtained by the ResNet18 + VQC hybrid model, which achieved 96.80% test accuracy and improved over the classical ResNet18 baseline by 2.40 percentage points.

ConvNeXt-Tiny also performed strongly, achieving 96.67% test accuracy with the hybrid quantum-classical setup. This was slightly higher than the pure classical ConvNeXt-Tiny baseline.

Overall, the experiments suggest that hybrid quantum-classical representation layers can provide competitive performance for dark matter substructure classification, especially when paired with strong classical image backbones.

## Extension Direction

This repository represents the pretrained-backbone + VQC phase of the project. The next phase will extend this direction toward equivariant networks and more geometry-aware architectures for lensing data.

An example of the broader Quantum ML test and extension plan is available here: [Specific Test III Quantum ML](https://github.com/sushmanthreddy/Task_2026/tree/main/Specific_Test_III_Quantum_ML).

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

notebooks/
  susmered_convnext_tiny.ipynb
  susmered_resnet18.ipynb
  susmered_resnet34.ipynb
  susmered_vit_small_patch16_224.ipynb
```

## Technologies Used

- Python
- PyTorch
- PennyLane or Qiskit-style quantum circuit simulation workflows
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

Ongoing work includes evaluating additional pretrained model families, extending the approach toward equivariant networks, testing quantum kernel methods, studying noise robustness under realistic NISQ conditions, expanding hardware-aware simulations, and comparing trained circuits across different qubit counts and circuit depths.
