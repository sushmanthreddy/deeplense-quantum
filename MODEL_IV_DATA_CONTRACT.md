# Model IV data-generation and proof-of-signal contract

This document records why the supplied Model IV release is not a valid
counterpart to Model II and defines the minimum evidence required before a
replacement Model IV release is used for expensive training. It is a data
contract, not a claim that a particular classifier can recover information
that is absent from its inputs.

## Current diagnosis: what is demonstrated

The following statements are supported directly by the audited files,
generator code, or held-out development-validation experiments:

1. **Models II and IV are not the same generative distribution.** They share a
   64-by-64 Euclid-like image setting, but Model II uses an analytic Sérsic
   source and legacy point-mass/vortex prescriptions. Model IV was designed
   around real DECaLS/Galaxy10 sources and pyHalo dark-matter populations.
2. **Released Model II is an artificially easy control.** Its `simple_sim_2`
   path computes Poisson noise and then saves `image_model + image_model`.
   Thus it doubles the noiseless model instead of adding the computed noise.
   This bug is visible in the released source and independently covered by
   upstream PR #9.
3. **The supplied Model IV archive is not reproducible from the committed
   Model IV scripts.** The archive and code disagree on array shape/container,
   intensity scale, axion-mass convention, sample counts, split sizes, and the
   `no_sub` output path. The generator commit, dependency lock, source IDs,
   latent parameters, and archive digest that actually produced the supplied
   files are unavailable. This is true across all four public Model-IV history
   revisions: every one fixes the ULDM log mass at -21, saves normalized
   three-channel arrays without a scalar payload, and writes the no-sub class
   under `no/`; the mounted archive instead has variable mass metadata,
   single-channel arrays, raw-scale values, and `no_sub/`.
4. **The public Model-IV code does not implement a controlled three-way
   intervention.** The no-sub script derives its macro lens from sampled
   velocity dispersion, omits external shear, and constructs a separate
   cosmology. The CDM and ULDM scripts instead use fixed Einstein radius 1.5,
   shear 0.05, and pyHalo cosmology. Source identity, macro nuisance, and noise
   are sampled independently by class, so class residuals cannot be isolated
   counterfactually.
5. **A later public attempt at paired real-source generation also has a dead
   ULDM path.** Its `simulate_uldm` function creates a pyHalo `ULDM` realization
   but never calls `lensing_quantities()` or appends any ULDM lens terms; its
   `lens_model_list_full` remains macro-only. The CDM function does append halo
   terms. This 2025 notebook is not claimed to have generated the mounted
   archive, but it demonstrates exactly why field- and pixel-reachability
   assertions are mandatory.
6. **Simple corruption does not explain the failure.** The audited Model IV
   images are finite, nonconstant, and unique, with no train/validation or
   cross-class content duplicates. Audited train and validation summaries do
   not show a material domain shift.
7. **No robust label signal has been demonstrated in the supplied pixels.**
   Historical ResNet, equivariant, and contrastive results are near chance.
   Current leakage-safe raw-pixel, normalized-pixel, residual, radial,
   frequency, and morphology controls are also near chance on the supplied
   validation split. The saved axion scalar has no out-of-sample relationship
   to the tested morphology summaries.

These results do **not** prove that every possible raw-pixel statistic is
independent of the label, nor do they identify a particular missing simulator
call. They do prove that Model II accuracy is not evidence that this Model IV
archive should be equally learnable.

## Best-supported inference, not a proven mechanism

The supplied release has no *demonstrated* class-conditioned dark-matter
imprint. The two live explanations are:

- the unpublished generation path failed to propagate the requested
  dark-matter class or axion mass into the rendered pixels, or mixed images and
  labels; and/or
- a real-source imprint exists before observation but is suppressed below the
  source-morphology, PSF, and detector-noise variation.

Architecture or optimizer tuning cannot distinguish these causes. A paired
counterfactual forward simulation can.

## Corrected paired-counterfactual generation contract

Each latent draw must produce an aligned triplet: `no_sub`, `cdm`, and `axion`.
The class intervention is the only allowed difference within that triplet.

For every `pair_id`, freeze and record:

- source catalog release, source ID, crop, pose, flux scaling, and redshift;
- macro-lens family and every macro-lens parameter, lens redshift, and
  coordinate/unit convention;
- instrument, band, pixel scale, exposure, PSF, detector model, and all
  observation/noise settings;
- nuisance, dark-matter, and observation random seeds; and
- the complete CDM or ULDM parameters, including `log10_m_uldm` where
  applicable.

The `no_sub` member removes only the substructure perturbation. CDM and ULDM
members use documented deterministic dark-matter seeds derived from the pair,
while all non-dark-matter latents remain identical. The same renderer function
and observation path must handle all three classes; class-specific save or
normalization code is forbidden.

The release must pin the generator repository and commit, record whether the
tree was clean, lock every simulation dependency, and identify the container
image by digest. It must also publish the immutable source-catalog identifier,
generation configuration, archive SHA-256, per-file hashes, and a relative-path
manifest containing the recorded latents. Machine-specific paths, credentials,
and storage locations are not provenance and must not appear.

### Serialization contract

- Save every lens image as a numeric `float32` NPY array of shape `(64, 64)`.
  If a future release intentionally uses multiple bands, use the same explicit
  channel-first shape for every class and update the model contract first.
- Never use object arrays. Store axion mass and other metadata in the release
  manifest, not inside one class's NPY payload.
- Use the same physical units, clipping, quantization, and dynamic-range policy
  for every class. Preserve raw flux/count images and document any derived
  normalized view; never apply a class-dependent or undocumented per-image
  normalization.
- Use the same opaque `pair_id` filename in each class directory. File headers,
  shapes, dtypes, compression, names, and directory depth must not provide a
  class shortcut beyond the label supplied by the dataset index.
- A clean rerender from the manifest must reproduce each stored image within a
  declared numerical tolerance and reproduce its hash when the environment is
  bitwise deterministic.

### Split contract

Create group splits before rendering. Keep all three members of a pair in the
same split, and keep each source identity in exactly one of train, validation,
or sealed test. Do not reuse a source crop, augmented view, macro-lens latent,
or observation seed across splits. Because every latent draw is a triplet,
class counts must be exactly balanced within each split. Publish the grouping
manifest and assert zero overlap by `pair_id`, source ID, and model-visible
content hash. The official test remains sealed until a final evaluation is
explicitly requested.

## Mandatory simulator assertions

Generation must fail rather than publish an archive when any assertion below
fails.

1. **Schema and integrity:** every output has the declared shape, dtype, units,
   finite range, and class-independent serialization; no output is constant or
   duplicated across pairs or splits.
2. **Determinism:** identical latents and seeds reproduce the same convergence,
   deflection, noiseless image, PSF-convolved image, and noisy image within the
   declared tolerance.
3. **Null-path identity:** with substructure explicitly disabled, calls through
   the three class paths return identical intermediate fields and images. This
   catches hidden class-dependent nuisance or rendering branches.
4. **Parameter reachability:** enabling CDM or ULDM changes the pre-PSF
   convergence and/or deflection map by more than numerical tolerance for a
   documented fraction of pairs. Disabling that perturbation must restore the
   null result. Assert that each non-null realization contributes its expected
   lens-model entries, halo kwargs, redshift entries, and numerical-deflection
   callback when the backend returns one; constructing a realization object is
   not evidence that it reached the renderer.
5. **Pixel reachability:** the field change must propagate to a nonzero
   noiseless pixel residual. Saved pixels must match the final simulator output,
   rather than an earlier unperturbed buffer.
6. **ULDM sweep:** with every nuisance fixed, sweep the intended mass range,
   including `log10_m_uldm` values from -22 through -18. A predeclared physical
   field/residual statistic must respond nontrivially and follow the expected
   trend wherever theory predicts monotonicity. Do not demand that every pixel
   itself be monotonic.
7. **PSF/noise budget:** for each pair, report the class residual
   `||I_class - I_no_sub||` and its spatial power spectrum before and after the
   PSF. Compare it with repeated-observation residuals made from independent
   noise seeds. For a release intended to be readily learnable, preregister an
   effect-to-noise threshold; a conservative default is that the lower 95%
   confidence bound on the median paired effect/noise ratio exceeds one.
8. **Metadata-leak null:** replacing all pixels with a constant must reduce any
   loader-level classifier to chance. Container type, dtype, shape, filename,
   manifest ordering, or missing metadata may not reveal the class.
9. **Source-disjoint signal gate:** freeze one low-capacity baseline and its
   preprocessing before inspecting validation results. On source-disjoint
   development validation, both macro one-vs-rest AUC and balanced accuracy
   must have a group-bootstrap lower 95% confidence bound above their chance
   values, 0.5 and 1/3. Confirm the result with a pair/source-grouped label
   permutation test and correct for any predeclared multiple baselines. A
   benchmark promising stronger learnability must preregister a higher margin;
   merely clearing chance does not promise Model-II-level accuracy.
10. **Mass-information gate:** if the axion mass is advertised as a learnable
    target, a frozen regressor must beat a constant predictor on
    source-disjoint validation and pass a grouped permutation test. Otherwise
    the mass is provenance metadata, not a supported prediction target.

Tiny-network memorization is useful only as an optimization sanity check. It
does not satisfy the source-disjoint signal gate.

## Causal ablation ladder

Run the same paired latent bank through this ladder and record where the signal
disappears. Do not redraw nuisance variables between rungs.

| Rung | Controlled render | What failure establishes |
|---|---|---|
| 0 | Repeated null render, identical seeds | Numerical nondeterminism or class-path leakage |
| 1 | CDM/ULDM versus no-sub convergence and deflection fields | Dark-matter intervention is not reaching the lens model |
| 2 | Analytic Sérsic source, no PSF, no detector noise | Intervention is not reaching ideal pixels |
| 3 | Real DECaLS source, no PSF, no detector noise | Source complexity suppresses the ideal-image effect |
| 4 | Real source plus PSF, still noiseless | The observing resolution removes the informative scales |
| 5 | Real source plus PSF and repeated detector noise | Signal lies below the observation-noise budget |
| 6 | Full nuisance distribution, still paired | Nuisance diversity overwhelms the surviving residual |
| 7 | Source-disjoint frozen baseline | Surviving signal does or does not generalize to new galaxies |

At each rung report paired residual norms, radial and spatial-frequency power,
and uncertainty by source-group bootstrap. Crossing analytic/real source with
no-PSF/PSF/noise conditions is the decisive factorial test: it separates a
dark-matter generator defect from a resolution/SNR problem.

## Release decision

A Model IV archive is training-ready only when its exact generator is
reproducible, every mandatory assertion passes, and the source-disjoint signal
gate succeeds without using the official test. Until then, the scientifically
correct result for the supplied release is chance-level development validation.
Searching many preprocessing choices for a favorable validation fluctuation or
using its class-specific object-array format would manufacture leakage, not
recover missing physics.

## Primary sources and historical context

- Audited DeepLenseSim commit:
  [`a0c0191`](https://github.com/mwt5345/DeepLenseSim/tree/a0c01910820a1c30593c56e728c7fdb3ac3701b4)
- Model II [analytic-source description](https://github.com/mwt5345/DeepLenseSim/blob/a0c01910820a1c30593c56e728c7fdb3ac3701b4/Model_II/README.md#L3-L11),
  [axion script](https://github.com/mwt5345/DeepLenseSim/blob/a0c01910820a1c30593c56e728c7fdb3ac3701b4/Model_II/sim_axion.py#L12-L21),
  and [`simple_sim_2` bug](https://github.com/mwt5345/DeepLenseSim/blob/a0c01910820a1c30593c56e728c7fdb3ac3701b4/deeplense/lens.py#L279-L313)
- Independent upstream [fix and regression test in PR #9](https://github.com/mwt5345/DeepLenseSim/pull/9)
- Model IV [real-source description](https://github.com/mwt5345/DeepLenseSim/blob/a0c01910820a1c30593c56e728c7fdb3ac3701b4/Model_IV/README.md#L3-L11),
  [source selection](https://github.com/mwt5345/DeepLenseSim/blob/a0c01910820a1c30593c56e728c7fdb3ac3701b4/Model_IV/sim_axion.py#L190-L216),
  [ULDM construction](https://github.com/mwt5345/DeepLenseSim/blob/a0c01910820a1c30593c56e728c7fdb3ac3701b4/Model_IV/sim_axion.py#L43-L84),
  [CDM construction](https://github.com/mwt5345/DeepLenseSim/blob/a0c01910820a1c30593c56e728c7fdb3ac3701b4/Model_IV/sim_cdm.py#L42-L70),
  and [save/normalization path](https://github.com/mwt5345/DeepLenseSim/blob/a0c01910820a1c30593c56e728c7fdb3ac3701b4/Model_IV/sim_axion.py#L173-L221)
- The later public [paired real-source notebook](https://github.com/ML4SCI/DeepLense/blob/ea6d954317ff4bcb4ca760ca428aca1f9a4794c8/DeepLense_Physics_Informed_Neural_Network_for_Dark_Matter_Morphology_Ashutosh_Ojha/Work%20on%20Real%20Lensing/genetate-paired.ipynb),
  whose ULDM branch leaves the full lens list macro-only
- Historical official [Model IV ResNet18 notebook](https://github.com/ML4SCI/DeepLense/blob/9f5ad845e02a1f73002ccd7b4fde9875aae2fc9e/Equivariant_Neural_Networks_for_DeepLense_GEO/notebooks/classification/resnet18/resnet18-model4.ipynb)
  and [contrastive-learning notebook](https://github.com/ML4SCI/DeepLense/blob/05b4316c41e346d51c3b73951775ba6cf7ef7fd0/Deeplens_Self_Supervised_Learning_Yashwardhan_Deshmukh/contrastive_learning/notebooks/model4_contrastive.ipynb)
- Firsthand GSoC report that multiple Model IV approaches remained near chance
  and that the simulations were under review:
  [Updating the DeepLense Pipeline, Part 2](https://medium.com/@saranga.boo/updating-the-deeplense-pipeline-part-2-gsoc-2023-with-ml4sci-299a48d0dd23)
