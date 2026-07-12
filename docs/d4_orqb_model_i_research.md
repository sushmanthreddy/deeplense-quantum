# D4 Orbit-Reuploading Quantum Bottleneck: Model-I Research Record

Status as of 2026-07-11: the development study, finite-shot simulation, and
prospectively locked v3 Model-I test evaluation are complete. Metrics are
labelled as development validation, official test, or historical context. Of
the three lock attempts, only v3 crossed the durable test-access boundary and
ran model inference on the 15,000-image official test set.

This document is an experiment record, not a paper claim. The frozen primary
three-quantum-model ensemble (`q3`) reached 98.5867% official-test accuracy
(Wilson 95% interval 98.3849--98.7635%) and 0.998907 macro AUC; its
parameter-matched classical ensemble (`c3`) reached 98.5800% and 0.998985.
Their paired accuracy difference was +0.0067 percentage point with a 95%
bootstrap interval of -0.1333 to +0.1467 point. Thus the compact hybrid is
competitive with its matched control, but there is no evidence of quantum
advantage. End-to-end optimization remains seed-sensitive, the three core
replicates are conditional on one fixed backbone, and neither these results nor
the intended submission venue provides any guarantee of workshop acceptance.

## Fixed data protocol and integrity

- Development root: `DEEPLENS_DATASETS/Model_I`
- Official test root: `DEEPLENS_DATASETS/Model_I_test`
- Classes, in canonical order: `axion`, `cdm`, `no_sub`
- Development set: 87,525 images
  (`28,897 / 29,772 / 28,856`)
- Official test set: 15,000 images, balanced at 5,000 per class
- Fixed stratified development split: seed 42, validation fraction 0.2
- Training set: 70,021 images (`23,118 / 23,818 / 23,085`)
- Validation set: 17,504 images (`5,779 / 5,954 / 5,771`)
- Model input: 96 x 96, one channel
- Model-visible preprocessing: replace nonfinite values, normalize each image
  by its maximum when that maximum exceeds one, antialiased bilinear resize,
  and persist the cache as float16
- The axion files contain an image and mass value; the loader deliberately
  discards the mass so that no label-correlated metadata reaches the model.

The cache manifest records relative path, class, label, and SHA-256 digest. A
complete scan of all 102,525 model-visible images found:

- 87,525 unique development images and 15,000 unique test images;
- no within-class or cross-class duplicate image;
- no development/test image-digest overlap; and
- no development/test filename overlap.

The split indices and manifest are persisted with each run. Training and model
selection used validation only and set `evaluate_test=false`. Some early data
reports contain test-cache metadata from integrity/cache preparation; that is
not model inference. The later v3 test predictions are separately sealed and
were created only after the development replay passed and the durable access
marker was written.

### Test-lock scope

The lock is prospective for this D4-ORQB study, not a claim that the repository's
official test set is globally pristine. Older notebooks in this repository,
including the prior P4M and steerable-QVF studies, already evaluated the same
official test data. Consequently, the D4-ORQB result is a single,
protocol-frozen evaluation for the new study, but it cannot be described as the
first human exposure to this repository's test set.

## Development-only data diagnostics

These probes were used to understand Model I and choose inexpensive inductive
biases. They are not official benchmark results. A bank of 94 D4-invariant
summary features was evaluated with linear ridge/LDA probes on development data:

| Feature family | Accuracy | Macro AUC |
|---|---:|---:|
| Radial intensity summaries | 76.02% | 0.8805 |
| Fourier/power-spectrum summaries | 70.79% | 0.8544 |
| Gradient summaries | 63.35% | 0.8013 |
| Combined 94-feature probe | 85.32% | 0.9379 |
| Location-only controls | 34.63% | not used |
| Sparse pixel-shortcut controls | 46.50% | not used |

The combined probe recalled `no_sub` at 98.87%; most residual difficulty was
the physically relevant `axion` versus `cdm` distinction. The largest
standardized class effects occurred in central log intensity (about 1.50 SD),
central gradient (about 1.20 SD), a radial annulus at roughly 5.7--11.3 pixels
(about 0.97 SD), and wavelengths of roughly 5--14 pixels (about 0.82 SD).

The same simple separability check at 96, 128, and native 150 pixels gave
80.97%, 81.23%, and 80.87%, respectively. Frequencies lost by resizing to 96
pixels carried only about 0.04--0.05 SD of class separation. This did not
support paying the much higher training cost of native resolution.

Centroid and gross-location controls were approximately class matched, so the
observed signal is not explained by a simple position label leak. The strong
central/radial signal can nevertheless be simulator-specific. Small simulated
translations of up to two pixels reduced the hardest aperture effect by more
than half, motivating a translation stress/fine-tuning experiment rather than
assuming that the central feature is robust physics.

## Architecture

The primary model is the tiny, base-physics D4 orbit-reuploading quantum
bottleneck (D4-ORQB):

1. A fixed, zero-parameter morphology bank maps the intensity image to eight
   channels: intensity, log intensity, Sobel magnitude, absolute Laplacian,
   difference of local averages, radial gradient, tangential gradient, and a
   stabilized mixed derivative of a squared log-intensity ratio. It is
   LensPINN-inspired preprocessing, not a PINN: no differential-equation
   residual is optimized.
2. The model constructs all eight exact D4 image transforms. A single shared
   MobileNet-style MBConv encoder processes them in one vectorized orbit batch;
   its weights therefore do not grow with the group order.
3. A shared linear map projects each 128-dimensional view embedding to four
   heads, two regular-representation fields per head, and eight D4 positions.
   The resulting tensor has shape `batch x 4 x 2 x 8`.
4. Each head is a reusable eight-qubit circuit. The two fields are densely
   encoded with RY/RZ gates, followed by tied local RX/RY/RZ gates and commuting
   XX/ZZ Pauli rotations over complete left-Cayley edge orbits. Two data
   reuploads are used.
5. Local Z/X and Cayley-edge ZZ/XX expectations are reduced to 12 invariant
   scalars per head, or a 48-dimensional bottleneck. A `48 -> 32 -> 3` head
   emits class logits.
6. The primary model has no classical context bypass. The image must influence
   the classifier through the quantum feature bottleneck.

The matched classical control preserves the same physics bank, encoder,
projection, invariant readout dimension, head, total parameter count, and
training schedule. It replaces the circuit's 88 trainable values with an
88-parameter elementwise Fourier mixer.

The radial variant adds two zero-parameter channels: a signed 5-versus-13-pixel
multiscale response and a soft central log-intensity envelope. Only the first
convolution grows, by 800 trainable weights.

## Symmetry guarantee and expressivity caveat

An input D4 transform permutes the eight orbit views by the right-regular
action. The image encoder and orbit projection are shared over those views.
Within the circuit, every one-qubit operation is tied over all eight qubits,
and each two-qubit angle is tied over a complete left-Cayley edge orbit. Pauli
rotations within each edge family commute, so the circuit is equivariant to the
induced permutation. Orbit-averaging the local and edge observables then makes
the returned 48 features, and hence the logits, D4-invariant.

This is a structural, exact-in-real-arithmetic argument. Numerical tests agree:
CPU unit tests show maximum differences around `6e-8`; the saved bfloat16 H200
audits had maximum full-model logit discrepancies of `3.71e-4` for the base
quantum run and `4.51e-4` for the radial quantum run.

The bottleneck must not be described as having *exactly* D4 symmetry. An
exhaustive check of all `8!` qubit permutations found 16 permutations that
preserve the colored undirected edge families `E_r`, `E_r2`, and `E_s`. In
addition to right-regular D4, the map

`alpha: (k, f) -> (-k, f)`

(conjugation by the reflection generator) preserves all three families.
Composing `alpha` with right-regular D4 gives 16 bottleneck automorphisms. A
random-state numerical check changed the invariant vector by only about
`7e-7`. Thus the bottleneck is at least D4-invariant and has an additional
bottleneck-only colored-graph automorphism that may restrict expressivity. It
does **not** establish a ninth generic image-space transformation: the learned
orbit encoder need not realize `alpha` as a transformation of input images.

## Exact parameter and logical-gate accounting

### Trainable parameters

| Variant | Physics | Encoder | Orbit projection | Core | Classifier/context | Total |
|---|---:|---:|---:|---:|---:|---:|
| Tiny D4-ORQB, base | 0 | 242,338 | 1,032 | 88 quantum | 1,763 | 245,221 |
| Tiny matched classical, base | 0 | 242,338 | 1,032 | 88 classical | 1,763 | 245,221 |
| Tiny context pretrainer | 0 | 242,338 | 1,032 | 88 classical | 29,347 | 272,805 |
| Tiny D4-ORQB, radial | 0 | 243,138 | 1,032 | 88 quantum | 1,763 | 246,021 |
| Small D4-ORQB, base | 0 | 667,716 | 1,544 | 88 quantum | 1,763 | 671,111 |
| Repaired/adapted LensPINN-small | included | included | n/a | n/a | included | 6,235,157 |

Only 88 parameters, 0.0359% of the base model, reside in the quantum circuit;
the parameter-efficiency claim is therefore about the complete compact hybrid,
not an assertion that the classical encoder is negligible. The base hybrid has
25.43 times fewer trainable parameters than the repaired LensPINN-small used in
the same-split comparison.

### Circuit gates

Each head and reupload has 11 tied parameters:

- four data scale/bias values for dense RY/RZ encoding;
- three tied local RX/RY/RZ angles; and
- four edge-family angles for `RZZ(E_r)`, `RZZ(E_s)`, `RXX(E_r)`, and
  `RXX(E_s)`.

Therefore `4 heads x 2 reuploads x 11 = 88` circuit parameters. The logical
gate count is:

| Scope | One-qubit rotations | Two-qubit Pauli rotations |
|---|---:|---:|
| One head, one reupload | 40 | 24 |
| One head, two reuploads | 80 | 48 |
| Four-head model, per image | 320 | 192 |

The 40 one-qubit rotations are 16 data RY/RZ gates plus 24 tied local
RX/RY/RZ gates. The 24 pair rotations are eight `RZZ(E_r)`, four `RZZ(E_s)`,
eight `RXX(E_r)`, and four `RXX(E_s)`. These are logical Pauli rotations;
native entangling-gate count and depth depend on device connectivity and
compiler decomposition and have not been measured on hardware.

Each head has an eight-qubit, 256-amplitude statevector. The analytic readout
uses eight local Z, eight local X, and 16 edge observables in each of the Z and
X bases before reduction to 12 invariants. Commuting observables within a basis
do not require separate measurement settings.

## Completed validation results

### Primary seed-0 base run

The context model is a training scaffold: its physics bank, encoder, and orbit
projection initialize both no-context bottleneck models, but its context branch
is absent from the primary quantum and matched-classical classifiers.

| Model | Best epoch | Parameters | Accuracy | Macro AUC | Metric source |
|---|---:|---:|---:|---:|---:|
| Tiny context pretrainer | 18 | 272,805 | 98.1833% | 0.998261 | development validation |
| Tiny D4-ORQB | 17 | 245,221 | 98.4289% | 0.998171 | development validation |
| Tiny matched classical | 20 | 245,221 | 98.4632% | 0.998227 | development validation |
| Equal Q/C probability ensemble | fixed pair | 490,442 | 98.6003% | 0.998713 | development validation |

Probability averaging, not logit averaging, is the defined ensemble operation.
The two individual bottlenecks differ by six validation examples, with the
classical model ahead by 0.034 percentage point. This is a tie, not evidence of
quantum advantage. The circuit was nevertheless trained: saved histories have
nonzero core gradients and the seed-0 circuit parameter update has L2 norm
1.974.

The equal `q0`/`c0` probability ensemble was later frozen as the secondary
`qc0` test rule. A validation-tuned Q/C/context ensemble was explored but was
not frozen or reported on test because its weights were selected on the same
validation data.

### End-to-end replication and optimization instability

| Seed | Context pretrainer accuracy | D4-ORQB accuracy | Matched classical accuracy |
|---:|---:|---:|---:|
| 0 | 98.1833% | 98.4289% | 98.4632% |
| 1 | 95.5610% | 95.5953% | 95.8867% |
| 2 | 34.0151% (33.3333% balanced) | not run | not run |

Seed 2 selected one class for every validation image and was stopped after the
context stage; spending GPU time on its downstream cores would not have tested
the intended bottleneck comparison. The roughly 2.6-point seed-0/seed-1 gap and
seed-2 collapse show that the full training procedure is not yet stable. A
single 98.43% run must not be presented as a seed-robust estimate.

To separate backbone failure from bottleneck behavior, the completed
fixed-backbone replication reused the validated seed-0 context backbone and
trained independently initialized quantum and matched-classical cores under
two additional seeds. This is a controlled paired comparison of bottlenecks,
not an independent end-to-end replication, and it cannot estimate
whole-pipeline variance.

| Core seed | D4-ORQB accuracy | D4-ORQB macro AUC | Matched classical accuracy | Matched classical macro AUC | Q minus C |
|---:|---:|---:|---:|---:|---:|
| 0 | 98.4289% | 0.998171 | 98.4632% | 0.998227 | -0.0343 point |
| 1 | 98.3432% | 0.997914 | 98.4518% | 0.998206 | -0.1085 point |
| 2 | 98.4746% | 0.998004 | 98.4918% | 0.998137 | -0.0171 point |
| Mean +/- sample SD | 98.4156% +/- 0.0667 point | 0.998029 +/- 0.000130 | 98.4689% +/- 0.0206 point | 0.998190 +/- 0.000047 | -0.0533 +/- 0.0486 point |

The unweighted three-quantum-member ensemble (`q3`) reached 98.5775%
development accuracy and 0.998590 macro AUC; the corresponding classical
ensemble (`c3`) reached 98.5718% and 0.998692. The six members have the same
fixed split and seed-0 backbone, so these small across-core dispersions must
not be reported as three independent end-to-end seeds. In particular, the
paired mean of -0.0533 point does not support a quantum-superiority claim.

### Radial/translation/hierarchical fine-tuning

The completed radial experiment warm-started the base quantum and classical
checkpoints, expanded the stem from eight to ten physics channels with the new
channel weights initialized to zero, applied up-to-two-pixel translations with
probability 0.5, and used a 0.3-weight hierarchical auxiliary loss.

| Model | Best epoch | Parameters | Accuracy | Macro AUC | Delta from corresponding base |
|---|---:|---:|---:|---:|---:|
| Radial D4-ORQB | 19 | 246,021 | 98.3147% | 0.998189 | -0.114 point |
| Radial matched classical | 16 | 246,021 | 98.3318% | 0.997637 | -0.131 point |

This recipe did not improve validation accuracy and is retained as a negative
ablation/robustness experiment. Because radial channels, translation, auxiliary
loss, and warm-start fine-tuning changed together, it is not a clean estimate
of the causal effect of the two radial channels alone.

### Repaired LensPINN-small on the same split

The archived LensPINN-small implementation could not serve as a trainable
same-protocol baseline without repair. Its hard `.long()` coordinate splat
severed gradients to the tokenizer/inversion path, the grid had an off-by-one
coordinate issue, and a registered 942,337-parameter block was unused in the
forward graph. The nominal archived count was 7,173,654.

The repaired/adapted baseline uses a centered float32 four-neighbor bilinear
soft splat, constrained Einstein radius, logits-based cross entropy, and no dead
registered block. Gradient assertions confirmed nonzero tokenizer and inversion
gradients. This is transparently labelled a repaired adaptation, not an exact
reproduction of the archived hard model.

| Model | Best epoch | Active parameters | Accuracy | Macro AUC | Official test? |
|---|---:|---:|---:|---:|---:|
| Repaired LensPINN-small | 10 | 6,235,157 | 84.9977% | 0.950005 | no |

On this one shared split/run comparison, the base D4-ORQB is 13.43 percentage
points more accurate with 25.43 times fewer parameters. This observed gap is
not yet a seed-robust superiority claim. The published LensPINN results use a
different 9,000-image, 64 x 64, 80/20 protocol and are not directly comparable
to this Model-I development validation.

## Other completed negative or noncompetitive experiments

- BatchNorm produced severe train/eval running-statistic instability in the
  eight-view orbit setting. GroupNorm removed that failure mode.
- Eight circuit heads with two reuploads doubled the quantum count to 176 but
  reached only 86.53% on the pilot split; adding heads was not a reliable route
  to capacity.
- Four heads with three reuploads used 132 quantum parameters and gained only
  about 0.17 point on the pilot while increasing time by roughly 22%; two
  reuploads were retained.
- Direct 128 x 128 training remained at chance (33.33%, macro AUC 0.4945) in
  the resolution pilot.
- A 96-to-128 curriculum lacking pixel-scale-normalized physics channels
  reached only 92.30% by epoch 8 and was stopped as noncompetitive.
- The larger `small` encoder's context stage reached only 87.73% by epoch 10,
  far below the tiny seed-0 scaffold, and was stopped before quantum/classical
  downstream training. It is a negative optimization result, not proof that the
  larger capacity is intrinsically worse.
- Archived learned-encoder ablations placed matched quantum and classical
  models near 81.8%; with a frozen representation, quantum reached 69.72% and
  classical 74.99%. These earlier results motivated joint representation
  learning and a strict parameter-matched control.
- The earlier 172,379-parameter P4M notebook reported 96.28% accuracy and
  0.9952 macro AUC on the official test. Its sequential overlapping circuit did
  not establish the structural equivariance proved for D4-ORQB, and its prior
  test exposure is historical context rather than a new-study baseline.

All failed and stopped GPU jobs and their artifacts are retained. A stopped
configuration is not silently converted into a successful scientific result.

## Completed finite-shot validation

The development-validation-only finite-shot evaluator completed for the seed-0
base D4-ORQB checkpoint. It has no test-data argument and records that the
official test cache was not touched. The predeclared protocol sampled joint
eight-bit outcomes in the computational Z and all-qubit-Hadamard X settings,
used 256 and 1,024 nested shots, and ran independent measurement seeds 17, 42,
and 314159. The direct plug-in estimator was primary; an ordered distinct-shot
U-statistic for quadratic terms was retained as a bias sensitivity analysis.

| Readout | Accuracy across three shot seeds | Macro AUC mean | Result versus -0.5-point margin |
|---|---:|---:|---:|
| Analytic replay | 98.4175% | 0.998167 | reference |
| 256-shot plug-in | 98.3870% +/- 0.0584 point (98.3204--98.4289%) | 0.998073 | all seeds noninferior |
| 256-shot U-statistic | 98.3851% +/- 0.0531 point (98.3261--98.4289%) | 0.998052 | all seeds noninferior |
| 1,024-shot plug-in | 98.3985% +/- 0.0282 point (98.3661--98.4175%) | 0.998139 | all seeds noninferior |
| 1,024-shot U-statistic | 98.4004% +/- 0.0206 point (98.3775--98.4175%) | 0.998138 | all seeds noninferior |

The reported `+/-` values are descriptive sample standard deviations across
three independent simulated measurement runs; predictions were not averaged.
Every shot-count/seed/estimator combination passed the predeclared paired,
true-class-stratified 10,000-resample noninferiority test. The worst one-sided
95% lower bound was -0.1657 percentage point, above the -0.5-point margin. The
analytic replay differed from the stored seed-0 prediction on two of 17,504
classes (99.9886% agreement) and passed the separately predeclared checkpoint
replay gate.

Four heads and two measurement bases imply 2,048 equivalent circuit shots per
image at 256 shots and 8,192 at 1,024 shots. Across all 17,504 validation images,
one 1,024-shot replicate corresponds to 143,392,768 circuit shots; the three
predeclared replicates correspond to 430,178,304. The result JSON has SHA-256
`5dfe19a161aba5b6e3d8b360f0f264efd38660cebc891b1b047770bbb8da8371`.

This remains ideal sampling from simulated statevectors. It does not measure
device noise, readout error, compilation overhead, decoherence, connectivity,
or hardware wall time. D4 invariance holds for the finite-shot estimator in
distribution, not for each independent finite sample.

## Locked v3 protocol and official Model-I evaluation

The locked evaluator uses a two-stage workflow. `freeze` validates completed
development artifacts, records code, runtime, checkpoint, split, and cache
hashes, and freezes probability ensembles and signed comparison thresholds
without opening test arrays. `run-test` requires the frozen manifest digest,
replays validation first, and writes a durable access marker immediately before
any test-cache access. It then verifies the test hashes and development/test
disjointness, performs inference, and seals ordered per-example predictions,
Wilson intervals, paired bootstrap intervals, exact McNemar tests, and
Holm-adjusted comparisons.

### Pre-marker aborts and replay diagnostics

Two frozen attempts stopped safely before official-test access:

- v1 froze six base D4 members, repaired LensPINN, and two radial members under
  manifest digest
  `1d4f420cda71a4229a573f1b30c1b3e9d98fbec1d15dc5496ab7e2d84fb562ff`.
  The repaired LensPINN validation replay had a maximum probability difference
  of 0.030046, above the frozen 0.002 gate. The job exited before a test marker;
  its official-test output directory is empty.
- A dedicated development-only diagnostic established batch- and
  repeat-dependent CUDA soft-splat accumulation. At batch size 128 the saved
  classes were reproduced, but the maximum saved-probability difference was
  0.030046 and two repeats differed by as much as 0.020875. Other batch sizes
  changed saved class decisions. LensPINN therefore remains a development-only
  adapted baseline rather than weakening a global replay gate for one model.
- v2 excluded LensPINN but retained the radial models under manifest digest
  `a6874498160fc10a6546ba997e88df551afa7146439e770e34d35d7d8bdade96`.
  It exited before the marker because `q0` differed from its stored
  probabilities by 0.004275, above 0.002, although every class prediction was
  unchanged. Its official-test output directory is also empty.
- A development-only D4 diagnostic then showed that all six base members
  reproduced their saved class decisions. The three quantum members had
  maximum probability differences of 0.003869--0.004763, means no larger than
  `1.09e-5`, 99th percentiles no larger than `2.05e-4`, and maximum metric drift
  `1.071e-4`. In contrast, the radial quantum/classical replays had maxima
  0.053535/0.062402 and each changed six class decisions. Because the radial
  recipe was already a negative development ablation, it was excluded from the
  official-test family rather than accommodated post hoc.

These aborts are protocol failures, not hidden test evaluations: neither v1
nor v2 produced `TEST_ACCESS_MARKER.json`, validation replay output, test
predictions, or result metrics. The replay diagnostics used development
validation only.

### Prospective v3 lock and audit hashes

After those diagnostics but before any D4-ORQB official-test inference, v3
froze only `q0/q1/q2` and the matched `c0/c1/c2`. It also froze four unweighted
probability ensembles: `q3=(q0,q1,q2)`, `c3=(c0,c1,c2)`, `qc0=(q0,c0)`, and
`qc6=(q0,q1,q2,c0,c1,c2)`. The H200 inference environment was PyTorch 2.1.2,
CUDA 11.8, cuDNN 8.7, bfloat16 autocast, batch size 128, four workers, and
loader seed 42.

The v3 validation replay required all five gates simultaneously:

1. exact equality of predicted classes;
2. maximum absolute probability difference at most 0.005;
3. mean absolute probability difference at most `2e-5`;
4. 99th-percentile absolute probability difference at most `3e-4`; and
5. maximum absolute drift among frozen metrics at most `2e-4`.

All six members passed the probability and exact-decision gates, while every
member and ensemble passed the metric/confusion gate, in a separate pretest job
whose interface had no official-test path or argument. The largest observed
values were 0.004763 maximum probability difference, `1.09e-5` mean
difference, `2.05e-4` 99th percentile, and `1.071e-4` metric drift, with exact
member class decisions. Only then was the frozen confirmation digest inserted
into the official-test job.

| Audit artifact | SHA-256 |
|---|---|
| Frozen v3 manifest / confirmation digest | `2a3b55de874522bf6d0bd1c3846536a2db11b0ca431011aa5b5f6e8ecd84c50b` |
| `seal.json` | `8f8b75d9aaf4a031387751a18bb0d815a53cc5d7aacaff5c2fd585605800ec2b` |
| Development-only v3 pretest replay | `abfc5d7a618981b89fc00d3d9ab53f70ace0419469e9ac9e3af0e3603f7974ee` |
| Durable test-access marker | `a348eac18e309fbcd2889cd5eeec53885b2ad81c0980a25e6289c9ff610d5902` |
| Official-job validation replay | `f99fe01ab4d9a0bcdb577a485e48165f3c178df5f3450c15e6e709ee458a22c5` |
| Official result | `86b0b1df9cf33a772719799dd0d7632e81312e837c0de39e0003fc4e8ca83340` |
| Official artifact seal | `9d53883684e416cd6fc8e0924cc692df75315d6918ee97a358c13f8164db23d8` |

The frozen test artifacts matched the expected SHA-256 values: manifest
`02b8395f1f895e39b84b41053fe87061e975919c68eafd7789d099396a4a9bb5`,
images
`056159ec214987f62f5060f163449392cd9987e244b00cf01baabedafba43758`,
labels
`50457c13893e97a995a730f968ac53aa0de0c0bd22305089fc1ca1459d3546da`,
and metadata
`cf0dade18aecdfa007c672276d2e510b4ab383a482fff192796f96e0e8793c6d`.
The post-marker check again found zero model-visible development/test digest
intersections.

### Official test metrics

All rows below are from that one sealed v3 Model-I test pass. Ensemble parameter
counts sum independently stored members and therefore represent total learned
parameters used for inference, not the size of one shared model.

| Frozen result | Composition | Parameters | Accuracy (Wilson 95%) | Macro AUC | Macro F1 | NLL |
|---|---|---:|---:|---:|---:|---:|
| `q0` | one D4-ORQB | 245,221 | 98.4667% (98.2573--98.6513%) | 0.998362 | 98.4648% | 0.05652 |
| `q1` | one D4-ORQB | 245,221 | 98.4333% (98.2218--98.6200%) | 0.998320 | 98.4311% | 0.05819 |
| `q2` | one D4-ORQB | 245,221 | 98.3667% (98.1511--98.5575%) | 0.998369 | 98.3650% | 0.05775 |
| `q3` | three D4-ORQB members | 735,663 | 98.5867% (98.3849--98.7635%) | 0.998907 | 98.5849% | 0.05028 |
| `c0` | one matched classical | 245,221 | 98.5267% (98.3211--98.7074%) | 0.998466 | 98.5250% | 0.05579 |
| `c1` | one matched classical | 245,221 | 98.4667% (98.2573--98.6513%) | 0.998382 | 98.4650% | 0.05557 |
| `c2` | one matched classical | 245,221 | 98.4133% (98.2006--98.6013%) | 0.998546 | 98.4121% | 0.05496 |
| `c3` | three matched classical members | 735,663 | 98.5800% (98.3778--98.7573%) | 0.998985 | 98.5785% | 0.05031 |
| `qc0` | `q0/c0` mean | 490,442 | 98.6133% (98.4134--98.7884%) | 0.998963 | 98.6116% | 0.05099 |
| `qc6` | all six members | 1,471,326 | 98.6667% (98.4703--98.8382%) | 0.999115 | 98.6649% | 0.04818 |

The five signed accuracy gates were frozen before test. Differences and paired
bootstrap intervals are in percentage points.

| Frozen comparison | A minus B | Paired bootstrap 95% interval | Minimum acceptable | Exact McNemar p | Holm p | Gate |
|---|---:|---:|---:|---:|---:|---:|
| `q3 - c3` | +0.0067 | -0.1333 to +0.1467 | -0.2500 | 1.000000 | 1.000000 | pass |
| `q3 - qc0` | -0.0267 | -0.1467 to +0.0933 | -0.2500 | 0.740653 | 1.000000 | pass |
| `q3 - q0` | +0.1200 | +0.0067 to +0.2333 | 0.0000 | 0.050452 | 0.201810 | pass |
| `q0 - c0` | -0.0600 | -0.2267 to +0.1000 | -0.2500 | 0.523301 | 1.000000 | pass |
| `qc0 - q0` | +0.1467 | +0.0400 to +0.2533 | 0.0000 | 0.010338 | 0.051688 | pass |

All predeclared minimum-difference criteria passed, but those margins are
acceptability gates, not proof of superiority. In particular, `q3` and `c3`
are empirically tied. The positive bootstrap intervals for `q3-q0` and
`qc0-q0` show ensemble gains on this test set, while their Holm-adjusted exact
McNemar p-values are 0.2018 and 0.0517; neither is a family-wise 0.05
superiority result. `qc6` was a frozen reported ensemble and achieved the
highest point estimate, but no post-test comparison was added for it.

Relative to the 6,235,157-parameter repaired LensPINN-small development
baseline, a single `q0` uses 25.43 times fewer parameters, `q3` uses 8.48 times
fewer, `qc0` uses 12.71 times fewer, and `qc6` uses 4.24 times fewer. LensPINN
was excluded from official testing for the replay reason above, so these are
parameter ratios and same-split development context, not official-test
head-to-head results.

### Interpretation and remaining limitations

- The matched quantum and classical results are tied. This study demonstrates
  a compact, trainable hybrid with exact D4 invariance, not quantum advantage.
- Seeds 0--2 for the bottleneck comparison share the selected seed-0 backbone.
  They are conditional core replicates; the collapsed end-to-end seed 2 and
  weaker end-to-end seed 1 remain evidence of upstream optimization
  instability.
- Circuit inference used ideal complex64 statevectors with bfloat16 surrounding
  neural layers. The finite-shot experiment adds ideal measurement sampling,
  not hardware noise or a hardware runtime demonstration.
- The result covers one simulated gravitational-lensing dataset and may exploit
  simulator-specific central/radial structure. Cross-simulator, shifted-lens,
  and real-observation robustness remain open.
- The prospective v3 procedure prevents selection on this test pass, but the
  repository's official test set is historically non-pristine because older
  notebooks had already used it.
- The repaired LensPINN is an adapted development baseline, and the additional
  16-element bottleneck automorphism may constrain expressivity. Logical gate
  counts have not been converted into a device-specific compiled resource
  estimate.
- A workshop submission still requires careful positioning, independent
  replication, and transparent reporting. High accuracy and parameter
  efficiency do not guarantee NeurIPS-workshop acceptance.

## Primary literature and methodological references

- Cohen and Welling, [Group Equivariant Convolutional
  Networks](https://proceedings.mlr.press/v48/cohenc16.html) (ICML 2016).
- Chang et al., [Approximately Equivariant Quantum Neural Network for p4m Group
  Symmetries in Images](https://arxiv.org/abs/2310.02323) (IEEE Quantum Week
  2023).
- San Sebastian, Canizo, and Orus, [Image Classification with
  Rotation-Invariant Variational Quantum
  Circuits](https://arxiv.org/abs/2403.15031) (Physical Review Research 2025).
- Perez-Salinas et al., [Data re-uploading for a universal quantum
  classifier](https://arxiv.org/abs/1907.02085) (Quantum 2020).
- Schuld, Sweke, and Meyer, [The effect of data encoding on the expressive power
  of variational quantum machine learning
  models](https://arxiv.org/abs/2008.08605) (Physical Review A 2021).
- Cerezo et al., [Cost Function Dependent Barren Plateaus in Shallow
  Parametrized Quantum Circuits](https://arxiv.org/abs/2001.00550) (Nature
  Communications 2021).
- Senokosov et al., [Quantum machine learning for image
  classification](https://arxiv.org/abs/2304.09224) (Machine Learning: Science
  and Technology 2024).
- Ojha et al., [LensPINN: Physics Informed Neural Network for Learning Dark
  Matter Morphology in
  Lensing](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_78.pdf)
  (NeurIPS ML4PS 2024).
- Bowles et al., [Better than classical? The subtle art of benchmarking quantum
  machine learning models](https://arxiv.org/abs/2403.07059) (2024).
