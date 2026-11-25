# PDEep/simple

Numerical PDE datasets, deterministic A/C/D/E/F/G/H-Geo reconstructions, conditional latent flows and classical baselines. The same CLI handles every supported tier; no notebook-specific training code is needed.

## Installation and quick start

Use Python 3.9+ with NumPy, SciPy, PyTorch and Matplotlib. The Git repository root contains `pyproject.toml` and `pdeep_simple/`:

```bash
git clone https://github.com/NiciiA/PDEep-simple.git
cd PDEep-simple
python -m pip install -e .
python -m pdeep_simple --help
```

[Open the Colab notebook](https://colab.research.google.com/github/NiciiA/PDEep-simple/blob/main/notebooks/pdeep_colab.ipynb), or upload [notebooks/pdeep_colab.ipynb](notebooks/pdeep_colab.ipynb) to Colab before the notebook is published on GitHub. It runs the complete four-model workflow with editable settings. Local checks are available with `python -m pdeep_simple check` and `python -m unittest discover -s tests -v`.

Run the commands below from the package root. Data and outputs stay under `data/` and `runs/`. Generation counts are **per PDE–geometry combination**, not totals. Repeating an identical generation command reuses completed files; changing its settings requires a new output directory. Training also requires a new run directory or `--resume`.

## Dataset generation

| Tier | Supported PDE names | Unknowns in Inverse mode |
|---|---|---|
| T1 | `poisson`, fixed unit square | Source amplitudes; four fixed Gaussian bases by default |
| G1 | `poisson,diffusion,elliptic_adr,heat,reaction_diffusion,wave,damped_wave` | Source and/or initial fields; operator known |
| T2 | G1 linear families, constant coefficient | Positive scalar conductivity, diffusivity or wave speed; source/initial fields known |
| T3 | `heat,diffusion` | High-dimensional initial field, or spatial diffusion coefficient |
| T4 | `semilinear,p_laplacian,eikonal` | Source field; Eikonal speed |

Geometry names: `square,rectangle,disk,ellipse,lshape,annulus,hole,polygon,triangle,trapezoid,star,cross,dumbbell,crescent,two_holes`. T1 requires `square`. Nonlinear equations stay in T4; incompatible tier/PDE selections fail explicitly. Without a config or explicit new PDE names, original generation defaults are retained.

### T1: fixed-domain source inversion

```bash
python -m pdeep_simple data --tier T1 --pdes poisson --geometries square \
  --train 32 --val 4 --test 4 --workers auto --out data/t1_small
```

### G1: varied domains and linear PDEs

```bash
python -m pdeep_simple data --tier G1 --pdes poisson,heat,wave \
  --geometries square,disk,lshape --train 8 --val 2 --test 2 \
  --workers auto --out data/g1_small
```

This produces 72 training, 18 validation and 18 test cases. Geometry sizes and physical inputs vary between cases.

### T2: unknown scalar parameters

```bash
python -m pdeep_simple data --tier T2 --pdes poisson,heat \
  --geometries square,disk --train 8 --val 2 --test 2 \
  --workers auto --out data/t2_small
```

The legacy T2 `poisson` name includes unknown constant conductivity. Hybrid parabolic cases also hide part of the initial field.

### T3: unknown spatial fields or coefficients

```bash
python -m pdeep_simple data --tier T3 --pdes heat,diffusion \
  --geometries square,disk --train 8 --val 2 --test 2 \
  --workers auto --out data/t3_small
```

Unconfigured T3 heat uses 32 frequency-damped basis amplitudes. T3 diffusion instead hides the spatial coefficient field.

### T4: nonlinear equations

```bash
python -m pdeep_simple data --tier T4 --pdes semilinear,p_laplacian,eikonal \
  --geometries square,disk,lshape --train 8 --val 2 --test 2 \
  --workers auto --out data/t4_small
```

All examples use homogeneous Dirichlet boundaries. Inspect meshes, physical fields, basis counts, parameters and numerical references with:

```bash
python -m pdeep_simple inspect --data data/g1_small --split train \
  --random 3 --seed 9 --out runs/inspection
```

## Dataset config

[configs/dataset.toml](configs/dataset.toml) is optional. It controls mathematics; the CLI still chooses tier, PDEs, geometries, split sizes, workers, seed and destination. Training and evaluation read the saved dataset without needing the TOML file again. Python 3.11+ uses `tomllib`; installation supplies `tomli` on 3.9/3.10.

```bash
python -m pdeep_simple data --config configs/dataset.toml --tier G1 \
  --pdes poisson,diffusion,heat,wave --geometries square,disk,lshape,star \
  --train 32 --val 4 --test 4 --workers auto --out data/g1_configured
```

| Section | Main settings |
|---|---|
| `basis` | Count, Gaussian/uniform amplitude law, mean/std, count scaling, Gaussian widths |
| `pdes.NAME` | Per-field basis counts; conductivity/diffusivity/speed; reaction, advection, damping, beta or p; forcing frequency |
| `geometries.NAME` | Aspect/axis ratios, triangle apex, trapezoid width, star/polygon counts, neck/arm widths, hole radii |
| `conditioning` | Hybrid reveal fractions for source, initial and coefficient fields |
| `numerics` | Resolution, frames, time step, Newton limits/tolerance, p-Laplacian regularization |

A scalar is fixed; `[low, high]` samples uniformly. Integer count ranges include both endpoints. Per-field counts override `basis.count` and may vary independently between cases. `scale_by_count=true` divides amplitude std by sqrt(K); overlapping bases can still correlate fields. Each basis has unit domain RMS, but solutions are **not** independently normalized. Centers lie inside the mesh; widths are fractions of bounding-box dimensions. Actual bases, amplitude labels and prior parameters are saved separately; hidden labels do not become conditioning inputs.

Explicit `--resolution`, `--frames` and `--dt` override config numerics. Hybrid ranges are sampled once per case, then nodal masks use that reveal probability. A fraction of known nodes is not a fraction of recoverable information: K independent exact source rows can determine all K amplitudes.

Original unconfigured defaults include Poisson amplitudes N(0, 9 I) with K=4, heat diffusivity U(0.03, 0.2), wave speed U(0.5, 1.2), T2 conductivity U(0.5, 2), and semilinear beta U(0.5, 3) with source amplitude std 12 and conductivity in (0.5, 2). The example config deliberately changes these distributions; its comments document each implemented option.

## Training

All variants use the same data, conditioning and field decoder interfaces. These are modest examples, not evidence of trained accuracy.

### A-Geo

```bash
python -m pdeep_simple train --data data/g1_small --model A --size s \
  --modes forward,inverse,hybrid --loss field=1 --updates 400 \
  --batch 2 --queries 512 --device auto --run runs/a
```

### C-Geo

```bash
python -m pdeep_simple train --data data/g1_small --model C --size s \
  --modes forward,inverse,hybrid --loss field=1 --updates 400 \
  --batch 2 --queries 512 --device auto --run runs/c
```

### D-Geo

```bash
python -m pdeep_simple train --data data/g1_small --model D --size s \
  --fourier-bands 6 --modes forward,inverse,hybrid --loss field=1 \
  --updates 400 --batch 2 --queries 512 --device auto --run runs/d
```

D reuses A's encoder and attention decoder. A already has five fixed Fourier bands; D uses six configurable bands at frequencies `2*pi*2**j`, including the raw normalized x/y/t coordinates, projected to the same width. Space is normalized by the mesh bounding box and physical time by its interval. Physical query coordinates remain available for conditioning and derivatives. D supports every existing PDE and geometry with the same losses.

### E-Geo

```bash
python -m pdeep_simple train --data data/g1_small --model E --size s \
  --wave-modes 32 --modes forward,inverse,hybrid --loss field=1 \
  --updates 400 --batch 2 --queries 512 --device auto --run runs/e_initial
```

E shares A's backbone. Every undamped `wave` case uses initial-condition-aware evolution, including when the operator is hidden. Initial displacement and velocity are supplied when known and inferred by the existing initial-state head when hidden. Hybrid masks retain each known initial node and infer the rest; P1 interpolation evaluates these fields at arbitrary query locations. One E model handles Forward, Inverse and Hybrid. Hidden targets never construct the operator or initial fields at inference.

The wave decoder uses `u = u0 + t*v0 + t^2*r(t)`, where t is physical time relative to the initial time. When the positive spatial operator is fully known, it adds `u_m(t) - u_m(0) - t*v_m(0)` from the existing mass-normalized FEM modes. Their analytic evolution retains constant/cosine forcing, including resonance; fully known initial fields and forcing are projected, while hidden modal coefficients are inferred from conditioning. Unknown operators simply omit this contribution and still use the initial-aware decoder.

Velocity is the physical-time derivative of displacement, using analytic modal derivatives and autograd for A's residual decoder, with separate u/v output scales. The chosen initial fields are exact at t=0 up to interpolation and floating-point accuracy. Training puts one quarter of E wave queries at the initial time (at least one), supervising both u and v through the usual reconstruction loss; validation draws stay unchanged. The redundant kinematic penalty is skipped for this branch and logged as zero; the existing finite-difference evaluation diagnostic remains available and need not be zero on coarse time grids. Inferred initial fields can still be wrong, and the construction does not enforce the full wave equation or boundary conditions. Other optional physics losses keep their existing definitions.

Other PDEs and `damped_wave` retain A's ordinary decoder. Modes keep their existing interpolation and per-mesh/operator/device memory cache, using O(nodes × modes) storage per operator; resume rebuilds the cache. Only low modes are computed (at most interior degrees of freedom minus one, except a single interior node). No dataset preparation or new parameters are needed. Use a new run directory for this changed E wave behavior. D/E retain normal device, resume and staged-training support; `--flow` remains limited to A/C.

```bash
python -m pdeep_simple eval --data data/g1_small --run runs/d --device auto
python -m pdeep_simple eval --data data/g1_small --run runs/e_initial --device auto
python -m pdeep_simple plot --run runs/e_initial --what losses,fields
```

Evaluation keeps the existing metrics and grid/refinement options. E records `modal_wave` (0/1) and the actual `wave_modes` count per case; `summary.csv` averages `modal_wave` into the fraction using FEM modes within each PDE/geometry/conditioning/sensor group. Plotting saved E wave cases writes an `*_e_wave.png` panel with FEM and predicted u/v and absolute errors, labeled with or without modes. Use `plot --runs runs/a runs/c runs/d runs/e_initial` for the usual comparison.

### F-Geo: learned wave evolution

```bash
python -u -m pdeep_simple train --data data/colab_g2 --model F --size m \
  --evolution-steps 32 --modes forward,inverse,hybrid --loss field=1 \
  --updates 5000 --batch 16 --queries 4096 --observations 0,8,16,32,64,128 \
  --noise .01 --seed 42 --validate-every 500 --device cuda --run runs/colab_f
```

F uses A's conditioning encoder to select a coherent nodal initial u/v field through the existing masks. A small two-layer mesh graph learns physical acceleration from the current state, time and available conditioning. Symplectic-Euler-like updates advance v, then u, over the physical interval. The full trajectory is computed once per case/conditioning draw and reused by field and physics losses and query chunks, with P1 spatial and linear temporal interpolation. Reconstruction at later times backpropagates through the entire rollout. Known zero Dirichlet nodes stay zero. The explicit integrator is not unconditionally stable or energy conserving, and 32 steps may miss rapid oscillations. Interpolation does not enforce continuous `u_t=v`; the usual kinematic metric/loss remains active when requested.

### G-Geo: Fourier features and SIREN

```bash
python -u -m pdeep_simple train --data data/colab_g2 --model G --size m \
  --fourier-bands 6 --siren-omega 30 --modes forward,inverse,hybrid --loss field=1 \
  --updates 5000 --batch 16 --queries 4096 --observations 0,8,16,32,64,128 \
  --noise .01 --seed 42 --validate-every 500 --device cuda --run runs/colab_g
```

G keeps D's normalized Fourier coordinates and conditioned attention decoder, replacing its final field MLP with two sinusoidal hidden layers and a linear output. The frequency-scaled uniform initialization follows the [SIREN authors' implementation](https://github.com/vsitzmann/siren/blob/master/modules.py). This readout acts on coordinate, geometry, physical and observation features together. It is used for every PDE. High-frequency features can also amplify derivative losses; improved reconstruction is an experimental hypothesis.

### H-Geo: learned propagation kernels

```bash
python -u -m pdeep_simple train --data data/colab_g2 --model H --size m \
  --green-sources 64 --modes forward,inverse,hybrid --loss field=1 \
  --updates 5000 --batch 16 --queries 4096 --observations 0,8,16,32,64,128 \
  --noise .01 --seed 42 --validate-every 500 --device cuda --run runs/colab_h
```

H uses the same masked initial-field approach and predicts `u=u0+t*v0+t^2*R`, with physical velocity obtained by autograd. R is a weighted application of small geometry/operator-conditioned kernels to the initial displacement, velocity and known or inferred forcing amplitudes. Known-operator kernels receive neither initial/forcing amplitudes nor observations; a second pass through the **shared** conditioning encoder supplies their operator/geometry features. With hidden operators, the full available conditioning informs those features, including observations and any supplied initial state. No hidden true operator or source enters the prediction.

Spatial integration uses 64 fixed importance samples of lumped FEM node quadrature (all nodes if fewer), with area weights. Cosine forcing uses two causal midpoint samples with `s<=t` and a `(t-s)` factor. Pairwise temporary tensors are limited to 128 output queries by the selected sources. Kernel width is capped at 32; F's graph width at 64. F/H skip penalties for their exactly preserved known initial values and zero boundaries; H also skips the redundant kinematic penalty. Evaluation diagnostics remain available. This is an approximate learned propagation representation, not an exact Green's-function solver or PDE guarantee; finite spatial/time quadrature can miss fine structure and fast forcing.

F/H use their specialized path for every undamped `wave`, including hidden operators. Other PDEs and `damped_wave` retain A's decoder. G uses its sinusoidal readout throughout. Each architecture trains one shared model across Forward/Inverse/Hybrid and mixed PDEs, using existing presets, losses, optimizer, staged training and resume. F/G/H keep ordinary random physical-time training queries; E's initial-time sampling is unchanged. Flow remains limited to A/C. Use separate run directories; these examples do not establish trained accuracy.

```bash
python -m pdeep_simple eval --data data/colab_g2 --run runs/colab_f --frames 5 --device cuda
python -m pdeep_simple eval --data data/colab_g2 --run runs/colab_g --frames 5 --device cuda
python -m pdeep_simple eval --data data/colab_g2 --run runs/colab_h --frames 5 --device cuda
python -m pdeep_simple plot --runs runs/colab_f runs/colab_g runs/colab_h --what losses,comparisons
python -m pdeep_simple plot --run runs/colab_f --what fields
```

With the usual stored time grid, `--frames 5` evaluates 0, .25, .5, .75 and 1; F/G/H field plots retain all five. Existing comparison plots separate u/v, include error versus physical time and can include A/C/D/E and baseline results. Use matching held-out cases, observation draws and training normalization, as for existing comparisons. CUDA/MPS selection remains through `--device`; eigenmodes and numerical FEM solvers are not used by F/G/H prediction.

### A-Geo + Flow

```bash
python -m pdeep_simple train --data data/g1_small --model A --size s --flow \
  --modes forward,inverse,hybrid --loss field=1 --latent-dim 32 \
  --ae-updates 100 --updates 400 --batch 2 --queries 512 \
  --flow-steps 16 --device auto --run runs/a_flow
```

### C-Geo + Flow

```bash
python -m pdeep_simple train --data data/g1_small --model C --size s --flow \
  --modes forward,inverse,hybrid --loss field=1 --latent-dim 32 \
  --ae-updates 100 --updates 400 --batch 2 --queries 512 \
  --flow-steps 16 --device auto --run runs/c_flow
```

### Staged physics training

```bash
python -m pdeep_simple train --data data/g1_small --model C --size s \
  --modes forward,inverse,hybrid \
  --loss "field=1,kinematic=.01,weak=.01,initial=.01,boundary=.01" \
  --staged --stage-start 80 --updates 400 --batch 2 --queries 512 \
  --device auto --run runs/c_staged
```

The same option works with the other deterministic models. Updates 1–80 use only `field=1`; updates 81–400 use exactly the selected `--loss` weights. `--stage-start` is the last field-only update and defaults to 20% of the total (rounded down, at least one); it must be less than `--updates`, which remains the total update count. The model, optimizer and learning rate continue across the transition. Without `--staged`, all selected losses apply from the start. This option is for deterministic models and cannot be combined with `--flow`.

Both CSVs record `stage=field` or `stage=physics`; physics terms are zero during the field stage. Existing loss plots mark the boundary with a dashed vertical line. Validation field error, fixed validation draws and best-checkpoint selection stay the same, with no extra validation pass. Resume restores the saved boundary and loss weights, even if the total update count is extended; a conflicting explicit `--stage-start` is rejected.

| Option | Meaning |
|---|---|
| `--size` | `s,m,l,xl,xxl`; construction prints the parameter count |
| `--updates` | Total desired updates, including earlier updates when resuming |
| `--batch`, `--queries` | Physical cases per update; fresh spatial/time queries per case |
| `--modes` | Conditioning modes sampled during training |
| `--loss` | Weighted terms, e.g. `field=1,aux=.1`; choose only applicable terms |
| `--device` | `auto,cpu,cuda,mps`; Colab notebook training explicitly uses `cuda` |
| `--flow` | Field representation stage followed by conditional latent flow matching |
| `--ae-updates`, `--latent-dim` | First-stage updates and latent dimension; require 1 <= AE updates < total updates |
| `--resume` | Restore `last.pt`, including model, optimizer, RNG and saved stage settings |

`--width`, `--layers`, `--heads` and `--memory` override presets; memory size affects the attention models A/D/E/F/G/H. `--observations`, `--noise`, `--seed`, `--lr` and `--validate-every` retain their normal CLI meanings. Inverse/Hybrid training uses nonzero observation counts. For T2 coefficient-head supervision add `aux=.1`; `kinematic` applies to wave/damped wave, `initial` to known initial nodes, and `boundary` to zero boundary values. `weak` uses P1 test functions and known forcing/operator information (or the T2 scalar prediction); it is not a strong-form derivative guarantee. Eikonal has no elliptic weak loss. The networks do not hard-enforce the PDE or boundary conditions.

Flow's **total** `--updates` includes both stages: the examples use 100 representation updates and 300 flow updates. First train a reference-field encoder and latent-conditioned A/C decoder; then freeze that representation and train a conditional velocity MLP using training-only latent normalization. Physical PDE time and latent ODE flow time are different variables. Inference uses known inputs and observations, never the reference encoder. One latent describes the entire space-time field, including wave u/v, and is reused across query chunks.

`best.pt` uses validation field loss for deterministic models and validation flow-matching loss after entering the flow stage. Sampling rejects an AE-only checkpoint. Flow samples form a learned conditional distribution, **not automatically calibrated Bayesian uncertainty**. Extend a completed run with the normal resume command:

```bash
python -m pdeep_simple train --data data/g1_small --resume runs/a_flow/last.pt \
  --updates 600 --device auto --run runs/a_flow
```

## Baselines

| Method | What it uses and where it is supported |
|---|---|
| `prior` | Physical-input-conditioned prior estimate, deliberately ignoring solution observations. Linear Gaussian field conditioning, T2 24-point scalar quadrature, and supported full/symmetric nonlinear cases |
| `rbf` | Observation-only spatial/time interpolation, separately for u/v. Too few observations use their mean; none use zero. No PDE enforcement |
| `bayes` | Exact Gaussian amplitude posterior for T1/G1 Poisson, diffusion and elliptic ADR with known operator/basis; numerical FEM responses, actual noise and exact source constraints. Saves K-by-K covariance and pointwise std; exact only for the stored discrete model up to rounding |
| `ridge` | T1 numerical basis-response regression with fixed ridge penalty and exact known-source constraints |
| `grid` | T2 bounded scalar prior/likelihood quadrature over 41 forward solves; requires known source and initial state |
| `pod` | Training-only lumped-mass-weighted SVD plus observation-based, noise-weighted ridge. Stationary scalar fields, at least two compatible physical-domain/PDE training cases |
| `gp` | Observation-only zero-mean spatial squared-exponential GP on stationary scalar fields; kernel posterior mean/std and intervals, without PDE/boundary enforcement |
| `map` | Local parameter-space MAP using existing FEM solvers and actual Gaussian/uniform priors; a point estimate, not a posterior mean or uncertainty distribution |

```bash
python -m pdeep_simple baseline --data data/g1_small \
  --methods prior,rbf,bayes,pod,gp --workers auto \
  --normalization-run runs/a --observations 0,8,16,32,64,128 \
  --out runs/baselines
```

Add `map` to the method list and use `--max-cases 2` with a fresh output directory for a small inverse experiment. MAP supports T2 scalars with known source/initial fields, configured T3 smooth coefficient bases, and T1/G1 linear elliptic or T4 semilinear/p-Laplacian source amplitudes. It optimizes at most **16 free parameters**; beta and other declared known parameters stay fixed. Gaussian Hybrid constraints use an SVD nullspace. Partial uniform-source constraints, saturated tanh samples, unknown coefficient interfaces, and joint unknown-scale/exact-coefficient constraints are unsupported. T3 coefficient Hybrid needs a fixed scale. Failed final PDE solves produce no prediction; finite early-stopped iterates have `map_success=0`. Inspect status, initial/final objective, iterations and forward-solve counts in CSV/NPZ output.

Unsupported combinations are recorded in `skipped.csv`, while other methods continue. Bayes does not support dynamic, nonlinear or non-Gaussian cases. The nonlinear prior requires full source information, partial samples that determine all amplitudes, or fully hidden symmetric zero-mean forcing; unknown spatial-coefficient priors are unsupported. POD never rescales unrelated domains: randomly resized G1 domains usually lack matching groups and are skipped; fixed-domain T1 works. Matching-domain meshes use existing FEM interpolation. POD alone needs a learned representation, saved as `OUT/pod.npz`; reuse with `--pod-model` and the same training split/rank. Tune rank/ridge or GP assumptions using validation, never test reference fields.

With zero observations, POD returns the compatible group's training mean; GP returns zero mean and its assumed prior std. A physical-input-conditioned prior can still be nonzero when source information is available.

Useful settings are `--pod-rank 32`, `--pod-ridge 1`, `--gp-lengthscale .2`, `--gp-amplitude 1` (physical units), and `--map-maxiter 100`. GP intervals assume that kernel and need empirical calibration checks. `--workers auto` uses bounded CPU processes with one BLAS/OpenMP thread each; `--memory-gb` controls the conservative memory budget. Operators/basis responses are reused. Baselines have no automatic resume: completed case CSVs survive interruption, but use a fresh output directory when changing settings.

## Evaluation

```bash
python -m pdeep_simple eval --data data/g1_small --run runs/a --split test \
  --observations 0,8,16,32,64,128 --device auto --out runs/a/evaluation
python -m pdeep_simple eval --data data/g1_small --run runs/a_flow --split test \
  --observations 0,8,16,32,64,128 --samples 16 --flow-steps 16 \
  --save-samples 2 --device auto --out runs/a_flow/evaluation
```

Use the same commands for `runs/c` and `runs/c_flow`. Fixed draws give every method identical observations for the same case, seed and count; sensor sets are not nested across counts. Noise is independent Gaussian with absolute std `--noise` (default .01), including velocity. Evaluation saves per-case `metrics.csv`, grouped `summary.csv`, settings, skipped cases and selected field NPZs. Filters include `--pde`, `--geometries`, `--modes` and `--max-cases`; `--frames` selects stored physical times (default five).

Errors use lumped spatial weights and trapezoidal time weights. `mse`/`rmse` are physical-unit errors; `normalized_mse=MSE/s_u^2` and `normalized_rmse=RMSE/s_u` use fixed **training-set RMS** with a .01 floor, separately for u/v. Relative L2 divides by reference energy and can be large near zero fields. Aggregates average per-case errors. Baselines should use `--normalization-run` matching the neural run, especially for geometry-family holdouts (`train --geometries square,disk`). No held-out magnitude sets an input scale.

Flow evaluation reports sample-mean error, unbiased sample std, empirical central intervals and coverage. With one sample, std/intervals are undefined. `--save-samples` retains a few complete fields; it does not change the number used for statistics. GP/Bayes use their analytic pointwise std. Deterministic, POD and MAP predictions have no claimed posterior uncertainty.

Optional query-resolution experiment:

```bash
python -m pdeep_simple eval --data data/g1_small --run runs/a --modes inverse \
  --observations 32 --grids 128,256,512 --max-cases 2 --chunk 512 \
  --device auto --out runs/a_multigrid
```

`baseline` accepts the same grid/chunk options. Connectivity masks exterior regions and holes; only valid-domain points enter errors. A finer query grid **does not improve the numerical reference**. Grid evaluation defaults to at most eight cases unless overridden. Optional `--refine 1..3` requires an explicit case limit and refines the reference with prolonged original P1 inputs on the same polygon, halving the internal time step. Observations stay on the original case. `reference_rmse` and refinement CSVs are convergence diagnostics; stored-model Bayes is not automatically exact against a refined reference.

## Plotting

```bash
# Training stages and saved prediction fields
python -m pdeep_simple plot --runs runs/a runs/c runs/a_flow runs/c_flow \
  --what losses --out runs/loss_comparison
python -m pdeep_simple plot --run runs/a --what fields --out runs/a/plots

# Error curves from matching saved evaluation tables
python -m pdeep_simple plot --runs runs/a runs/c runs/a_flow runs/c_flow runs/baselines \
  --what comparisons --metric normalized_rmse --out runs/comparisons

# Same case: reference, all predictions/errors, conditioning differences and uncertainty
python -m pdeep_simple plot --runs runs/a runs/c runs/a_flow runs/c_flow runs/baselines \
  --data data/g1_small --pde poisson --geometry disk --case 0 \
  --modes forward,inverse,hybrid --observations 32 --grid 128 \
  --samples 16 --flow-steps 16 --save-samples 2 --what fields,conditioning,uncertainty \
  --device auto --out runs/same_case

# Saved analytic Bayes/GP uncertainty (flow --what fields also shows samples)
python -m pdeep_simple plot --run runs/baselines --what uncertainty \
  --out runs/baseline_uncertainty
```

Comparisons use shared physical cases, seeds, noise, observation counts, times, grids and reference levels. Modes and u/v stay separate; PDE panels retain relative L2 and stationary cases omit time curves. Fields share color scales. Flow-matching loss is distinct from normalized field MSE; inspect matching training scales when comparing runs. `--case` indexes the selected PDE/geometry subset. Same-case plots recompute with identical inputs and reuse saved baseline settings. Individual samples, conditioning differences, GP kernel std and Bayesian std keep their different interpretations.

## Colab and saving results

[notebooks/pdeep_colab.ipynb](notebooks/pdeep_colab.ipynb) contains setup, editable settings, generation/inspection, four visible training commands, CPU baselines, four evaluations, comparisons and optional multi-resolution/MAP cells. Select **Runtime → Change runtime type → GPU**. It checks CUDA and uses `--device cuda` for neural training, reusing Colab's installed compatible PyTorch. SciPy FEM generation and classical baselines stay on CPU.

Drive mounting, saving and restoring are optional. Generate/train on Colab's local disk, then copy chosen data/run directories to `MyDrive/PDEep-simple/`. Local runtime files can disappear at session end; copy important `last.pt`/`best.pt` checkpoints to Drive. Restore files locally before using the normal `--resume` command. There is no automatic cloud synchronization.

## Short technical notes

- **Solvers:** sparse P1 FEM with consistent mass/forcing integration; nodal coefficient averages in stiffness and true cellwise interface coefficients. Heat/reaction–diffusion use implicit Euler; wave/damped wave use average-acceleration Newmark and retain u/v. Configured forcing may be `f(x)*cos(2*pi*frequency*t)`. Dynamic times include 0 and 1.
- **Nonlinear limits:** semilinear cubic reaction is mass-lumped and solved by damped Newton. The p-Laplacian uses `(epsilon^2+|grad u|^2)^((p-2)/2)` with configurable epsilon (default 1e-4); p=2 is exactly Poisson. Eikonal is boundary-segment distance divided by **constant** positive speed. General varying-speed Eikonal, Monge–Ampère, nonlinear heat, capillarity and nonzero/Neumann boundaries remain unsupported.
- **Coefficients/geometry:** configured smooth coefficients use `a0*(1+contrast*tanh(g))`, with contrast < .95 for positivity; the original unconfigured semilinear generator retains its bounded logistic field. Interfaces are two-valued and mesh-aligned at the mid-x threshold. ADR uses constant advection without stabilization; choose diffusion-dominated regimes. Curved boundaries are polygonal approximations; hole centers and crescent construction are fixed. Resolve narrow features adequately.
- **Conditioning:** known values and masks are separate, including known zero. Forward gives full physical inputs; Inverse hides the tier's unknowns; Hybrid reveals configured samples. Known basis definitions do not reveal amplitudes. For legacy four-basis Poisson, full-rank Hybrid source rows determine the entire source: nearly zero prior/Bayes/MAP error is legitimate. This need not remain true for larger K or fewer revealed nodes.
- **Models:** shared mesh/PDE/observation encoders feed A's attention-memory decoder or C's Fourier-coordinate FiLM field. Cases have variable node/basis counts and accumulate gradients sequentially within batches. Optional physics losses are soft constraints; unknown forcing/operator diagnostics are omitted rather than evaluated with hidden labels.
- **Reference accuracy:** fields are numerical solutions, not random heatmaps. Queries use actual-triangle P1 interpolation and piecewise-linear physical-time interpolation; stationary time is zero. Coarse stored frames can smooth fast waves and make finite-difference residuals misleading. Increase spatial/time resolution and check convergence for a new regime. A denser decoder grid alone establishes no discretization-invariance claim.
- **Scientific scope:** deterministic reconstructions, learned flow samples, exact discrete Gaussian Bayes, assumed-kernel GP, training-derived POD and local numerical MAP are different estimators with different information. None of the small example runs establishes useful reconstruction quality or calibrated uncertainty. Existing numerical checks and earlier smoke/audit evidence remain in [VALIDATION.md](VALIDATION.md).
