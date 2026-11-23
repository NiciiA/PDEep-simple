# PDEep/simple

A small research codebase for numerical PDE data, deterministic and conditional latent-flow A-Geo/C-Geo reconstruction, and mathematical baselines. There was no existing PDEep checkout in this workspace to reuse. Everything lives here; no files outside `simple/` are needed.

## Start here

Use Python 3.9+ with NumPy, SciPy, PyTorch and Matplotlib. From this directory:

```bash
python -m pip install -e .
python -m pdeep_simple --help
python -m pdeep_simple check
python -m unittest discover -s tests -v
```

Alternatively, install the four scientific dependencies (plus `tomli` for TOML on Python 3.9/3.10) in your environment and run from this directory without installing the package. The smoke runs used `/Users/nicolasackermann/anaconda3/envs/my_env/bin/python` (Python 3.9, NumPy 1.26.4, SciPy 1.12.0, PyTorch 2.2.2, Matplotlib 3.8.2). The system `python3` did not have those dependencies. CPU was available; CUDA/MPS were not available in this execution session. If a sandbox makes the usual font cache unwritable, set `MPLCONFIGDIR=/tmp/pdeep_mpl XDG_CACHE_HOME=/tmp/pdeep_cache`.

## Complete G1 workflow

Run these commands from `simple/` using the Python environment above. Each command has `--help`.

```bash
python -m pdeep_simple data --tier G1 --pdes poisson,heat,wave \
  --geometries square,disk,lshape --train 8 --val 2 --test 2 \
  --workers auto --seed 42 --out data/g1_small

python -m pdeep_simple inspect --data data/g1_small --split train \
  --case 0 --out runs/inspection
# Or select several cases reproducibly:
python -m pdeep_simple inspect --data data/g1_small --split train \
  --random 6 --seed 9 --out runs/inspection

python -m pdeep_simple check

python -m pdeep_simple train --data data/g1_small --model A --size s \
  --modes forward,inverse,hybrid --loss field=1,aux=.1,kinematic=.001 \
  --updates 30 --batch 2 --queries 256 --device auto --run runs/a_smoke

python -m pdeep_simple train --data data/g1_small --model C --size s \
  --modes forward,inverse,hybrid --loss field=1,aux=.1 \
  --updates 30 --batch 2 --queries 256 --device auto --run runs/c_smoke

python -m pdeep_simple baseline --data data/g1_small --split test \
  --observations 0,8,16,32,64,128 --out runs/g1_baselines

python -m pdeep_simple eval --data data/g1_small --run runs/a_smoke \
  --split test --observations 0,8,16,32,64,128 --out runs/a_smoke/evaluation
python -m pdeep_simple eval --data data/g1_small --run runs/c_smoke \
  --split test --observations 0,8,16,32,64,128 --out runs/c_smoke/evaluation

python -m pdeep_simple plot --run runs/a_smoke --what losses,fields,comparisons \
  --compare runs/c_smoke runs/g1_baselines
python -m pdeep_simple plot --run runs/c_smoke --what losses,fields,comparisons \
  --compare runs/a_smoke runs/g1_baselines
```

The generation counts are **per PDE/geometry combination**: 72 train + 18 validation + 18 test = 108 independent physical cases. Geometry sizes and physical inputs vary independently between cases. An existing run requires `--resume`; use a new run name for another experiment. Existing datasets with different configurations are rejected. Rerunning the identical data command reuses completed case files and retries missing files. Solves are saved incrementally; failures are reported and never saved as valid cases.

Inspection writes mesh/boundary plots, source/initial/coefficient fields, displacement and velocity snapshots at actual times, observation overlays, histograms, and `summary.json`. Its dataset summary covers all splits; `--split` chooses the cases displayed individually.

## Other tiers: implemented combinations only

| Tier | PDE | Geometry families | Unknowns in inverse mode |
|---|---|---|---|
| T1 | Poisson | Fixed unit square | Source amplitudes (four fixed bases by default) |
| G1 | Poisson, diffusion, elliptic ADR, heat, reaction–diffusion, wave, damped wave | All supported families | Source and/or initial fields; operator known |
| T2 | Same linear families as G1, with constant coefficient | All supported families | Positive scalar conductivity, diffusivity or wave speed |
| T3 | Heat; variable-coefficient diffusion | All supported families | High-dimensional initial field; diffusion coefficient field |
| T4 | Semilinear elliptic, p-Laplacian, constant-speed Eikonal | All supported families | Source field; Eikonal speed |

All these combinations use the same forward/inverse/hybrid and training/evaluation interfaces. Optional losses and mathematical baselines apply only where supported. Other tier/PDE combinations fail explicitly. Without a config or an explicit new name, the original PDE selections, geometry selections and generation defaults are retained.

```bash
python -m pdeep_simple data --tier T1 --pdes poisson --geometries square \
  --train 2 --val 1 --test 1 --resolution 8 --frames 11 --workers 1 --out data/t1_small
python -m pdeep_simple data --tier T2 --pdes poisson,heat --geometries square,disk,lshape \
  --train 2 --val 1 --test 1 --resolution 8 --frames 11 --workers 1 --out data/t2_small
python -m pdeep_simple data --tier T3 --pdes heat --geometries square,disk,lshape \
  --train 2 --val 1 --test 1 --resolution 8 --frames 11 --workers 1 --out data/t3_small
python -m pdeep_simple data --tier T4 --pdes semilinear --geometries square,disk,lshape \
  --train 2 --val 1 --test 1 --resolution 8 --workers 2 --out data/t4_nonlinear

python -m pdeep_simple inspect --data data/t4_nonlinear --case 1 --out runs/t4_inspection
python -m pdeep_simple train --data data/t2_small --model C --size s \
  --loss field=1,aux=.1,weak=.001,initial=.001,boundary=.001 \
  --updates 30 --batch 2 --queries 256 --run runs/t2_experiment
python -m pdeep_simple baseline --data data/t1_small --modes inverse \
  --observations 0,8,32 --out runs/t1_baselines
python -m pdeep_simple baseline --data data/t2_small --modes inverse,hybrid \
  --observations 8 --frames 3 --out runs/t2_baselines
```

`data/t4_small`, if present from this implementation session, is an earlier, milder nonlinear smoke dataset (generator version 0). It is retained. Use `data/t4_nonlinear` for the final T4 generator; its larger forcing makes the nonlinear reaction more visible.

## Conditioning and leakage

`data.known_inputs` is the single definition of available information. Values always have separate masks. Known zero is different from unknown. Geometry, homogeneous zero Dirichlet conditions, PDE structure, and the sampled source/initial basis definitions are known in every mode. Amplitudes, hidden fields, target solutions, and hidden scalar labels never enter the encoders. Basis functions are embedded without their amplitudes; variable vertex and basis counts are supported.

- **Forward:** all physical inputs are known. Observations may be absent.
- **G1/T1/T3/T4 inverse:** forcing or initial fields are hidden; coefficients remain known. Wave hides both initial displacement and velocity. Configured dynamic forcing is also hidden. T3 diffusion instead hides the conductivity field; Eikonal hides constant speed.
- **G1/T1/T3/T4 hybrid (original defaults):** approximately half the nodal values of those fields are known, with masks. For low-rank priors this can determine the remaining values; it is not necessarily a difficult inverse task.
- **T2 inverse:** scalar coefficient hidden; source and initial state known. The Poisson source always remains known.
- **T2 hybrid:** coefficient hidden and, for heat, half the initial field known. Poisson has the same known physical inputs as its inverse mode.

For the original defaults, heat diffusivity is uniform on [0.03, 0.2]; Poisson T2 conductivity on [0.5, 2]; wave speed on [0.5, 1.2]. T4 has smooth conductivity in (0.5, 2), beta in [0.5, 3], and Gaussian source amplitudes with standard deviation 12. T1 has fixed localized, boundary-tapered, domain-RMS-normalized source bases and independent N(0, 9) amplitudes. T3 uses 32 smooth localized functions with varied widths and damped amplitudes, rather than vertex white noise.

Observation noise is independent Gaussian with **absolute** standard deviation `--noise` (default 0.01), including velocity observations. Locations are area-uniform within triangles; time is uniform on [0,1]. Training samples fresh observations/queries and nonzero counts for inverse/hybrid. Evaluation count zero deliberately tests the unobserved prior. Evaluation draws depend only on case seed, evaluation seed and count, so A, C and baselines receive the same measurements and query mesh.

Normalization uses fixed training-distribution RMS scales, separately for displacement/temperature and wave velocity, with a 0.01 floor. No held-out target is used to scale model inputs. Training area-uniform queries estimate a domain integral without extra quadrature weights. Evaluation uses lumped P1 spatial weights and trapezoidal weights over selected physical times. It reports raw physical-unit errors and normalized RMSE.

## Numerical representation

`geometry.py` retains structured rectangle/L meshes and concentric-ring disk/ellipse meshes. Other shapes use polygon boundaries, segment recovery by bisection in a Delaunay mesh, and domain filtering. Hole boundaries are preserved, rather than bridged by triangles. The L reentrant corner is a mesh vertex; no elements cross the excluded quadrant. Connectivity and topological boundary edges/nodes are stored. Coordinates are physical, not image pixels.

`physics.py` assembles sparse consistent P1 mass and stiffness matrices. A nodal P1 coefficient is integrated exactly in the stiffness term via its element average. Nodal forcing is integrated with the consistent mass matrix. Boundary degrees of freedom are eliminated.

- Poisson: sparse linear solve.
- Heat: implicit Euler, unforced in generated datasets.
- Wave: average-acceleration Newmark, including nonzero initial velocity and the consistent initial acceleration. Both u and v are retained.
- T4: Newton with residual/Jacobian for `K(a)u + beta * lump(M) * u^3 = M f`. The cubic reaction is mass-lumped in both solver and weak loss.

Internal time steps are at most `--dt` (default 0.002), subdividing each stored frame interval exactly. Stored dynamic times include 0 and 1. Ground truth is always a numerical solve; random functions are physical inputs. Mesh/time accuracy still needs refinement studies for a new experiment. The default analytical checks and a 12-case time-step comparison are recorded in `VALIDATION.md`.

Files contain float64 geometry/times, float32 physical fields/solutions, integer connectivity, and plain JSON metadata. `geometry.interpolate` uses the actual triangles with barycentric interpolation, then piecewise linear physical-time interpolation. It recovers stored frames exactly and rejects points outside the mesh (apart from a tiny barycentric roundoff tolerance). Stationary queries require t=0. Stored time interpolation can smooth rapidly oscillating waves: increase `--frames` and reduce `--dt` when needed.

## Models and losses

`models.py` shares two mesh message-passing layers, relative edge coordinates, boundary indicators, known values/masks, typed fixed PDE operator graphs, and a padded observation-set encoder. An always-valid null token handles empty observation sets. Cases with different meshes/counts are processed sequentially within a minibatch and their gradients accumulated.

A-Geo constructs learned global latent memory tokens from mesh/PDE/observation tokens. Continuous coordinate queries use local interpolated mesh features and spatial/time-local observation summaries, then cross-attend to the full local/global memory. C-Geo uses the shared physical encoders and the same local information to modulate a Fourier-coordinate residual field using FiLM; it has no query cross-attention decoder. Both predict two channels, using the second only for wave, and have a positive scalar auxiliary head `exp(log_coefficient)` for T2.

Deterministic trainable counts (default presets; `check` counts large models on the meta device):

| Preset | A-Geo | C-Geo |
|---|---:|---:|
| s | 1,500,035 | 1,312,387 |
| m | 6,478,339 | 5,739,267 |
| l | 27,908,611 | 27,082,243 |
| xl | 62,715,651 | 56,150,787 |
| xxl | 145,024,003 | 133,350,403 |

Presets are ordinary dictionaries. L uses eight attention decoder blocks for A and nine FiLM blocks for C to meet the parameter targets. Override `--width`, `--layers`, `--heads`, `--memory` as desired, including XXL. C does not use latent memory tokens, so `--memory` only affects A. No large-preset training was performed.

Losses: `field`, `aux` (T2 log coefficient), wave-only `kinematic` (autograd du/dt-v), `weak`, `initial`, `boundary`. Every weight is independently selectable. Inapplicable mixed-PDE terms are logged and omitted; globally incompatible requests such as kinematic loss on Poisson-only data are rejected.

Weak loss evaluates the **P1 projection** of nodal neural predictions against mesh test functions. Heat uses centered du/dt, wave centered d²u/dt², with physical-time spacing 0.01; spatial terms use the actual FEM matrices. This is a consistent finite-dimensional weak diagnostic, not a strong-form derivative test of the entire neural field. Hidden T2 coefficients use the auxiliary prediction; hidden forcing has no auxiliary head, so those cases omit weak loss. Initial penalties only use known nodes; boundary values are known zero. Derivatives of interpolated local mesh features are not used as strong spatial PDE derivatives. The field is not hard constrained to satisfy initial/boundary conditions.

Easy comparisons:

```text
A base / C FIELD_ONLY: field=1,aux=.1
A KIN_001:            field=1,aux=.1,kinematic=.001
C ALL_001 (G1):       field=1,aux=.1,weak=.001,initial=.001,boundary=.001
```

## Baselines and evaluation

`baselines.py` supplies:

- Prior conditional mean: condition the known Gaussian basis prior on any exact known field nodes, without using test observations; solve with its mean for linear equations. T2 averages actual numerical forward solutions over 24 uniform-prior coefficient quadrature points. Fully hidden T4 forcing has zero solution mean by odd symmetry. A partially known T4 forcing must determine all its amplitudes; an underdetermined nonlinear conditional prior is explicitly unsupported.
- T1 ridge and T1/G1 linear elliptic `bayes`: solve numerical basis responses and use the same barycentric observation map/noise/prior as generation. Bayes saves coefficient covariance and propagated pointwise solution standard deviations. The Gaussian posterior is exact for this discrete linear-Gaussian model, up to stored float32 rounding.
- RBF observation interpolation in physical space/time, separately for u/v. With too few observations it uses their mean; with none, zero. It need not satisfy the PDE.
- T2 scalar grid: 41 actual forward solves over the bounded uniform prior, likelihood-weighted predictions and coefficient estimates. Heat hybrid with a partly hidden initial state is skipped with a reason in `skipped.csv`; the scalar grid does not solve a function-valued inverse problem.

For **G1 Poisson**, `prior` is the **physical-input-conditioned prior mean**, without solution observations: `E[u | known physics]`. It is not an unconditional zero field in every mode or the observation-conditioned `bayes` estimator.

| Mode | Known physical inputs for A/C and `prior` | `prior` prediction |
|---|---|---|
| Forward | Full source, a=1, domain/mesh, zero Dirichlet boundary and basis definitions | Fresh FEM solve; nearly zero error is expected |
| Inverse | Source hidden; other physics and basis/prior known | Zero Gaussian mean source gives zero mean solution |
| Hybrid | Roughly half the exact source nodal values with masks; other physics known | Condition the source prior on those values, then solve with its mean source |

Hybrid does **not** supply the full source directly. However, `f = B b` has only four unknown amplitudes. If the known rows of B have rank four, they determine the whole source, explaining nearly zero Hybrid error even without u sensors. No hidden amplitude labels or reference solution are used. Rank-deficient constraints need not recover the true source.

A/C use both known physical inputs and selected noisy solution observations. The prior ignores solution observations. **RBF uses only those observations**, ignoring source, conductivity and boundary constraints; its smoothing is `max(noise_std**2, 1e-7)` for the current uniform-noise generator. Thus RBF is identical across modes for the same observation packet, while prior predictions do not change with sensor count. These are comparisons of the stated estimators, not of identical use of physical information.

For the original four-basis Poisson datasets, T1/G1 `bayes` uses known physics, available source information **and noisy solution observations**. The generator draws four independent amplitudes from N(0, 3²), so the actual prior is mean zero and covariance **9 I**, without fitting any amplitude labels or approximating the generating law. Four FEM basis responses are reused across modes/counts; H evaluates them at the supplied sensors and uses each supplied noise standard deviation. Cholesky solves update the remaining amplitude directions. Forward recovers amplitudes from the full source; Inverse uses the prior plus observations (zero mean at zero sensors); Hybrid conditions on exact known source rows through an SVD nullspace before using observations. Full-rank source constraints determine the solution and give zero posterior covariance; incomplete rank retains uncertainty. Only the K×K amplitude covariance and pointwise solution standard deviations are stored; K=4 for the original datasets. Configured Gaussian priors use their actual K, mean and variance, including count scaling. Known-operator diffusion and elliptic ADR are also supported. Non-Gaussian Bayes is explicitly unsupported. Heat/wave and T2/T3/T4 Bayes are explicitly unsupported. T1 ridge shares the response construction. T2 `grid` evaluates candidate coefficients without reading the hidden true coefficient.

For the existing G1 Poisson checkpoints, run from `simple/`:

```bash
python -m pdeep_simple baseline --data data/g1_test --pde poisson \
  --methods prior,rbf,bayes --workers auto --normalization-run runs/a_test3k \
  --observations 0,8,16,32,64,128 --out runs/g1_bayes
python -m pdeep_simple plot --runs runs/a_test3k runs/c_test3k runs/g1_bayes \
  --what comparisons --metric normalized_rmse --out runs/g1_bayes_comparison
```

The default modes include Forward, Inverse and Hybrid. These two checkpoints share the same training normalization; `--normalization-run` applies it to all three baselines. Add `--max-cases 2` to the baseline command for a quick check; plots then use only those shared cases. `plot --run runs/g1_bayes --what uncertainty` displays saved Bayesian posterior standard deviations alongside the reference, mean and error. To select an Inverse case directly, use `plot --runs runs/g1_bayes --data data/g1_test --pde poisson --geometry disk --modes inverse --what uncertainty`. Existing latent-flow uncertainty plots retain their sample-based interpretation.

Metrics use domain/time weights. `normalized_rmse = RMSE/s_u`, using a fixed training RMS; the CSV's `mse` is raw, and `normalized_mse = MSE/s_u**2`. Relative L2 instead divides by reference energy. Aggregates are means of per-case errors, not pooled RMSE. Use matching scales or `--normalization-run` for normalized comparisons. Error plots default to physical RMSE; `--metric normalized_rmse` selects normalized sensor/geometry/time curves with distinct filenames. The PDE panel retains relative L2. Sensor sets match across methods at each count, but are not nested across counts.

The focused audit of `a_test3k`, `c_test3k` and `g1_base_test3k` found no leakage or mismatched observations, references or normalization. Their existing results remain valid; no baseline rerun or plot regeneration is required. Detailed evidence is in `VALIDATION.md` and `runs/baseline_audit.json`.

Evaluation saves `metrics.csv`, grouped `summary.csv`, draw settings and selected prediction NPZs. Groups distinguish tier, PDE, geometry, conditioning mode, count, method, quantity, time and geometry generalization. Metrics include MSE/RMSE/MAE/relative L2/max error, normalized RMSE, boundary/initial errors, T2 coefficient errors, P1 residual diagnostics, wave kinematics and complete per-case inference time (including encoding/query preparation/transfers).

By default five selected stored times are evaluated; set `--frames 21` (or the dataset's full frame count) for denser trajectory quadrature and temporal diagnostics. Finite-difference residuals on sparse time frames can be inaccurate and should not be interpreted as a strong physics guarantee. Hidden forcing or an unavailable coefficient produces NaN residuals, never a residual using a hidden oracle input. Relative error near a zero reference field can be large; use physical RMSE alongside it. Plotting keeps u/v and conditioning modes separate, with shared truth/prediction color scales. Cross-method plots use the intersection of available physical cases and observation counts, so an unsupported baseline combination cannot change the compared case populations; time curves and coefficient scatter distinguish PDEs. White circles in dynamic reconstruction panels show observations within 0.16 physical-time units of that frame; stationary panels show all observations.

New physical cases and new geometry instances are the default test split. For an actual family holdout, exclude it from **both training and validation** via `--geometries`:

```bash
python -m pdeep_simple train --data data/g1_small --model C --geometries square,disk \
  --updates 30 --run runs/family_holdout
python -m pdeep_simple eval --data data/g1_small --run runs/family_holdout \
  --geometries lshape --out runs/family_holdout/evaluation
python -m pdeep_simple baseline --data data/g1_small --geometries lshape \
  --normalization-run runs/family_holdout --out runs/family_holdout_baselines
```

Use the same normalization run for fair normalized baseline comparisons in this restricted-training experiment. The CSV marks `unseen_family`. To test entirely new cases/resolutions, generate another dataset and pass it to `eval`; the saved model's training scales remain fixed. This interface does not establish discretization invariance or unseen-PDE generalization.

## Resume and files

`--updates` means **total desired updates**, not additional updates:

```bash
python -m pdeep_simple train --data data/g1_small --model A \
  --resume runs/a_smoke/last.pt --updates 60 --run runs/a_smoke
```

Resume restores saved scientific settings, optimizer, update count and RNG; `--data` honors the supplied path so a dataset can move machines. `--updates`, device and run destination may change. If the destination has no best checkpoint, best selection starts with the resumed updates. Use the same run directory to retain its complete CSV/history and best checkpoint. Training writes `config.json`, `train.csv`, `validation.csv`, `last.pt`, `best.pt`; latest checkpoints are written at validation intervals and the final update. Deterministic best selection uses validation field loss; flow stages are described below. Evaluation defaults to `best.pt`. Ordinary `map_location` permits changing devices.

The package is about 2,700 lines of ordinary Python:

```text
geometry.py   meshes, barycentric queries, area weights
physics.py    sparse FEM, implicit dynamics, nonlinear solves and boundary distance
data.py      generation/storage, masks, observations and model/query inputs
models.py     shared encoders, A-Geo attention and C-Geo FiLM
losses.py     individually selectable objectives
training.py   AdamW minibatches, fixed validation, resume/checkpoints
baselines.py  conditional priors, T1/G1 elliptic Gaussian, T1 ridge, RBF and T2 grid
evaluation.py  fixed held-out metrics and CSVs
plotting.py   inspection, losses, reconstructions and comparisons
flow.py       field autoencoder, latent conditioning and Heun flow sampling
comparison.py same-case field/uncertainty figures and separate stage loss curves
checks.py     small numerical/model checks
cli.py        seven argparse commands
```

No other nonlinear PDEs, nonzero/Neumann boundary conditions, arbitrary domain meshing, general symbolic parser, spatially unknown T4 conductivity, or auxiliary hidden-field prediction head is implemented. Larger meshes make A's attention and the small dense matrices used by the neural weak loss expensive; numerical solvers themselves remain sparse. This is a readable starting point, not a scalable FEM library or a trained scientific result. See `VALIDATION.md` for exactly what was executed.

## Parallel classical baselines

The same baseline command now accepts CPU workers. Cases are independent process tasks; matrix operations inside each worker use one BLAS/OpenMP thread. `auto` uses at most four workers and half the CPU count, further bounded by estimated case memory, `--memory-gb` (default 2 GiB total), and half the OS estimate of available/reclaimable RAM. These are conservative estimates, not a memory guarantee; reduce workers or raise the explicit budget for a larger experiment.

```bash
python -m pdeep_simple baseline --data data/g1_small --workers auto \
  --observations 0,8,32,128 --out runs/g1_parallel
python -m pdeep_simple baseline --data data/t1_small --workers 2 \
  --modes inverse --observations 0,8,32 --out runs/t1_parallel
# --workers 1 is the default and uses the same process/thread limits.
```

Sparse operators and factorizations are cached by the actual mesh/coefficient, with at most four operators per worker. T1/G1 elliptic basis responses and T2 candidate fields are reused within a case. Inference is prepared once for each conditioning set and queried on all requested grids in chunks. Training-distribution normalization is streamed case by case. Progress prints cases/s and ETA. Each completed case writes its own CSV plus appended `metrics.csv`; prediction NPZs are saved as they finish. The final table is put in deterministic case order. An interrupted evaluation retains completed files; there is no automatic baseline-resume system. Use a fresh output directory when changing settings.

The prior, RBF, ridge, Bayesian posterior and T2 scalar-grid methods are retained. T1/G1 elliptic Bayes is exact only for the matched stored linear-Gaussian observation model, up to storage rounding. No Gaussian posterior formula is applied to T3/T4 nonlinear or non-Gaussian problems. Different worker counts use identical physical cases, masks, observations and seeds; only timing/progress order can differ.

## Conditional latent flow matching

Add `--flow` to either architecture. Without it, deterministic models and old checkpoints retain their existing architecture. Size presets and manual overrides still work; construction prints the total count including the small field encoder and flow MLP.

```bash
python -m pdeep_simple train --data data/g1_small --model A --size s --flow \
  --latent-dim 32 --ae-updates 1000 --updates 3000 --batch 4 --queries 256 \
  --modes forward,inverse,hybrid --run runs/a_flow
python -m pdeep_simple train --data data/g1_small --model C --size s --flow \
  --latent-dim 32 --ae-updates 1000 --updates 3000 --batch 4 --queries 256 \
  --modes forward,inverse,hybrid --run runs/c_flow
# A genuinely short execution check instead of the longer examples above:
python -m pdeep_simple train --data data/t1_small --model A --size s --flow \
  --ae-updates 2 --updates 4 --batch 2 --queries 32 --latent-points 64 \
  --observations 0,8 --flow-steps 8 --run runs/a_flow_smoke
```

`--updates` is the **total across both stages**. `--ae-updates` selects the first stage; its default is `min(500, max(1, updates // 3))`. Both stages need at least one update. Start a new flow run; resume restores the saved architecture and stage rather than converting a deterministic checkpoint.

1. **Field representation:** a small set encoder receives area-uniform reference field samples with physical coordinates/Fourier features, including both u and v for wave. Dynamic cases include samples at physical t=0. There is no fixed image grid or fixed vertex count. A compact latent (default 32) modulates the existing A attention memory or C FiLM decoder and a spatial latent-dependent residual. The usual field/auxiliary and applicable physics losses train this representation. `--latent-points` defaults to 256.
2. **Flow:** freeze the field encoder, A/C conditioning encoder and coordinate decoder. Standardize latent targets using training-only means/scales, then train a small conditional velocity MLP. It minimizes `||v(z_s,s,c) - (z1-z0)||²` for `z_s=(1-s)z0+s*z1`, `z0~N(0,I)`, and uniform **flow time s**. Physical PDE time remains the third query coordinate. Optional PDE losses belong to the representation stage; they are not silently added to the flow objective.

The reference field encoder is used only for AE training/validation and supervised latent targets. Generation uses known physical inputs/masks and observations, draws a latent, and integrates the learned ODE with Heun's method. It never reads an unknown reference solution or solves the PDE. One latent represents the **whole space-time field**, including both wave quantities, and is reused across chunks and resolutions. The explicit spatial latent path and small checks exercise nonzero latent dependence; a long trained model can still underuse its latent, so inspect reconstruction and sample variation.

```bash
python -m pdeep_simple eval --data data/g1_small --run runs/a_flow \
  --samples 32 --flow-steps 32 --modes forward,inverse,hybrid \
  --observations 0,8,32,128 --max-cases 3 --out runs/a_flow/evaluation
python -m pdeep_simple train --data data/g1_small --model A --flow \
  --resume runs/a_flow/last.pt --updates 4000 --run runs/a_flow
```

Inference automatically recognizes a flow checkpoint. `--samples` is the number of complete fields; `--flow-steps` controls Heun integration. Means, unbiased sample standard deviations (`ddof=1`), central empirical intervals (`--interval .9`) and domain-weighted interval coverage/width are recorded. With one sample, standard deviation and intervals are undefined and saved as NaN. `--save-samples 4` retains only the first four complete sample fields while statistics use every sample. Latent endpoints are also saved, in the unstandardized AE latent space. Sampling processes coordinates in chunks, not a full samples-by-512² GPU tensor.

This is a learned conditional distribution, **not automatically calibrated Bayesian uncertainty**. Forward variation should become small when the fully specified problem determines the field, but no artificial variance suppression enforces that. Tiny smoke runs do not demonstrate useful reconstruction or uncertainty. A flow covers the same compatible tiers/PDEs as its field decoder, including the nonlinear T4 problem; T2's latent-conditioned auxiliary coefficient head remains positive.

Checkpoints contain both learned components, latent normalization, stage, optimizer and RNG state. `last.pt` can resume either stage. After the first flow update, `best.pt` is selected by fixed validation **flow-matching loss**, not AE reconstruction MSE or test error. A checkpoint still in the AE stage is rejected by sampling evaluation. Extending a completed flow run adds flow updates without retraining its frozen representation.

## Multi-resolution queries and optional refined references

Both `eval` and `baseline` accept `--grids`. A grid covers the original mesh bounding box; triangle connectivity masks the exterior and holes. Only valid-domain coordinates enter errors. Tensor-product trapezoidal grid weights approximate the domain integral; FEM-node evaluation without `--grids` retains lumped P1 weights. Time quadrature is unchanged. The CSV includes grid size, valid-point count, estimated area, noise and evaluated physical times. Summary/plot groups keep different grid/reference levels separate.

```bash
python -m pdeep_simple eval --data data/g1_small --run runs/a_smoke \
  --grids 128,256,512 --max-cases 2 --chunk 1024 \
  --modes inverse --observations 8,32 --out runs/a_multigrid
python -m pdeep_simple eval --data data/g1_small --run runs/a_flow \
  --grids 128,256,512 --max-cases 2 --chunk 1024 --samples 32 --flow-steps 32 \
  --modes inverse --observations 8,32 --out runs/a_flow_multigrid
python -m pdeep_simple baseline --data data/t1_small --workers auto \
  --methods prior,bayes,ridge,rbf --grids 128,256,512 --max-cases 2 \
  --modes inverse --observations 8,32 --out runs/t1_multigrid
```

A grid evaluation defaults to **at most eight physical cases** unless `--max-cases` is explicitly supplied. Existing no-grid commands still evaluate their entire selected split. `--pde heat` and `--geometries disk` can narrow selection before the limit. Saved NPZs contain the mask, grid coordinates, reference/prediction fields, observations and exact known-field masks. Grid density changes the query/quadrature resolution; **it does not improve the stored PDE reference**.

For a mesh/time comparison on selected cases only:

```bash
python -m pdeep_simple eval --data data/g1_small --run runs/a_smoke \
  --pde heat --geometries lshape --max-cases 1 --refine 2 \
  --grids 128,256 --modes inverse --observations 32 --out runs/a_refined
python -m pdeep_simple baseline --data data/t1_small --methods bayes --workers 1 \
  --max-cases 1 --refine 1 --grids 128 --modes inverse --observations 32 \
  --out runs/t1_refined
```

`--refine 1..3` requires an explicit case limit. Red refinement keeps exactly the original polygon and prolongs its stored source, initial and coefficient fields as the **same continuous P1 inputs**; it does not resample a new physical case or improve the polygonal disk boundary. Each level solves the PDE with a halved internal time step. `refinement_CASE.csv` compares consecutive reference levels; `reference_rmse` in the main metrics compares the stored and final references on the actual evaluation coordinates. Observations and conditioning stay on the original case, so refined truth cannot leak into inference. T1/G1 elliptic Bayes remains exact for its original discrete observation/solution model, not automatically for the new refined reference. These differences are convergence diagnostics, not certified reference accuracy.

## Comparing saved runs on the same case

The original `plot --run ... --what ... --compare ...` commands remain available. Use `--runs` plus `--data` to recompute predictions on **one identical physical case**, including identical observations, noise, masks, valid coordinates and physical times:

```bash
python -m pdeep_simple plot --runs runs/a_smoke runs/c_smoke runs/a_flow \
  --data data/g1_small --pde poisson --geometry disk --case 0 \
  --modes forward,inverse,hybrid --observations 32 --grid 256 \
  --samples 32 --flow-steps 32 --out runs/same_case_comparison

# A saved baseline directory can join the field comparison; select its methods:
python -m pdeep_simple plot --runs runs/a_flow runs/g1_parallel \
  --data data/g1_small --pde poisson --geometry disk --methods prior,rbf \
  --observations 32 --grid 128 --out runs/flow_classical_comparison

# Curves from saved evaluation tables, with no new inference:
python -m pdeep_simple plot --runs runs/a_smoke runs/c_smoke runs/a_flow runs/g1_parallel \
  --what losses,comparisons --out runs/curve_comparison
```

`--case` indexes cases matching split/PDE/geometry, with `--split test` by default. `--frames` selects time slices, and wave u/v figures are separate. Solution figures show reference, each run's prediction and absolute errors with shared scales. Conditioning figures compare modes and differences from the first selected mode. Flow figures show reference, mean, standard deviation, individual fields and absolute error of the mean. `--what fields` includes conditioning/uncertainty; these can also be requested individually as `conditioning` and `uncertainty`. White sensor circles use the same physical-time window as the existing plots.

Training curves have separate axes for deterministic reconstruction, AE reconstruction, latent flow matching and validation sample-mean reconstruction. Field-loss curves use each run's stored training RMS normalization; check that these scales match when interpreting cross-run loss differences. Sensor/geometry/time comparisons default to physical RMSE; choose `--metric normalized_rmse` when training scales match. FM values are never plotted as field MSE; inactive stage losses are blank in the CSV. Run names distinguish different checkpoints of the same architecture. Error-versus-sensors/PDE/geometry plots use common cases, seeds, noise, physical query times, grid and reference level; methods missing a combination are not compared to a different population. Single-run flow fields and uncertainty plots also work with the ordinary `plot --run runs/a_flow --what losses,fields,comparisons` command.

The flow extension adds only `flow.py` and `comparison.py`; it reuses the existing conditioning, solvers, losses, storage, metrics and plotting. No flow package or training framework is required. See the extension section of `VALIDATION.md` for the actual small CPU checks and their limits.

## Optional dataset configuration and additional equations

`data --config configs/dataset.toml` reads mathematical generation settings. The CLI still selects tier, PDEs, geometries, split sizes, workers, seed and destination. Without `--config`, the original commands keep their defaults. Explicit `--resolution`, `--frames` and `--dt` override config numerics. The resolved settings are saved in the manifest and cases; **train/eval do not need the TOML file**. Python 3.11+ uses `tomllib`; installation supplies the small `tomli` backport on 3.9/3.10.

In the config, `count = 16` is fixed; `count = [4, 32]` samples an integer including both endpoints. Other numeric pairs, such as `beta = [0.1, 2.0]`, sample uniformly. Gaussian and uniform amplitudes use the specified mean and standard deviation. `scale_by_count = true` divides amplitude std by sqrt(K), without normalizing each solution. Overlapping bases can still produce correlated fields. Widths are fractions of bounding-box dimensions; centers are sampled inside the actual meshed domain, including holes.

Each source, initial displacement/temperature, initial velocity and smooth coefficient field has its own saved basis values, centers, widths, amplitude labels and prior parameters. PDE-specific counts override `[basis].count`; they can vary independently between fields and cases. Models receive basis definitions, field roles and prior parameters through the existing encoders, never amplitude labels or reference values. The original encoder layer sizes and checkpoint keys are unchanged.

| PDE name | Equation / numerical method | Important config parameters | Tier |
|---|---|---|---|
| `poisson` | −Δu=f; sparse P1 FEM | `forcing_basis` | T1, G1; scalar-conductivity T2 convention retained |
| `diffusion` | −div(a∇u)=f; P1 FEM | `conductivity`, `coefficient_kind`, `coefficient_basis`, `coefficient_contrast`, `forcing_basis` | G1, T2 constant, T3 unknown field |
| `elliptic_adr` | −div(a∇u)+b·∇u+ru=f; Galerkin FEM | `diffusivity`, `advection`, `reaction`, `forcing_basis` | G1, T2 |
| `heat` | u_t−div(a∇u)=f; implicit Euler | `diffusivity`, `initial_basis`, `forcing_basis`, `forcing_frequency` | G1, T2, T3 |
| `reaction_diffusion` | u_t−div(a∇u)+ru=f; implicit Euler | Heat settings plus `reaction` | G1, T2 |
| `wave` | u_tt−div(c²∇u)=f; Newmark | `wave_speed`, `displacement_basis`, `velocity_basis`, `forcing_basis`, `forcing_frequency` | G1, T2 |
| `damped_wave` | u_tt+γu_t−div(c²∇u)=f; Newmark | Wave settings plus `damping` | G1, T2 |
| `semilinear` | −div(a∇u)+βu³=f; damped Newton | `beta`, `conductivity`, `forcing_basis` | T4 |
| `p_laplacian` | Regularized p-Laplacian; convex P1 energy and damped Newton | `p` ≥2, `forcing_basis`, numerical regularization/tolerance | T4 |
| `eikonal` | \|∇u\|=1/c; boundary-segment distance / constant c | `speed`; no forcing basis | T4 |

All equations currently use homogeneous Dirichlet boundaries. Dynamic times include 0 and 1. Forcing can be steady or `f(x)*cos(2*pi*forcing_frequency*t)`. Wave and damped wave retain separate displacement and velocity targets. Advection is a constant two-vector, sampled componentwise; use diffusion-dominated regimes, since there is no advection stabilization. Reaction and damping are nonnegative.

For linear operators and semilinear diffusion, `coefficient_kind` can be `constant`, `smooth` or `interface`. Smooth coefficients use `a0*(1+contrast*tanh(g(x)))`, with `0 <= contrast < .95`, so positivity is preserved. Interfaces use two positive mesh-cell values `a0*(1±contrast)` on either side of the domain's mid-x threshold. These are genuinely discontinuous, mesh-aligned FEM coefficients, stored and plotted per cell; they are not claimed to resolve an arbitrary analytic interface. T2 always uses a constant unknown scalar. T3 diffusion hides the coefficient field instead of the source.

The p-Laplacian solves `-div((epsilon²+|grad u|²)^((p-2)/2)*grad u)=f`, with configurable `numerics.p_regularization` (default 1e-4). This is an explicit regularization of the degenerate equation, with **p=2 exactly Poisson**. Newton convergence is recorded; failed solves are not saved. Eikonal supports positive constant speed only, on the polygonal mesh domain. Its stored P1 interpolant approximates the nonsmooth distance function. General varying-speed Eikonal, Monge–Ampère, nonlinear heat and capillarity remain unsupported because they require different numerical methods.

Geometry names are `square,rectangle,disk,ellipse,lshape,annulus,hole,polygon,triangle,trapezoid,star,cross,dumbbell,crescent,two_holes`. The example config controls rectangle aspect ratio, ellipse axis ratio, triangle apex, trapezoid top width, star point count/inner radius, polygon side count, cross arm width, dumbbell neck width and hole/annulus radii. Hole centers and the crescent construction are fixed. Curved boundaries are polygonal approximations. Avoid features much narrower than the mesh spacing; increase resolution to study thin regions.

`conditioning.hybrid_known_source_fraction`, `hybrid_known_initial_fraction` and `hybrid_known_coefficient_fraction` each accept fixed values or ranges in [0,1]. A range is sampled once per physical case; conditioning draws use that probability independently at each node. Existing solution-observation controls remain on train/eval/baseline. K independent exact source rows can still determine K amplitudes. Inverse hides configured dynamic forcing as well as initial fields. Eikonal Hybrid reveals speed samples; since speed is constant, any exact sample can determine it.

`bayes` supports arbitrary K and nonzero-mean Gaussian source priors for T1/G1 Poisson, diffusion and elliptic ADR with a fully known operator. It uses the actual per-case prior std, including count scaling, exact Hybrid constraints, and the existing observation packet. It stores K×K covariance and pointwise std. Non-Gaussian Bayes, dynamic Bayes and nonlinear Bayes remain unsupported. `prior` still ignores solution observations; `rbf` only uses them. Unsupported partial non-Gaussian/nonlinear prior expectations and unknown spatial-coefficient priors are recorded in `skipped.csv`, without substituting a different estimator. Weak losses require known source/operator information (or the T2 scalar head); Eikonal has no elliptic weak loss. Stationary cases do not produce physical-time error curves.

```bash
# Original defaults
python -m pdeep_simple data --tier G1 --pdes poisson \
  --geometries square,disk,lshape --train 128 --val 16 --test 16 \
  --workers auto --out data/g1_default

# Configured linear families
python -m pdeep_simple data --config configs/dataset.toml --tier G1 \
  --pdes poisson,diffusion,heat,wave --geometries square,disk,lshape,star \
  --train 128 --val 16 --test 16 --workers auto --out data/g1_configured

# Nonlinear families stay in T4
python -m pdeep_simple data --config configs/dataset.toml --tier T4 \
  --pdes semilinear,p_laplacian --geometries square,disk,lshape \
  --train 64 --val 16 --test 16 --workers auto --out data/t4_configured

python -m pdeep_simple inspect --data data/g1_configured --random 2 --out runs/configured_inspection
python -m pdeep_simple train --data data/g1_configured --model A --size m \
  --modes forward,inverse,hybrid --loss field=1 --updates 3000 --batch 4 \
  --queries 4096 --device auto --run runs/a_configured
# The same command supports --model C or --flow.
python -m pdeep_simple baseline --data data/g1_configured --pde diffusion \
  --methods prior,rbf,bayes --observations 0,8,16,32,64,128 --workers auto \
  --normalization-run runs/a_configured --out runs/configured_baselines
python -m pdeep_simple eval --data data/g1_configured --run runs/a_configured \
  --grids 128,256,512 --max-cases 2 --out runs/a_configured/evaluation
```

Changed files are `data.py`, `geometry.py`, `physics.py`, `models.py`, `losses.py`, `baselines.py`, `cli.py`, `evaluation.py`, `training.py`, `plotting.py`, `comparison.py`, `tests/test_science.py`, `pyproject.toml` and this README. The only new project file is `configs/dataset.toml`; `pyproject.toml` adds the TOML backport for older Python. Small checks covered all PDE names and mesh families, reproducibility, boundaries, basis counts, p=2, model gradients, multiprocessing and ordinary loading/plotting. Only one/two-update training smoke runs were used; they do not establish useful learned accuracy.

## POD, spatial GP and parameter MAP baselines

The existing `baseline --methods` also accepts `pod,gp,map`. Defaults and the meanings of `prior`, `rbf` and `bayes` are retained. Unsupported method/case/mode combinations are listed in `skipped.csv`; other methods continue. All methods receive the same selected observations, masks, physical query coordinates and metric normalization.

| Method | Representation and inference | Information used / supported cases |
|---|---|---|
| `pod` | Centered training snapshots, lumped-P1-mass-weighted SVD; noise-weighted ridge for modal coefficients | Training solutions plus supplied observations; stationary scalar fields on a compatible physical domain/PDE group |
| `gp` | Zero-mean squared-exponential spatial kernel; Cholesky posterior mean and pointwise std | Supplied observations and declared noise; stationary scalar fields on every meshed geometry |
| `map` | L-BFGS-B in a small physical parameter space, repeatedly calling the existing FEM solver | Known physics, actual parameter priors, exact revealed field constraints and supplied observations; supported cases below |

**POD** fits only the manifest's **training split**, even when evaluating validation cases. It uses the first training mesh in each group as the common physical discretization, interpolating other matching-domain snapshots with the existing P1 interpolation. Collinear boundary subdivisions are ignored when comparing domains; genuinely different polygon boundaries, sizes or shapes are separate groups. At least two training snapshots are required. **The current randomly resized G1 domains usually have no matching training/evaluation group**, so POD is reported unsupported there. Fixed-domain T1 works directly. No bounding-box rescaling or flattening of unrelated meshes is applied.

`--pod-rank 32` limits the retained nonzero modes (also bounded by training rank). `--pod-ridge 1.0` sets lambda in `sum(((A*z-b)/noise_std)^2) + lambda*sum(z^2)`. Zero observations return the compatible group's training mean. The representation is saved once in `OUT/pod.npz`; use `--pod-model PATH.npz` to create or reuse it across evaluations, keeping the same rank and training split. POD field prediction uses only observations, ignoring additional Forward/Hybrid physical inputs; its reduced basis does not automatically enforce the PDE. Choose rank/ridge using validation performance, never test references.

**GP** uses `k(x,x') = amplitude^2 * exp(-|x-x'|^2/(2*lengthscale^2))`, with `--gp-lengthscale 0.2` in physical coordinate units and `--gp-amplitude 1.0` as the prior field standard deviation. These defaults are assumptions, not fitted to held-out data. Zero observations give zero mean and the prior standard deviation. The supplied, possibly unequal observation variances form R. One small observation-covariance factorization is reused across query chunks and grids; only the posterior covariance diagonal is calculated. No dense query-by-query covariance is built. GP uses Euclidean spatial distance, does not impose PDE/boundary conditions, and currently has no dynamic or multi-output wave variant. Saved `solution_std`/`sample_std` and central Gaussian intervals feed the existing uncertainty plots and empirical coverage metrics; calibration depends on the assumed kernel.

**MAP** supports these small inverse parameterizations, with at most **16 free parameters** after exact constraints:

- T2: the existing bounded uniform-prior conductivity/diffusivity/wave-speed scalar, with fully known source and initial states. Partially hidden T2 Hybrid initial fields are unsupported.
- T1/G1 linear elliptic and T4 semilinear/p-Laplacian: unknown source amplitudes with a known operator. Gaussian amplitudes use their actual mean/std and a standardized quadratic penalty. Fully hidden uniform amplitudes use their actual box bounds and constant log prior. Underdetermined partial uniform constraints are unsupported.
- Configured T3 diffusion: Gaussian coefficients of the existing positive smooth `a0*(1+contrast*tanh(B*a))` field. The scalar scale uses its declared fixed value or bounded uniform prior. Exact Hybrid coefficient samples are inverted to linear basis constraints when the scale is fixed and samples are unsaturated. Joint unknown-scale/exact-field constraints and unknown interface fields are unsupported.

For an Inverse T4 MAP experiment, set `pdes.semilinear.forcing_basis` or `pdes.p_laplacian.forcing_basis` to at most 16 in the dataset config; the broader example config also permits larger problems, which MAP skips. Exact Gaussian source/field constraints use the same SVD nullspace as Bayes, without artificial measurement noise. Known beta, reaction and other parameters stay fixed; hidden amplitude labels, reference fields and unknown true coefficients are never used to initialize an optimizer. Fully specified Forward cases need one solve. Zero-observation cases use a prior mode; a flat uniform prior may have many such modes. `--map-maxiter 100` controls local L-BFGS-B with central numerical differences. MAP is a **parameter-space point estimate**, not a posterior mean or uncertainty distribution, and local optimization does not guarantee a global mode.

CSV/NPZ output records `map_success`, status/message, iterations, initial/final objective, objective evaluations, forward solves and failed solves, together with total inference time. A finite final iterate can be saved with `map_success=0` if optimization stops early; inspect this flag before treating it as a converged MAP estimate. Failed final PDE solves produce no prediction. POD fitting is separate from per-case inference time. Normalized MSE/RMSE, physical errors, relative L2, GP interval coverage and existing geometry/PDE/mode breakdowns retain the ordinary CSV structure.

```bash
# Compatible fixed-domain example; add --max-cases 1 for a tiny check.
python -m pdeep_simple baseline --data data/t1_small \
  --methods prior,rbf,bayes,pod,gp,map --pod-rank 32 --pod-ridge 1 \
  --workers auto --out runs/t1_classical

# G1: incompatible POD groups are explicitly skipped.
python -m pdeep_simple baseline --data data/g1_configured --pde poisson \
  --methods prior,rbf,bayes,pod,gp,map --workers auto \
  --normalization-run runs/a_configured --observations 0,8,16,32,64,128 \
  --out runs/g1_classical

# MAP applies only to the supported parameterizations above.
python -m pdeep_simple baseline --data data/t4_configured --pde semilinear \
  --methods prior,rbf,gp,map --workers auto --map-maxiter 100 \
  --out runs/t4_classical

# Evaluate the trained A/C runs on the same data/counts before comparing tables.
python -m pdeep_simple plot --runs runs/a_configured runs/c_configured runs/g1_classical \
  --what comparisons --metric normalized_rmse --out runs/classical_comparison
python -m pdeep_simple plot --runs runs/a_configured runs/c_configured runs/g1_classical \
  --data data/g1_configured --pde poisson --geometry square --modes inverse \
  --observations 32 --grid 128 --what fields,uncertainty --out runs/classical_fields
```

Use `--grids 128,256,512 --max-cases 2` with `baseline` for the existing multi-resolution evaluation; query chunking is unchanged. Same-case plots reuse each baseline directory's saved POD/GP/MAP settings. Only POD needs a learned training representation; GP defaults and MAP priors are explicitly specified. The existing training normalization still determines error units for every method; use `--normalization-run` matching the neural comparison run.

This addition changes `baselines.py`, `cli.py`, `evaluation.py`, `comparison.py`, `tests/test_science.py` and this README, with no new modules or dependencies. Checks were limited to small POD/GP/MAP examples, hidden-input invariance and the ordinary baseline/plot commands; no new neural training or large evaluation was run.
