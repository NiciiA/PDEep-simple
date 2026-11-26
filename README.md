# PDEep/simple

Numerical PDE datasets, A/C/D/E/F-Geo reconstructions, B-Geo information-routed physical posterior generation, conditional latent flows and classical baselines. The same CLI handles every supported tier; no notebook-specific training code is needed.

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
| G1 | `poisson,diffusion,elliptic_adr,heat,reaction_diffusion,wave,damped_wave,helmholtz,darcy,navier_stokes` | Source and/or initial fields; operator known; NS requires A joint conditioning with vector outputs |
| T2 | G1 scalar linear families, constant coefficient; excludes NS | Positive scalar conductivity, diffusivity, wave speed, wavenumber or permeability; source/initial fields known |
| T3 | `heat,diffusion,darcy` | High-dimensional initial field, or spatial diffusion/permeability coefficient |
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

## Fine references, small model meshes and 128 x 128 output

The existing `data` command now separates three resolutions. `--resolution` keeps
its old behavior and is also named `--model-resolution`: it controls the physical
input mesh. `--solver-resolution` requests a minimum numerical resolution, reached
by nested red refinements of that same polygon; alternatively specify
`--solver-refinements`. Each refinement divides every triangle into four. Targets
round **up** to a power-of-two multiple of the even model resolution; actual levels,
vertices and requested/effective resolutions are saved in case metadata.
`--evaluation-resolution 128` stores a regular masked 128 x 128 reference grid.
These output points are queries, not additional Transformer mesh tokens.

The physical fields are drawn with the existing priors on the model mesh, then
prolonged as exactly the same P1 functions to the numerical mesh; elementwise
coefficient fields remain cellwise constant under refinement. The PDE is solved
again there: dense output is interpolation of this **fine solve**, not enlargement
of a coarse solution image. Model-node labels, training queries and sensors all
use the fine reference. This preserves the original physical-field distributions;
it does not add unresolved fine-scale forcing or initial-condition complexity.
Configure basis counts separately if richer physical priors are wanted.

Nested refinement preserves the model polygon, including holes. For curved shapes,
boundary fidelity is therefore still set by the model geometry discretization.
Numerical spatial refinement does not establish temporal convergence: retain/use
appropriate `--dt` and `--frames`. Physical time interpolation remains piecewise
linear between stored snapshots; 128 x 128 is spatial, not a 128-frame time axis.

Run these commands from the package root. Counts are per PDE/geometry combination.

```bash
# A. Existing PDEs: solver 128, model 32, output 128 x 128.
python -m pdeep_simple data --tier G1 --pdes poisson,heat,wave,diffusion \
  --geometries square --train 8 --val 2 --test 2 \
  --solver-resolution 128 --model-resolution 32 --evaluation-resolution 128 \
  --frames 21 --dt .002 --workers 1 --seed 42 --out data/highres

# B. Add real Helmholtz and Darcy pressure.
python -m pdeep_simple data --tier G1 \
  --pdes poisson,heat,wave,diffusion,helmholtz,darcy --geometries square \
  --train 8 --val 2 --test 2 --solver-resolution 128 --model-resolution 32 \
  --evaluation-resolution 128 --frames 21 --dt .002 --workers 1 \
  --seed 42 --out data/highres_extended

# C. Optional steady low-Re Navier-Stokes, a separate vector dataset.
python -m pip install -e ".[navier-stokes]"
python -m pdeep_simple data --tier G1 --pdes navier_stokes --geometries square \
  --train 8 --val 2 --test 2 --model-resolution 16 --solver-resolution 64 \
  --evaluation-resolution 128 --workers 1 --seed 42 --out data/navier_stokes

python -m pdeep_simple inspect --data data/highres_extended --split test \
  --random 3 --observations 500 --noise 0 --out runs/highres_inspection
python -m pdeep_simple inspect --data data/navier_stokes --split test \
  --case 0 --observations 500 --noise 0 --out runs/navier_inspection
```

The paper-style experiment uses **500 total space-time measurements** across all
physical times, not 500 at each frame. Counts 0/32/64/128/256/500/512 already work
with `inspect`, `train` and `eval`. Observation sets are sampled on demand, not
duplicated in dataset files. `--noise 0` is now supported by neural evaluation
and plotting; `.01` retains the current noisy experiment. B uses its positive
surrogate-error/numerical variance floor even for noiseless reference sensors.
Likelihood-based classical baselines still require positive noise explicitly.

| Family | Actual numerical definition and inverse support |
|---|---|
| Poisson, diffusion, heat, wave | Existing P1 consistent-mass FEM, implicit Euler/Newmark, boundary/forcing and coefficient conventions unchanged. Existing mode definitions remain. |
| Helmholtz | Real `-Delta u-k(x)^2*u=f`, zero Dirichlet. Exact P1 polynomial reaction integration. The generator enforces `max(k)^2 < .8*pi^2*(1/Lx^2+1/Ly^2)` to remain below the first Dirichlet mode. Defaults k in [.5,1.5]; spatial variation is configurable. G1 hides source; T2 hides constant k with source known. |
| Darcy | Exactly the stationary diffusion solver, `-div(K grad p)=f`, zero pressure on the boundary. Positive smooth/interface permeability uses the existing coefficient generator. G1 hides source; T2 hides constant permeability; T3 hides smooth/constant nodal permeability. T3 interface inversion is rejected because current physical decoders do not generate elementwise interfaces. Stored flux is cell-mean `-K_e grad p_h`, not a conservative H(div) reconstruction or a new point-observation quantity. |
| Navier-Stokes | Optional scikit-fem Taylor-Hood P2 velocity/P1 pressure: steady Picard, or unsteady backward Euler with converged Picard. Zero velocity on all walls, `integral p(t) dx=0`. Constant viscosity defaults to [.1,.3], smooth random body-force component amplitudes default to std .01. Cases must converge and have a discrete Reynolds estimate <=5. |

Navier-Stokes stores both velocity components, pressure, viscosity, both forcing
components, geometry and no-slip information. Its actual P2 velocity polynomial is
stored and used for interpolation; it is not reduced to a P1 vertex image. Sensor
IDs are 0=velocity_x, 1=velocity_y, 2=pressure. These have separate training-only
RMS scales. `navier_stokes.known_physical_inputs` hides forcing for steady
Inverse/Hybrid; unsteady cases instead hide all/part of initial velocity, retaining
known forcing, viscosity and walls. A supports these fields with the opt-in
`--joint-conditioning --vector-output` configuration below. B/C/D/E/F and classical
baselines still reject NS. Hidden viscosity and nonzero/unknown boundaries remain
unsupported. This closed-domain setup is not a turbulence or lid-driven benchmark.

Helmholtz's power/sign graph requires A's `--joint-conditioning` or B. Darcy uses
the existing diffusion graph as well as its directed gradient/product/divergence
representation. NS stores typed momentum/advection/pressure-gradient and
incompressibility graphs, with velocity time derivatives and initial constraints
only for unsteady cases. Original PDE IDs, graph
vocabularies, parameter shapes and checkpoints are unchanged.

### Physical time and unsteady Navier-Stokes

| Implemented family | Physical time |
|---|---|
| Poisson, diffusion, elliptic ADR, semilinear, p-Laplacian, Eikonal, Helmholtz, Darcy | Stationary; only t=0. Here `diffusion` means stationary conductivity, and Darcy is stationary pressure. |
| Heat, reaction-diffusion | Time-dependent, first-order time derivative |
| Wave, damped wave | Time-dependent, second-order time derivative; retain displacement and its time derivative |
| Navier-Stokes | Steady by default, or explicitly `--navier-stokes-mode unsteady` |

A already encodes raw physical (x,y,t) and Fourier features in `Coordinates`,
retains observation times, and decodes arbitrary coordinate queries. This decoder
is reused. Physical time t is separate from B's flow-matching interpolation time s.
Existing scalar/wave A configurations and checkpoints keep their original shapes
and predictions. `--vector-output` adds a third output, a third observation ID,
and the second forcing component to the same A encoder/decoder, and requires a
fresh deterministic A joint-conditioning run. In mixed datasets, fluid and scalar
families use separate training-only normalization statistics; fluid velocity_x,
velocity_y and pressure have independent RMS scales (floor 1e-8).

Unsteady NS currently requires the **unit square**. Initial velocity is the curl
of a smooth random streamfunction that vanishes with its first derivatives on the
walls, projected into the discrete no-slip, divergence-free velocity space. The
initial pressure follows from a constrained acceleration solve, not an independent
IC. Each implicit step enforces weak incompressibility and a zero-mean pressure
gauge. Taylor-Hood velocity is weakly, rather than pointwise, divergence-free;
solver metadata records both weak divergence and quadrature L2 divergence.
The stored initial velocity is supplied only through masked mesh fields; the
streamfunction coefficients, initial pressure and reference trajectory are absent
from neural conditioning. Initial and boundary losses constrain velocity only.
The scalar weak loss is not available for NS.

```bash
python -m pip install -e ".[navier-stokes]"
# Small example, independent model/solver/output resolutions; not an accuracy claim.
python -m pdeep_simple data --tier G1 --pdes navier_stokes --geometries square \
  --navier-stokes-mode unsteady --time-end 1 --frames 21 --dt .005 \
  --model-resolution 8 --solver-resolution 16 --evaluation-resolution 128 \
  --train 8 --val 2 --test 2 --workers 1 --out data/ns_unsteady
# Steady uses the same command with --navier-stokes-mode steady and a new --out.
# Heat/wave and stationary diffusion share the loader and physical time settings:
python -m pdeep_simple data --tier G1 --pdes heat,wave,diffusion --geometries square \
  --time-end 1 --frames 21 --dt .002 --model-resolution 8 --solver-resolution 16 \
  --evaluation-resolution 128 --train 8 --val 2 --test 2 --out data/scalar_time

python -m pdeep_simple train --data data/ns_unsteady --model A \
  --joint-conditioning --vector-output --size s --updates 3000 --batch 2 \
  --queries 256 --modes forward,hybrid,inverse \
  --observations 0,32,64,128,256,500,512 --noise .001 \
  --loss field=1,initial=.1,boundary=.1 --device auto --run runs/a_ns_unsteady
python -m pdeep_simple eval --data data/ns_unsteady --run runs/a_ns_unsteady \
  --pde navier_stokes --modes forward,hybrid,inverse --observations 0,128,500 \
  --noise .001 --frames 5 --grids 128 --chunk 256 --device auto
python -m pdeep_simple inspect --data data/ns_unsteady --split test --case 0 \
  --observations 500 --noise .001 --out runs/ns_inspection
python -m pdeep_simple plot --run runs/a_ns_unsteady --what fields
```

The new generation settings can also be placed in the existing TOML config:
`[numerics] time_end=1.0` and `[pdes.navier_stokes] mode="unsteady"`,
`initial_amplitude=0.02`, `viscosity=[0.1,0.3]`, `forcing_frequency=0.0`.
Forcing is the spatial vector field times `cos(2*pi*forcing_frequency*t)`.
`--frames` controls saved physical times, `--dt` the maximum internal step;
both endpoints, including t=0, are stored. Sensors are continuous space-time
draws, not restricted to frames; 500 means 500 total scalar measurements across
the three quantities. Uniform continuous draws do not guarantee a sample exactly
at t=0; explicit endpoint queries are supported. A quarter of unsteady NS training
queries are placed at t=0 to supervise the velocity initial-state reconstruction.

References use P2 spatial velocity/P1 pressure and linear interpolation between
stored physical times; this interpolation is not an exact solution of the PDE.
NS evaluation uses five uniform times by default and saves all three quantities,
their errors, velocity magnitude and subsampled arrows. Pressure errors use the
stored zero-mean gauge without silently aligning a predicted pressure offset;
pressure mean is reported separately, and pressure IC/boundary errors are omitted.
The reported neural divergence is explicitly the **P1 projection** on the model
mesh, distinct from the numerical solver's P2 weak divergence. For NS refinement,
regenerate with a finer solver mesh/time step; eval `--refine` remains scalar-only.
Check convergence for each new regime: 128x128 output alone establishes no accuracy.

```bash
# Current compatible models; no training starts as part of generation.
python -u -m pdeep_simple train --data data/highres_extended --model A \
  --joint-conditioning --size m --modes forward,inverse,hybrid --loss field=1 \
  --updates 2500 --batch 2 --queries 512 --observations 0,32,128,500,512 \
  --noise .01 --device cuda --run runs/highres_a

python -u -m pdeep_simple train --data data/highres_extended --model B --size m \
  --b-repr-updates 1000 --b-forward-updates 2000 --updates 5000 \
  --modes forward,inverse,hybrid --loss field=1 --batch 1 --queries 512 \
  --observations 0,32,128,500,512 --noise .01 --device cuda --run runs/highres_b

python -m pdeep_simple eval --data data/highres_extended --run runs/highres_a \
  --pde wave --modes forward,inverse,hybrid --observations 500 --noise 0 \
  --grids 128 --frames 21 --chunk 128 --max-cases 3 --device cuda
python -m pdeep_simple eval --data data/highres_extended --run runs/highres_b \
  --pde wave --modes forward,inverse,hybrid --observations 500 --noise 0 \
  --grids 128 --frames 21 --chunk 128 --max-cases 1 --samples 2 --device cuda
```

Change `--pde` for other scalar families; T2/T3 experiments require separate
datasets with those tiers. B's 500-sensor information Jacobians can dominate time
and memory; retained rank does not reduce exact Jacobian cost. These are starting
commands, not established training budgets or GPU memory guarantees. A 32-subdivision
square has 1,089 model mesh nodes; a 128-subdivision solver has 16,641 vertices.
No mesh is silently downscaled to fit a model. Query chunking limits output-query
memory, not the cost of the model mesh or B's sensor sensitivities.

Flat NPZ files retain old case keys plus `reference_*` fine-mesh arrays and dense
`evaluation_points/values/mask/grid_x/grid_y/weights`. Invalid grid cells have no
stored solution values. Dense weights are equal over valid points and sum to the
polygon area; the existing weighted relative L2 and MSE/RMSE use the same points
for predictions and references, per scalar/wave channel and physical time. The
generic metric helper also supports a combined Euclidean vector norm. NS neural
evaluation remains unsupported until a vector architecture is added.

Fine plus dense 21-frame wave values alone cost about 5.3 MiB per square case
before compression; meshes, inputs and model-node labels add storage. Larger
solver meshes and more frames scale this cost accordingly. Geometry/time remain
float64; physical labels are float32. NS at solver resolution 128 has about
149,000 mixed unknowns, and sparse factorization can require much more memory
than its saved fields. Use a small generation run and check refinement convergence
before scaling up. The scalar high-resolution path remains dependency-free beyond
the existing stack; NS fails with installation guidance if its optional backend
is absent. Existing datasets/loaders remain compatible. Training skips loading the
redundant dense-output arrays, but still loads fine references eagerly; very large
case counts can exceed CPU RAM. No lazy-loading framework is added here.

## Training

All variants use the same data, conditioning and field decoder interfaces. These are modest examples, not evidence of trained accuracy.

### A-Geo

```bash
python -m pdeep_simple train --data data/g1_small --model A --size s \
  --modes forward,inverse,hybrid --loss field=1 --updates 400 \
  --batch 2 --queries 512 --device auto --run runs/a
```

### A-Geo with joint conditioning

Original A keeps individual mesh, PDE and sensor tokens alongside learned summary
tokens; coordinate queries cross-attend to that memory. `--model A
--joint-conditioning` enables three changes together:

- One existing cross-attention block lets mesh nodes read PDE and observation
  tokens once per case. Updated mesh nodes stay distributed in memory and are
  available through barycentric interpolation. Original local mesh features also
  remain available to each query.
- A small per-head MLP adds sensor-specific bias **before softmax** in the existing
  query-to-memory attention. Its inputs are signed dx/dy/dt, spatial distance,
  observed-quantity embedding, and log normalized noise. Differences use domain
  spans and the physical time interval; stationary dt is zero. Attention remains
  global and allows earlier/later and cross-quantity observations. This replaces
  A's fixed Gaussian sensor-summary path; individual sensor tokens remain.
- Directed expression graphs replace the old undirected family templates. Edges
  run operand -> operator with input roles, including ordered subtraction and
  powers. Masked scalar values and field references enter **before** two graph
  message-passing rounds. Spatial fields stay on the mesh; permitted exact element
  coefficient samples remain spatial tokens bound to the coefficient node. Basis
  channel embeddings are separate from mathematical operator embeddings.

The graphs follow `physics.solve`: write A(u) = -div(a grad u). Poisson/diffusion
use A(u)=f; ADR adds b dot grad u + rho*u; semilinear adds beta*u^3. Heat and
reaction-diffusion use u_t+A(u)+rho*u=f*cos(2*pi*nu*t), with rho=0 for heat.
Waves use u_tt-div(c^2 grad u)=f*cos(2*pi*nu*t); damped waves add gamma*u_t.
The legacy unconfigured T4 wave coefficient convention is retained explicitly.
p-Laplacian uses -div((epsilon^2+|grad u|^2)^((p-2)/2) grad u)=f with the actual
solver regularization; Eikonal uses |grad u|=1/c. Initial conditions and homogeneous
Dirichlet constraints are represented separately. Hidden values never populate
graph nodes; partial field masks remain on the mesh. The new graph does not use a
PDE-family ID token. It describes the implemented equations, not arbitrary PDEs
or demonstrated generalization to unseen compositions.

Train from scratch in a separate directory (run from the package root):

```bash
python -u -m pdeep_simple train \
  --data data/colab_g2 --model A --joint-conditioning --size m \
  --modes forward,inverse,hybrid --loss "field=1" \
  --updates 2500 --batch 16 --queries 4096 \
  --observations 0,8,16,32,64,128 --noise 0.01 --seed 42 \
  --validate-every 500 --device cuda --run runs/colab_a_joint

python -m pdeep_simple eval --data data/colab_g2 --run runs/colab_a_joint \
  --pde wave --modes forward,inverse,hybrid \
  --observations 0,8,32,128 --noise .01 --seed 123 \
  --frames 21 --max-cases 3 --device cuda --out runs/eval_a_joint

python -m pdeep_simple plot --run runs/eval_a_joint --what fields
```

Repeat evaluation with the original A run and the same filters, seed, counts,
noise and frames, changing only `--run` and `--out`. Existing metrics retain
PDE/mode/sensor-count groups, physical-time errors and separate wave u/v errors;
existing reference/prediction/absolute-error plots are reused. Random test cases
do not establish held-out operator-composition or geometry-family generalization.
Repeat with `--pde heat` and `--pde diffusion` in separate output directories for
the early-time heat and spatial-coefficient comparisons.

Without the flag, A and its checkpoints retain their original behavior. C/D/E/F
are unchanged and reject the flag. Eval/plot restore the variant from the saved
configuration; original and joint A checkpoints cannot be interchanged. Resume
restores the saved architecture. No training stage, sampling rule, loss or output
channel convention changes: A still predicts wave u/v independently.

Additional cost is one mesh-to-conditioning attention block per case,
O(N_mesh*(N_PDE+N_obs)*width), plus small directed-graph and relative-bias MLPs.
The sensor bias is computed once per query chunk and reused by all decoder
layers, without another dense query-to-mesh bias matrix. Standard query attention
still reads A's full memory. Accuracy improvements need training and comparison;
none are assumed here.

### B-Geo: PRISM research implementation

B generates **physical causes**, then passes each candidate through a learned
forward solution operator. Its posterior is approximate; it is not an established
Bayesian solver or evidence of generalization to arbitrary equations.

The physical codec has a configurable 128-dimensional latent, grouped by forcing,
initial displacement/temperature, initial velocity, and coefficient. A shared
coordinate/geometry decoder is nonlinear in the latent. Wave u0 and v0 have
separate reconstruction losses. Known entries are clamped exactly; coefficients
are positive, with scalar prior bounds used where the dataset supplies a range.
An exact known entry of a declared constant coefficient fixes it globally; this
information is propagated without reading any hidden entries.
Physical reconstruction uses the training field scales (including separate u0/v0
scales); solution and sensor likelihoods use the solution-output scales.
Current boundary conditions, beta, damping, reaction and advection are supplied
and stay known: B does not invent unknown versions of these quantities.

The forward network reuses A's memory/query decoder and the existing directed
physical encoder. It receives complete **candidate** fields and no observations,
original sensor values, or FEM operator caches. E's initial-state anchoring and
v=du/dt are reused for time-dependent equations; no modal basis or numerical solver
is run at inference. One candidate determines the entire trajectory at all query
coordinates. Forward error, boundary error and PDE residuals can remain nonzero.

The posterior network uses the same physical/observation encoder and one shared
attention backbone (medium: width 256, eight heads, six layers). Four interacting
hypothesis tokens describe candidate physical groups. Compact information tokens
carry sensitivity directions and their strengths. These groups are **not** fixed
observable/ambiguous latent coordinates.

Training-only complete physical representations estimate a regularized covariance
and Cholesky whitening transform; both directions of the transform are saved in
the checkpoint. Whitening is approximate and does not establish a Gaussian
physical prior. Conditional flow matching transports a Gaussian base to the
conditional physical latent distribution, including zero-sensor conditional priors.

For each selected candidate, B differentiates only the point-sensor predictions:
J=dh/dz and G=J^T Sigma_eff^-1 J, in whitened latent coordinates. Sensor values,
predictions and noise use the same quantity-specific scales. Sigma_eff includes
measurement variance, a small numerical floor, and per-output forward-validation
MSE as a conservative surrogate-error estimate. This global error floor is not a
calibrated, state-dependent surrogate error model.
The variance enters routing and likelihood diagnostics; CFM learns from FEM
observations and does not perform an exact likelihood-corrected Bayesian update.

The retained eigenvectors implement W v = V diag(lambda/(lambda+tau)) V^T v
without a dense projector. Two small heads share the attention backbone:
v = W*v_evidence + (I-W)*v_ambiguity. Projected head supervision has weight .1;
a small initial-estimate loss has weight .05. Gradients stop through the information
calculation. `--b-routing ordinary` uses ordinary CFM with the same two-head
capacity and no information calculation; `fixed` computes information once at a
learned initial estimate; `dynamic` refreshes at selected states of each sample
(default start and midpoint). Both heads see the full context and latent state.
Zero observations give W=0; completely known physical systems bypass posterior
integration and use the known fields directly.

Three explicit stages share one model/checkpoint and existing data/loss conventions:

1. Physical representation and per-quantity reconstruction.
2. Forward operator, supervised with complete FEM physical inputs and field targets.
3. Frozen codec/forward operator, trainable conditioning and routed posterior flow.

The stage counts below are an experimental allocation, **not a claim of sufficient
training**. B is inherently generative; `--flow` is unnecessary. Run from the package root:

```bash
python -u -m pdeep_simple train --data data/colab_g2 --model B --size m \
  --b-repr-updates 1000 --b-forward-updates 2000 --updates 5000 \
  --b-latent-dim 128 --b-info-rank 16 --b-tau 1 --b-routing dynamic \
  --modes forward,inverse,hybrid --loss "field=1" \
  --batch 16 --queries 4096 --observations 0,8,16,32,64,128 \
  --noise .01 --seed 42 --validate-every 500 --device cuda --run runs/colab_b

python -m pdeep_simple eval --data data/colab_g2 --run runs/colab_b \
  --pde wave --modes forward,inverse,hybrid --observations 0,8,32,128 \
  --nested-observations --noise .01 --seed 123 --frames 21 --max-cases 3 \
  --samples 16 --flow-steps 32 --device cuda --out runs/eval_b_wave

python -m pdeep_simple plot --run runs/eval_b_wave --what fields,uncertainty
```

Of 5,000 updates, 1,000 train the codec, 2,000 train the forward map and 2,000 train
the posterior. Resume restores these settings and optimizer/RNG state. Flow-stage
validation uses CFM loss; use `eval` for posterior-mean field accuracy and coverage.
Training noise varies from .5 to 1.5 times `--noise`. A/B/C/D/E/F share existing
sampling and mask definitions; B additionally trains inverse/hybrid zero-sensor
priors. `--nested-observations` explicitly uses prefixes of one common sensor draw
across counts, and can also be used for matching baseline observations.

Optional `--b-sensitivity-weight .001` adds forward directional supervision for
multiplicative coefficient changes, using neural JVPs and centered differences
from two existing FEM solves on 16 queries. It is omitted for p-Laplacian, whose
solver does not use that coefficient. Other physical sensitivities receive field
supervision only; their derivative accuracy has not been established.

Evaluation retains ordinary PDE/mode/count/geometry/time/u-v metrics, posterior
mean/std/intervals and solution sample plots. B also saves physical means/std,
reference fields, physical samples, scalar coefficient samples, per-sample solution
RMSE, leading information eigenvalues/weights, and candidate states at refreshes.
Metrics include unknown-entry physical RMSE/coverage, observation residuals,
effective information dimension and counts of retained weights >= .5. `plot`
adds physical uncertainty maps, coefficient histograms and information spectra.
Reference quantities are accessed only for training targets and evaluation errors.

All ten existing PDE families use the same codec and learned forward network:
Poisson, diffusion, heat, wave, semilinear, ADR, reaction-diffusion, damped wave,
p-Laplacian and Eikonal. G1 wave speed is known; its inverse task hides initial
fields (and nonzero configured forcing). T2 supplies scalar-coefficient inverse
tasks; T3 diffusion supplies spatial-coefficient inversion; Eikonal hides speed.
Known damping/reaction are supported, their inversion is not. No new PDEs,
boundary types, sensor types or numerical solvers are introduced.

For ablations, train separate runs with `--b-routing ordinary`, `fixed`, and
`dynamic`, keeping all dimensions, stage allocations, data and random seeds the
same. An isolated exact linear-Gaussian experiment uses B's actual routed-flow
component with an identity codec and exact linear forward map:

```bash
python scripts/prism_gaussian.py                         # exact reference; no training
python scripts/prism_gaussian.py --updates 2000 --routing dynamic --samples 256
```

It prints learned versus exact posterior mean/covariance errors after requested
training. In this linear problem fixed/dynamic information coincide. This isolates
transport quality and does not qualify the complete learned PDE model as calibrated.

Information computation forms an exact sensor-by-latent Jacobian in VJP blocks of
eight (scalar VJPs on MPS), then a small CPU float64 eigensolve. Only the leading
`--b-info-rank` directions are retained. Rank truncation treats discarded likelihood
information as zero; reducing rank does not reduce the exact-Jacobian cost. The
cost scales with sensor count and refresh count, and can dominate sampling. CSV
metrics report actual `information_seconds` and refresh counts. Candidate forward
contexts are cached across query chunks/grids; there are no repeated FEM modes.
CUDA/MPS use ordinary PyTorch; small whitening/eigen solves use CPU when needed.

Local likelihood sensitivity is not global identifiability, and it can miss
disconnected posterior modes. The latent representation, Gaussian-base transport,
finite rank, limited training and neural forward derivatives all introduce
approximation. Forward intervals express physical-state sampling uncertainty,
not a full distribution of surrogate error. No optional joint-refinement stage,
streaming update operator, active sensing, or equation discovery is added. Original
A (including joint conditioning), C, D, E and F behavior/checkpoints are unchanged.

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
python -u -m pdeep_simple train --data data/colab_g2 --model E --size m \
  --wave-modes 64 --modes forward,inverse,hybrid --loss field=1 \
  --updates 2500 --batch 16 --queries 4096 --observations 0,8,16,32,64,128 \
  --noise .01 --seed 42 --validate-every 500 --device cuda --run runs/colab_e_evolution
```

E shares A's conditioning backbone and uses initial-state-aware evolution for every genuinely time-dependent PDE in this project. Initial temperature, or wave displacement and velocity, are supplied when known and inferred by the existing initial-state head when hidden. Hybrid masks retain each known initial node and infer the rest; P1 interpolation evaluates these fields at arbitrary query locations. One E model handles Forward, Inverse and Hybrid. Hidden targets never construct the operator or initial fields at inference.

| PDE | Evolution |
|---|---|
| `heat` | Mass-normalized FEM diffusion modes decay as `exp(-lambda*t)`. |
| `wave` | Existing analytic oscillation, forcing and initial-state-aware wave decoder. |
| `damped_wave` | The same spatial modes with `q'' + gamma*q' + omega^2*q = F(t)`. |
| `reaction_diffusion` | Diffusion modes plus the actual dataset reaction `R(u) = -rho*u`. |
| `poisson`, `diffusion`, `elliptic_adr`, `semilinear`, `p_laplacian`, `eikonal` | Existing coordinate decoder; these equations are stationary. |

Heat and reaction-diffusion use `u = u0 + Phi*(q(t)-q(0)) + t*r(t)`, where r is A's existing coordinate decoder and t is physical time from the initial state. Diffusivity is included in the stiffness eigenvalues once. Constant forcing uses `(1-exp(-lambda*t))/lambda`, with its zero-rate limit; the dataset's cosine forcing uses its corresponding analytic convolution. Reaction-diffusion currently has a **linear sink**, not a nonlinear reaction: `Phi^T M R(Phi*q) = -rho*q`, so the exact reduced rates are `lambda + rho`. There is no additional nonlinear solver or invented reaction function.

The wave decoder uses `u = u0 + t*v0 + t^2*r(t)`, where t is physical time relative to the initial time. When the positive spatial operator is fully known, it adds `u_m(t) - u_m(0) - t*v_m(0)` from the existing mass-normalized FEM modes. Their analytic evolution retains constant/cosine forcing, including resonance; fully known initial fields and forcing are projected, while hidden modal coefficients are inferred from conditioning. Unknown operators simply omit this contribution and still use the initial-aware decoder.

Damped wave uses this same anchored displacement/velocity construction. One differentiable 4x4 matrix exponential per mode advances displacement, velocity and the cosine-forcing clock on 64 intervals per case. It handles underdamped, critically damped and overdamped dynamics, including forcing resonance. Cubic Hermite interpolation reuses that trajectory at arbitrary query times; its derivative supplies modal velocity. The undamped wave formula above is unchanged.

Wave velocity is the physical-time derivative of displacement, using modal derivatives and autograd for A's residual decoder, with separate u/v output scales. Chosen initial fields are exact at t=0 up to interpolation and floating-point accuracy. Training puts one quarter of E's time-dependent queries at the initial time (at least one); validation draws stay unchanged. The redundant kinematic penalty is skipped for both E wave branches and logged as zero; the existing finite-difference evaluation diagnostic remains available and need not be zero on coarse time grids. Inferred initial fields can still be wrong, and these constructions do not enforce the full PDE or boundary conditions. Other optional physics losses retain their definitions, including with `--staged`.

All four dynamic branches retain their initial-aware learned evolution when an operator is unavailable. With `--infer-operator`, hidden T2 scalar diffusivity/speed uses the existing positive coefficient head and cached unit-operator geometry modes: rates scale by alpha for diffusion and squared frequencies by c^2 for waves. Known coefficients remain fixed. Without that option, unavailable modal contributions are omitted. The compact latent, assimilation and flow additions below remain specific to undamped waves.

`--wave-modes` controls the shared mode count for all four dynamic PDEs. Modes keep the existing per-mesh/operator/device cache, using O(nodes × modes) storage per operator; resume rebuilds it. Only low modes are computed (at most interior degrees of freedom minus one, except a single interior node). The evolution extension adds no parameter keys; existing weights still load strictly, and undamped wave and stationary predictions remain unchanged. Heat, reaction-diffusion and damped-wave decoding has changed, so train in a new run directory for comparisons. Architectural settings are restored on resume.

E has four optional additions for undamped inverse waves. None enables the others automatically:

| Flag | Effect |
|---|---|
| `--initial-latent 32` | Infer one compact initial-state vector; 16 also works. A small shared coordinate/geometry decoder reconstructs nodal u0/v0 on variable meshes, then uses the existing modal projections and evolution. Known initial nodes bypass this reconstruction exactly. |
| `--assimilation` | Correct unknown initial-state components using a diagonal modal prior, actual sensor coordinates/times, u/v quantity and normalized noise. The small Gaussian update uses `torch.linalg.solve` and the complete predicted observation, including fixed forcing and learned residual. No observations means no correction. |
| `--infer-operator` | For hidden T2 constant wave speed/diffusivity, reuse the positive coefficient head with cached geometry-only FEM modes. Reduced mass is I; stiffness is scaled by c^2 for waves or alpha for diffusion. Gradients reach the head without recomputing eigenmodes for each prediction. Known parameters remain fixed. |
| `--flow --initial-latent 32` | Reuse the existing AE/FM stages, velocity MLP, Gaussian noise, linear matching path and Heun sampler for the initial-state latent. One sample defines a complete u/v trajectory; known components bypass samples. Add `--assimilation` to correct each sampled trajectory. |

The compact encoder sees complete initial fields only as representation-training targets. In deterministic training, a mass-weighted normalized initial-state reconstruction loss (weight 1, logged as `initial_reconstruction`) accompanies the existing selected field/physics losses. Conditional inference uses only observations and masked known inputs. E flow first learns the initial representation and E decoder, then freezes them and trains the existing conditional flow; `--initial-latent` controls E's dimension rather than `--latent-dim`. Its flow stage uses only undamped wave cases, while non-wave decoding remains deterministic. Both training and validation splits must contain waves. Staged physics regularization remains available for deterministic E and cannot be combined with flow.

For Colab, use the usual dataset and CUDA settings, with separate runs:

```bash
# Compact deterministic initial state
python -u -m pdeep_simple train --data data/colab_g2 --model E --size m \
  --wave-modes 64 --initial-latent 32 --modes forward,inverse,hybrid \
  --loss field=1 --updates 2500 --batch 16 --queries 4096 \
  --observations 0,8,16,32,64,128 --noise .01 --seed 42 \
  --validate-every 500 --device cuda --run runs/colab_e_latent

# Independent correction experiment, using the original initial-state heads
python -u -m pdeep_simple train --data data/colab_g2 --model E --size m \
  --wave-modes 64 --assimilation --loss field=1 --updates 2500 \
  --batch 16 --queries 4096 --validate-every 500 --device cuda --run runs/colab_e_assimilation

# Conditional initial-state flow; correction and operator inference stay optional
python -u -m pdeep_simple train --data data/colab_g2 --model E --size m \
  --wave-modes 64 --initial-latent 32 --flow --ae-updates 1000 \
  --flow-steps 16 --loss field=1 --updates 2500 --batch 16 --queries 4096 \
  --validate-every 500 --device cuda --run runs/colab_e_flow
```

For the scalar-operator experiment, add `--infer-operator` to an E command and train on existing **T2 time-dependent** data, with `--loss field=1,aux=.1` if coefficient supervision is wanted. Adding this flag to G1 data does not hide its already supplied speed/diffusivity. It can also be used without a compact latent or assimilation.

```bash
python -m pdeep_simple eval --data data/colab_g2 --run runs/colab_e_flow \
  --pdes wave --limit 1 --modes forward,inverse,hybrid --observations 16 \
  --samples 8 --save-samples 4 --flow-steps 16 --device cuda
python -m pdeep_simple plot --run runs/colab_e_flow --what fields,uncertainty
```

Evaluation retains the existing normalized/physical field errors and `initial_rmse`, and adds observation u/v, normalized and noise-weighted RMSE before/after correction, inferred/true T2 coefficient errors, and `initial_sample_std_mean` for compact flow samples. The correction minimizes noise-weighted discrepancy; individual channel errors or normalized field errors can increase. Saved fields include individual samples, mean, standard deviation and reference at five physical times by default. `eval --assimilation` / `--no-assimilation` and `--infer-operator` / `--no-infer-operator` override those two settings without modifying checkpoint weights, and save separate selected evaluations. The existing `--pdes`, `--limit` and `--modes` filters still apply.

These are research approximations. Truncation leaves unresolved spatial components to the residual; a compact latent may lose further detail. The 64-interval damped trajectory has interpolation error, especially for rapid dynamics. The current dataset supplies scalar constant damping and reaction; spatial damping matrices and nonlinear reactions are not implemented by the dataset or this extension. Only hidden T2 **constant scalar** speed/diffusivity is inferred; unavailable spatial operators retain the initial-aware learned fallback. The undamped-wave Gaussian correction is conditional on a fixed residual/operator with a unit diagonal prior in normalized modal increments; it can leave the compact decoder's latent manifold. Noise variance has a 1e-6 normalized floor. Unknown c uses one bounded, regularized observation-consistency step with a decrease check, rather than an exact parameter posterior. Assimilation without an available modal operator records errors but skips correction. Flow samples represent learned initial-state uncertainty, not sampled forcing/operator uncertainty or calibrated Bayesian uncertainty; variation and lower mean MSE are not guaranteed. Modal FEM construction stays cached on CPU. The small assimilation solve and damped matrix exponential use CPU fallbacks for MPS where needed, retaining gradients.

```bash
python -m pdeep_simple eval --data data/colab_g2 --run runs/colab_e_evolution \
  --pdes heat,wave --limit 3 --modes forward,inverse,hybrid --device cuda
python -m pdeep_simple plot --run runs/colab_e_evolution --what losses,fields
```

Evaluation keeps the existing metrics and grid/refinement options. For E's time-dependent PDEs, default evaluation/field plots show physical times 0, .25, .5, .75 and 1, interpolating stored FEM references as needed. E records `modal_evolution`, `initial_aware_evolution`, `evolution_rule`, and the actual `wave_modes` count for every PDE. The existing `modal_wave` diagnostic and `*_e_wave.png` undamped-wave panel remain available. Use the normal `plot --runs` comparisons across separate runs.

### F-Geo: hidden-state inference with E's physical decoder

F replaces the former learned-evolution F, SIREN G and kernel H models. It subclasses E, reusing its conditioning backbone, initial-state codec, cached FEM modes, heat/wave/damped-wave/reaction-diffusion evolution, residual decoder and wave velocity derivative. Stationary PDEs keep E's coordinate decoder. No new PDE solver or flow network is introduced. Existing results remain on disk; old F checkpoints are rejected by an architecture-version marker, and G/H are no longer model choices.

| Variant | Adds |
|---|---|
| `--f-variant base` | Compact initial latent (default 32) and E's noise-aware modal correction. |
| `--f-variant align` | Base plus `align_weight * mean((z_obs - stopgrad(z_full))^2)`, default weight .1. |
| `--f-variant iterative` | Alignment plus up to three regularized space-time latent/parameter corrections, replacing the one-shot modal update. |
| `--f-variant observable` | Iterative inference with a small local sensitivity matrix, regularized covariance and preconditioned updates. |
| Any variant + `--flow` | Existing AE/FM pipeline and Heun sampler over the initial-state latent. |

On undamped waves, F-Base reproduces E with the same weights and `--initial-latent 32 --assimilation` (and the same operator-inference setting). E currently applies these two options only to undamped waves. F extends that codec and the same modal Gaussian update to scalar heat/reaction-diffusion states and damped-wave u/v, retaining E's physical decoder throughout. Scalar initial encodings use u only, with an unused zero v channel. Known initial nodes bypass the codec and remain fixed through every correction. Other known physics remains supplied by E's existing masks. Add `--infer-operator` to permit the existing positive T2 scalar speed/diffusivity estimate; hidden spatial operators still use E's initial-aware learned fallback.

The complete initial field is used only for training targets and representation losses. Alignment trains the observation estimate toward the stopped-gradient full-state encoding; it never passes the full field into the conditioning encoder. Validation and inference reconstruct from known inputs and observations, including during F's AE stage. `initial_reconstruction` and `alignment` are logged separately from the field loss. Alignment is used by align/iterative/observable during deterministic or AE training; the existing FM stage freezes that representation and trains only the flow velocity network.

Iterative inference minimizes `sum((prediction-observation)^2 / variance) + ||state-prior||^2`, using actual coordinates, physical times, u/v quantities and noise. The state is the initial latent plus the log scalar operator parameter when it is both hidden and supported. The complete E prediction, including the residual and fixed contributions, is used. Gradients of each update are stopped to avoid second-order unrolling, while the final field loss still reaches the observation encoder through the updated state. Small steps and a short decrease check regularize the refinement. `--f-steps 2`, `3` or `4` controls the maximum count; zero sensors skips correction.

Observable uses `J = d(predicted sensors)/d(state)` and `P = solve(I + J.T @ R^-1 @ J, I)`, with a unit diagonal prior and the existing 1e-6 normalized noise-variance floor. This smoothly restrains weak directions rather than discarding them. The local covariance also preconditions correction. Evaluation reports `latent_mean_variance`, `latent_strong_directions`, `latent_weak_directions` and `latent_effective_directions`; a strong direction has marginal latent covariance eigenvalue below .5. Zero sensors retains the prior covariance. Known-field masks can make latent directions unobservable even when the physical field itself is supplied exactly.

```bash
# Reference; replace base with align, iterative or observable, using a separate run.
python -u -m pdeep_simple train --data data/colab_g2 --model F --f-variant base \
  --size m --wave-modes 64 --initial-latent 32 --modes forward,inverse,hybrid \
  --loss field=1 --updates 2500 --batch 16 --queries 4096 \
  --observations 0,8,16,32,64,128 --noise .01 --seed 42 \
  --validate-every 500 --device cuda --run runs/colab_f_base

python -u -m pdeep_simple train --data data/colab_g2 --model F --f-variant iterative \
  --size m --wave-modes 64 --initial-latent 32 --align-weight .1 --f-steps 3 \
  --modes forward,inverse,hybrid --loss field=1 --updates 2500 --batch 16 \
  --queries 4096 --observations 0,8,16,32,64,128 --noise .01 --seed 42 \
  --validate-every 500 --device cuda --run runs/colab_f_iterative

# Flow can be added to any variant; total updates include both stages.
python -u -m pdeep_simple train --data data/colab_g2 --model F --f-variant observable \
  --size m --wave-modes 64 --initial-latent 32 --flow --ae-updates 1000 \
  --flow-steps 16 --loss field=1 --updates 2500 --batch 16 --queries 4096 \
  --validate-every 500 --device cuda --run runs/colab_f_observable_flow

python -m pdeep_simple eval --data data/colab_g2 --run runs/colab_f_iterative \
  --pdes heat,wave,damped_wave,reaction_diffusion --limit 3 \
  --modes forward,inverse,hybrid --observations 0,8,16,32,64,128 --device cuda
python -m pdeep_simple plot --run runs/colab_f_iterative --what losses,fields
```

Deterministic F defaults to correction on; `--no-assimilation` is an explicit ablation. Flow defaults to correction off because its condition already contains the sensors. Adding `--assimilation` applies at most one small regularized consistency refinement per conditional sample, rather than counting those observations as independent new evidence. Observable flow without correction still reports local sensitivity diagnostics. One sampled initial state defines a complete trajectory across all times/query chunks; known fields bypass every sample. Training/validation splits for F flow need at least one supported dynamic case. Scalar PDEs and both wave channels share the existing AE/FM code; stationary predictions remain deterministic. Flow still cannot be combined with staged physics training.

Saved evaluations distinguish F variants by method name and retain field/initial/boundary errors, separate wave u/v errors, observation errors before/after, accepted `refinement_steps`, and five physical times 0, .25, .5, .75 and 1. Flow uses the existing individual-sample, mean, standard-deviation, reference and error plots. Compare against a separate E run with `--initial-latent 32 --assimilation`, using identical cases, sensor draws, seeds and training normalization.

These are research approximations: a small latent can miss detail, three first-order steps need not find the best reconstruction, and lower sensor error need not lower field error. The one-shot modal correction can leave the compact decoder's latent manifold; iterative variants remain in it. Observable covariance is local and depends on the chosen latent units/unit prior, not calibrated uncertainty or a global Bayesian posterior. With flow it is a sensitivity diagnostic/preconditioner, not independent evidence added to the learned conditional distribution. The Jacobian and solves are small, but observable inference costs more than base/iterative. The scalar reaction/damping and modal truncation/interpolation limits described for E still apply. CUDA/MPS use the existing decoder support and small CPU solve fallbacks where needed.

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
| `--flow` | Field representation stage followed by conditional latent flow matching (A/C, E with `--initial-latent`, or F) |
| `--ae-updates`, `--latent-dim` | First-stage updates and A/C latent dimension; E/F use `--initial-latent`; require 1 <= AE updates < total updates |
| `--resume` | Restore `last.pt`, including model, optimizer, RNG and saved stage settings |

`--width`, `--layers`, `--heads` and `--memory` override presets; memory size affects the attention models A/D/E/F. `--observations`, `--noise`, `--seed`, `--lr` and `--validate-every` retain their normal CLI meanings. Inverse/Hybrid training uses nonzero observation counts. For T2 coefficient-head supervision add `aux=.1`; `kinematic` applies to wave/damped wave, `initial` to known initial nodes, and `boundary` to zero boundary values. `weak` uses P1 test functions and known forcing/operator information (or the T2 scalar prediction); it is not a strong-form derivative guarantee. Eikonal has no elliptic weak loss. The networks do not hard-enforce the PDE or boundary conditions.

Flow's **total** `--updates` includes both stages: the A/C examples use 100 representation updates and 300 flow updates. First train a reference-field encoder and latent-conditioned A/C decoder (or the compact initial-state representation and E decoder); then freeze that representation and train a conditional velocity MLP using training-only latent normalization. Physical PDE time and latent ODE flow time are different variables. Inference uses known inputs and observations, never the reference encoder. One latent describes the entire space-time field, including wave u/v, and is reused across query chunks.

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

For a quick look at a few wave cases:

```bash
python -m pdeep_simple eval --data data/colab_g2 --run runs/colab_e \
  --pdes wave --limit 3 --modes forward,inverse,hybrid --device mps
python -m pdeep_simple plot --run runs/colab_e --what fields
```

`--pdes` also accepts comma-separated names such as `wave,heat,diffusion`. `--limit` is an alias for `--max-cases`: it takes the first matching physical cases in manifest order, before loading fields or running inference. Selection is deterministic and independent of the model; modes and observation draws retain their existing meanings. A limit of three with all three modes evaluates three physical cases under each mode.

Filtered neural evaluations use a fresh `evaluations/selection_*` folder inside the run (`--out`, if supplied, is the parent instead), preserving previous full results in `evaluation/`. Existing `evaluation.json` records the selected case paths, and filtered runs save all selected predictions by default; `--save-cases` can explicitly reduce this. Plotting the run uses its most recent default full or selected evaluation, without inference. Filtered field plots include each saved case/mode at its largest saved observation count, use all groups unless `--max-fields` limits them, and save inside that selection's `plots/` folder. Custom output folders can be plotted directly with `plot --run PATH --what fields`. Selected wave runs use five uniform physical times by default (0, .25, .5, .75, 1), with existing FEM time interpolation for references; u/v are plotted separately on the same domain and shared reference/prediction colour limits.

Inspect the exact sparse sensors separately from dense known initial fields and coefficients:

```bash
python scripts/inspect_observations.py --data data/colab_g2 --run runs/colab_f_iterative \
  --case test_wave_square_00000 --observations 128
```

This prints all three modes, quantity IDs, physical-time counts, noise, known-input mask counts and the first ten actual measurements, then saves `observation_times.png` and `space_time_observations.png` under the evaluation's `observation_inspection/CASE_nCOUNT/` folder (`--out` overrides it). It reads saved NPZ sensors/masks first, choosing the latest matching evaluation selection. Use `--evaluation PATH/TO/selection_*` for a specific selection or custom evaluation output. Missing files are reproduced with the saved evaluation seed/noise and the same case-specific sensor/mask RNGs as `eval`, without loading model weights or running inference. Without any saved evaluation settings, it explicitly uses the current eval CLI defaults (or `--seed`/`--noise`), never the training seed; unrecorded case/count/mode combinations are labeled as reproductions.

Inspect the trained F-Iterative prediction before correction and after each accepted step:

```bash
python scripts/inspect_f_corrections.py --data data/colab_g2 --run runs/colab_f_iterative \
  --case test_wave_square_00000 --observations 128 --device mps
```

The utility requires saved sensors and masks and never samples new measurements. It uses the selected evaluation's checkpoint/settings (`--evaluation` and `--checkpoint` can select them explicitly), the same FEM nodes and five physical times at every stage, and the normal decoder/evolution. CSVs under `RUN/correction_diagnostics/CASE_nCOUNT/` contain each mode's sensor/field/initial RMSE, all per-time field errors, and Hybrid known/hidden initial errors. The main field RMSE is the arithmetic mean of five spatial FEM-quadrature RMSEs; sensor RMSE is over the actual measurements. Each mode gets correction-versus-RMSE and initial reference/stage-0/final/error PNGs. A stopped/skipped correction repeats its final state and is labeled unchanged, rather than inventing additional updates. An optional `refine(..., return_stages=True)` returns intermediate contexts without changing the default result, parameters or checkpoints.

The diagnostic reports physical-unit errors and explains the current normalization: both measured values and sigma are divided by their own u/v output scales. The correction's data term is equivalent to `sum(residual_physical^2 / (sigma_physical^2 + 1e-6*output_scale^2))`; the variance floor slightly changes relative channel weighting. There is no extra velocity-amplitude multiplier, though sensor counts, physical residuals and sensitivities affect the update. The existing unit prior-distance penalty, RMS-limited .25 gradient direction, backtracking and optional log-parameter clipping are unchanged. The script prints the actual scales and effective noise for the loaded run.

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

- **Solvers:** sparse P1 FEM with consistent mass/forcing integration; nodal coefficient averages in stiffness and true cellwise interface coefficients. Heat/reaction–diffusion use implicit Euler; wave/damped wave use average-acceleration Newmark and retain u/v. Configured forcing may be `f(x)*cos(2*pi*frequency*t)`. Dynamic times include 0 and `--time-end` (default 1); optional NS uses mixed P2/P1 FEM as described above.
- **Nonlinear limits:** semilinear cubic reaction is mass-lumped and solved by damped Newton. The p-Laplacian uses `(epsilon^2+|grad u|^2)^((p-2)/2)` with configurable epsilon (default 1e-4); p=2 is exactly Poisson. Eikonal is boundary-segment distance divided by **constant** positive speed. General varying-speed Eikonal, Monge–Ampère, nonlinear heat, capillarity and nonzero/Neumann boundaries remain unsupported.
- **Coefficients/geometry:** configured smooth coefficients use `a0*(1+contrast*tanh(g))`, with contrast < .95 for positivity; the original unconfigured semilinear generator retains its bounded logistic field. Interfaces are two-valued and mesh-aligned at the mid-x threshold. ADR uses constant advection without stabilization; choose diffusion-dominated regimes. Curved boundaries are polygonal approximations; hole centers and crescent construction are fixed. Resolve narrow features adequately.
- **Conditioning:** known values and masks are separate, including known zero. Forward gives full physical inputs; Inverse hides the tier's unknowns; Hybrid reveals configured samples. Known basis definitions do not reveal amplitudes. For legacy four-basis Poisson, full-rank Hybrid source rows determine the entire source: nearly zero prior/Bayes/MAP error is legitimate. This need not remain true for larger K or fewer revealed nodes.
- **Models:** shared mesh/PDE/observation encoders feed A's attention-memory decoder or C's Fourier-coordinate FiLM field. Cases have variable node/basis counts and accumulate gradients sequentially within batches. Optional physics losses are soft constraints; unknown forcing/operator diagnostics are omitted rather than evaluated with hidden labels.
- **Reference accuracy:** fields are numerical solutions, not random heatmaps. Queries use actual-triangle P1 interpolation and piecewise-linear physical-time interpolation; stationary time is zero. Coarse stored frames can smooth fast waves and make finite-difference residuals misleading. Increase spatial/time resolution and check convergence for a new regime. A denser decoder grid alone establishes no discretization-invariance claim.
- **Scientific scope:** deterministic reconstructions, learned flow samples, exact discrete Gaussian Bayes, assumed-kernel GP, training-derived POD and local numerical MAP are different estimators with different information. None of the small example runs establishes useful reconstruction quality or calibrated uncertainty. Existing numerical checks and earlier smoke/audit evidence remain in [VALIDATION.md](VALIDATION.md).
