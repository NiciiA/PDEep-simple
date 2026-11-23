# Executed validation

The initial section records the deterministic implementation on 19 November 2026. The 20 November research extensions and their checks are recorded at the end.

All results below were actually produced locally using `/Users/nicolasackermann/anaconda3/envs/my_env/bin/python`, on CPU with two PyTorch threads. Python 3.9, NumPy 1.26.4, SciPy 1.12.0, PyTorch 2.2.2 and Matplotlib 3.8.2 were available. CUDA/MPS execution and large-preset training were not tested. Commands are in `README.md`; raw artifacts remain in `data/` and `runs/`.

## Numerical checks

`python -m pdeep_simple check` reported **14 PASS, 0 WARN, 0 FAIL**. Full numbers: `runs/checks.json`.

| Check | Actual result |
|---|---:|
| Square n=8 | 81 nodes, 128 triangles, area 1, minimum area 0.0078125, 32 boundary nodes |
| Disk n=8 | 129 nodes, 224 triangles, polygon area 0.7803612881, minimum area 0.0015241431, 32 boundary nodes |
| L-shape n=8 | 65 nodes, 96 triangles, area 0.75, minimum area 0.0078125, 32 boundary nodes |
| Poisson relative L2, n=8 | 0.0377300268 |
| Poisson relative L2, n=16 | 0.0096413510 |
| Heat error at t=0.25 / 0.5 / 1 | 0.00242233 / 0.00285888 / 0.00209337 |
| Wave displacement error at t=0.25 / 0.5 / 1 | 0.00289437 / 0.00764670 / 0.00304650 |
| Wave velocity error at t=0.25 / 0.5 / 1 | 0.00641313 / 0.00528964 / 0.01490632 |
| Unforced Newmark relative energy drift | 7.37e-16 |
| Nonzero initial wave velocity, t=1 displacement error | 0.00351961 |
| T4 nonlinear residual norm | 3.99e-13 (two Newton steps) |
| Affine space/time interpolation max error | 1.11e-16 |

Poisson uses the requested sine manufactured solution. Heat uses alpha=0.1, wave c=0.7, n=16 and internal dt=0.002. Dynamic errors are spatial lumped-L2 errors normalized by the **initial displacement amplitude**, or by omega times that amplitude for velocity. This avoids dividing by a near-zero exact oscillation. They are not timewise relative errors against a vanishing solution.

A separate temporal refinement comparison used six heat and six wave G1 training cases, spanning all geometry families, comparing dt=0.002 with dt=0.001 at all 21 stored times. Maximum trajectory relative differences: heat **0.00334151**, wave displacement **0.00409366**, wave velocity **0.00958063**. See `runs/time_refinement.json`. This checks these sampled cases, not arbitrary future meshes/coefficients/frequencies.

The final T4 test solutions differ from the corresponding variable-coefficient *linear* solves by relative L2 **0.04545 (square), 0.01330 (disk), 0.01167 (L)**. Their Newton residual norms are 1.86e-15, 4.42e-12 and 4.22e-12. See `runs/nonlinear_effect.json`. These are actual nonlinear solves, including the cubic reaction.

## Focused regression tests

`python -m unittest discover -s tests -v`: **6/6 passed**.

- L-shape elements/sample points stay out of the removed quadrant; exterior queries are rejected.
- Mutating hidden coefficients/fields and target solutions leaves assembled inverse model inputs unchanged for T2, G1 wave and T4.
- Stationary reference queries reject nonzero time.
- The Gaussian posterior agrees with a dense one-dimensional numerical posterior integral to nine decimal places.
- A and C predictions are invariant within float32 tolerance to observation permutation and query chunking.
- The variable-coefficient nonlinear weak residual is small when supplied a genuine converged FEM solution.

All seven CLI help pages succeeded. Unsupported T3-wave, T4-Poisson, and Poisson-only kinematic-loss requests failed with useful errors (`runs/cli_checks.log`).

## Generated datasets and inspection

| Directory | Train / validation / test | Total | Failed solves |
|---|---:|---:|---:|
| `data/g1_small` | 72 / 18 / 18 | 108 | 0 |
| `data/t1_small` | 2 / 1 / 1 | 4 | 0 |
| `data/t2_small` | 12 / 6 / 6 | 24 | 0 |
| `data/t3_small` | 6 / 3 / 3 | 12 | 0 |
| `data/t4_nonlinear` | 6 / 3 / 3 | 12 | 0 |

The exact G1 command in the request used workers=auto and completed in 1.48 seconds. Repeating it reused 108 cases, with **all SHA-256 file hashes unchanged** and 108 distinct physical-case seeds. No case is reused across splits. T4 generation was also exercised with two worker processes.

Inspection was executed on every tier. G1 includes six seeded random examples, covering heat/wave and all domain families. PNGs, all-split summaries and distributions are in `runs/inspection`, `runs/t1_inspection`, `runs/t2_inspection`, `runs/t3_inspection`, `runs/t4_inspection`. Representative inspection, loss, field and comparison PNGs were opened for visual review. An overlapping observation colorbar was removed and the inspection images regenerated.

The earlier, milder `data/t4_small` experiment was retained separately, marked generator version 0. It is not the final T4 generator output.

## Neural forward/backward and training

The check command ran each **S-sized** model on square/disk wave cases with observation counts **0 and 7**, computed normalized field + kinematic + boundary loss, backpropagated, checked output shape (8,2) and finite gradients, and performed an Adam step that changed the parameters:

| Model | Combined smoke loss | Finite gradients / changed parameters |
|---|---:|---|
| A-Geo S | 0.72829084 | PASS |
| C-Geo S | 0.79417500 | PASS |

The requested G1 smoke training ran **30 updates, batch 2, 256 queries, all three modes**:

| Run | Model / objective | Last train field | Last validation field | Best validation field | Training time |
|---|---|---:|---:|---:|---:|
| `runs/a_smoke` | A S, field + wave kinematic 0.001 | 0.15997 | 0.99554 | 0.91655 | 5.70 s |
| `runs/c_smoke` | C S, field | 0.20275 | 0.92275 | 0.91178 | 3.27 s |

G1 has no scalar auxiliary targets, so aux=0.1 was explicitly reported as omitted. Validation uses all 18 cases and fixed draws. Short runtimes are measurements from this session, not performance promises.

Both S architectures also completed three-update training plus held-out evaluation for **T1, T2, T3 and T4**, using weak/boundary losses and applicable initial/auxiliary losses. Runs are `runs/t1_a_smoke` through `runs/t4_c_smoke`; the command/output transcript is `runs/tier_smoke.log`. Final validation field losses:

| Tier | A S | C S |
|---|---:|---:|
| T1 | 0.15501 | 0.26938 |
| T2 | 1.01293 | 0.99475 |
| T3 | 1.55716 | 1.48207 |
| T4 | 0.33797 | 0.23840 |

A separate `runs/c_all_smoke` runs the G1 C ALL_001 objective, including weak heat/wave terms, initial and boundary terms. All these are execution/gradient checks, **not converged reconstructions**.

Actual parameter counts:

| Preset | A | C |
|---|---:|---:|
| s | 1,500,035 | 1,312,387 |
| m | 6,478,339 | 5,739,267 |
| l | 27,908,611 | 27,082,243 |
| xl | 62,715,651 | 56,150,787 |
| xxl | 145,024,003 | 133,350,403 |

Large model counts were obtained on PyTorch's meta device; this did not allocate/train those networks.

## Held-out metrics and baselines

Full G1 evaluation ran on all 18 test physical cases, three conditioning modes, counts 0/8/32/128 and five selected times. Both neural models, prior and RBF shared all case/seed/mode/count/quantity/time keys. Raw and grouped CSVs and prediction NPZs are in each run's `evaluation/`, or directly in `runs/g1_baselines`. `plot` generated losses, per-PDE losses, physical-time reconstructions, observation-count/PDE/geometry comparisons and time-error plots. T2 plots include coefficient scatter.

Illustrative **inverse, 32 observations** results (mean per-case physical RMSE; u over 18 cases, v over six wave cases):

| Method | u RMSE | v RMSE |
|---|---:|---:|
| A S, best validation checkpoint | 0.50519 | 5.87254 |
| C S, best validation checkpoint | 0.50427 | 5.83447 |
| Prior mean | 0.49743 | 5.79653 |
| RBF | 0.33511 | 6.12587 |

**Thirty updates did not learn useful inverse reconstructions beyond the prior.** The plots show this plainly. No model-quality or architecture superiority conclusion is supported by this smoke experiment.

T1 analytical Bayes/ridge evaluation ran with 0, 8 and 32 observations on its one held-out case. At 32 observations: prior RMSE 0.0589728, Bayes 0.00341888, ridge 0.00316484, RBF 0.0626428. Posterior covariance and propagated solution standard deviations are saved with the Bayesian prediction NPZs in `runs/t1_baselines`.

T2 prior/RBF/scalar-grid evaluation ran on six held-out cases with eight observations. In inverse mode: prior mean per-case RMSE 0.0528461, grid 0.00443368, RBF 0.270485. The three heat-hybrid grid combinations were explicitly skipped because their initial fields are partly hidden; `runs/t2_baselines/skipped.csv` explains this. The scalar grid was not given hidden test coefficients.

Prior and RBF baselines also completed evaluation for T3 and T4 on all three test geometry families and modes with eight observations (`runs/t3_baselines`, `runs/t4_baselines`). Comparison plots restrict methods to common case/count populations.

## Resume and geometry holdout

`runs/resume_split` ran two updates, resumed to **four total**; `runs/resume_full` ran four uninterrupted updates. All model state tensors were **exactly equal**, not merely close. See `runs/resume_result.json` and `runs/resume_test.log`.

A separate relocated-dataset resume continued into `runs/portable_resume`, honored the new `--data` path, saved both last/best checkpoints in the new directory, and completed evaluation. See `runs/portable_resume_result.json`. This tests path portability on CPU; actual Colab/CUDA execution was not available.

`runs/family_holdout_smoke` used a small manual C configuration for two updates on square/disk families only: **48 train and 12 validation cases**. Evaluation ran on **six L-shaped test cases**, and every metric row was labeled `unseen_family`. This verifies the routing/split behavior; it is not evidence of good family generalization.

## Scope of confidence

The numerical solvers, storage/query pipeline, masks, two distinct architectures, selectable objectives, baselines, portable checkpoint resume and CSV/PNG workflow were exercised. Numerical reference accuracy remains mesh/time dependent. Weak losses are residuals of a finite P1 projection; temporal evaluation diagnostics use finite differences. Larger meshes/presets, CUDA/MPS, meaningful long training, unknown T4 conductivity, other PDE families, probabilistic flows and discretization invariance remain untested or unsupported as described in the README.

## Research extensions - 20 November 2026

The extensions reused the existing datasets. No large dataset generation or long training experiment was launched. Execution remained CPU-only in the same `my_env` environment. The A/C definitions in `models.py` are unchanged; `flow.py` and `comparison.py` are the only new source modules. Existing CLI/model/solver/plotting files were extended, and two focused tests were added to the existing test file.

**Numerics and focused tests:** the existing `check` command again produced 14 PASS, 0 WARN, 0 FAIL (`runs/extension_checks.json`), with the same analytical Poisson/heat/wave errors as the initial implementation. The expanded unittest file has eight passing tests. New assertions cover finite AE and flow gradients, frozen AE gradients at the stage transition, reference-free sampling, nonzero sample variation, both wave output channels, chunk-size consistency for A/C, valid-domain grids and preservation of nonzero P1 input fields under red refinement. The field encoder is explicitly patched to fail if inference calls it in the sampling test.

**Case multiprocessing:** one- and two-worker T1 baseline commands evaluated the same two independent physical cases, all three conditioning modes, counts 0/8, all four applicable baseline methods, and grids 10/16. All **192 metric rows matched exactly**, excluding inference times: maximum numerical difference **0.0**. The training split supplied two available T1 cases for this numerical process-parity check only; this is not a held-out predictive-quality experiment. See `runs/extension_worker_parity.json` and `runs/extension_t1_workers1`, `runs/extension_t1_workers2`. A two-worker T2 case evaluation also completed. Auto-worker evaluation was exercised on T1 and nonlinear T4; it obeyed the CPU/memory cap and printed throughput/ETA. Each run retained per-case CSVs and prediction NPZs plus combined metrics/summary tables.

**Flow execution:** A-flow completed two AE plus two flow updates on T1 in `runs/extension_a_flow`. C-flow completed two AE plus two flow updates on G1, with applicable kinematic/weak/initial/boundary losses during AE, in `runs/extension_c_flow`. Additional three-update train/evaluation checks exercised T2 (A-flow with its coefficient target), T3 (C-flow with its hidden initial field), and T4 (C-flow S, including its actual nonlinear weak residual). The latter used **1,490,435 parameters** including the added field/flow components. See `runs/extension_tier_flow.log` and the `runs/extension_*_flow_t*` directories. These runs checked finite execution, not reconstruction convergence.

**Resume:** a C-flow run trained for three updates, resumed to five total updates, and matched an uninterrupted five-update run **exactly in every model tensor**, including latent normalization buffers. Both ended in the flow stage. See `runs/extension_resume_result.json`, `runs/extension_resume.log`. The checkpoint contains the AE/decoder, conditional velocity MLP, stage, normalization, optimizer and RNG state. Uncomputed stage losses are blank rather than reported as zero; validation sample-mean field loss is distinct from latent FM loss.

**Whole-field and multi-grid consistency:** a saved C-flow generated three complete samples in each conditioning mode, using the same latents at grids 9 and 17. At shared physical coordinates, the maximum difference in an individual sample was **9.3132e-9**. Latent arrays, observation positions/values/noise, and known-field/scalar masks matched exactly between grids. See `runs/extension_grid_consistency.json`, `runs/extension_flow_final`. Separate A/C unit checks compared different chunk sizes and verified positive sample standard deviation; sampling never invoked the reference-field encoder.

**Actual 128/256/512 evaluation:** one held-out T1 case was evaluated with eight observations at all three requested grids, both by exact stored-model Bayes and by a small C-flow (two samples, three Heun steps). All grid points were valid on this square: 16,384 / 65,536 / 262,144. No new PDE reference was generated by increasing the query grid.

| Method | RMSE at 128 | RMSE at 256 | RMSE at 512 |
|---|---:|---:|---:|
| T1 Bayes, stored reference | 0.00603408 | 0.00603432 | 0.00603437 |
| Tiny C-flow smoke checkpoint | 0.06185464 | 0.06185549 | 0.06185570 |

See `runs/extension_bayes_large_grids`, `runs/extension_flow_large_grids`. The tiny flow's mean sample std at grid 512 was only **0.00047414**, and its empirical nominal-90% interval covered only **0.00576** of the domain on this case. With only two samples and a few training updates, this is an execution example, **not useful or calibrated uncertainty**. Neither a two-sample interval nor a single case supports a calibration conclusion. A flow S preset was tested on T4, but large flow presets and GPU execution were not tested.

**Refined numerical references:** an old deterministic A checkpoint evaluated a wave/L-shaped case against one red-refined reference (`runs/extension_refined`). A T2 Poisson case used two refinement levels (`runs/extension_convergence`). Its successive-level field differences were:

| Refined level | Nodes | Triangles | Previous-level RMSE | Relative L2 difference |
|---|---:|---:|---:|---:|
| 1 | 289 | 512 | 0.01610192 | 0.04813080 |
| 2 | 1,089 | 2,048 | 0.00420554 | 0.01253781 |

Only the explicitly limited cases were solved again. Geometry remained the original polygon, stored physical inputs were prolonged as the same P1 functions, and time steps were halved at each level. These are mesh/time discretization differences, not a proof of continuum accuracy. Conditioning and observation draws used the original case. Refined truth was used only for evaluation labels.

**Plots and old interfaces:** same-case comparisons were generated for old deterministic A/C runs plus C-flow on a wave/disk case, in all modes, with u/v separated (`runs/extension_comparison`). A second comparison included deterministic A, C-flow, prior and T1 Bayes on one case (`runs/extension_classical_comparison`). They show reference/predictions/errors/sensors, mode differences and flow reference/mean/std/individual samples/error. Multi-run loss figures have separate deterministic-field, AE-field, FM and flow sample-mean axes. Representative PNGs were opened for visual review. A common-grid deterministic/flow/prior/Bayes sensor comparison was also generated from matched held-out T1 evaluations at grids 12/20 and counts 0/8 (`runs/extension_common_curves`); its axes use physical RMSE. Masked grid arrays were initialized explicitly to remove an irrelevant Matplotlib overflow warning from uninitialized masked storage.

A fresh deterministic A S run without `--flow` also completed two training updates, retaining the original parameter count and CSV loss format (`runs/extension_deterministic`).

The original single-run `plot --run ... --what losses,fields,comparisons --compare ...` path worked with a flow run (`runs/extension_single_plots`), and saved-CSV multi-run curves worked without new inference (`runs/extension_curves`). The seven help pages succeeded; incompatible Poisson kinematic loss and refinement without an explicit case limit were rejected (`runs/extension_cli_checks.log`).

**Current limits:** these checks do not establish long-run reconstruction quality, posterior calibration, unseen-PDE generalization or resolution invariance. Grid quadrature approximates an irregular-domain integral; refined references preserve the original polygon and P1 input functions. Physics penalties train the AE representation; the flow stage optimizes only conditional latent FM. Existing T2 heat-hybrid scalar-grid and underdetermined nonlinear-prior restrictions remain. There is no MCMC, Bayesian calibration procedure, distributed evaluator, automatic interrupted-evaluation resume, deterministic-to-flow checkpoint conversion, or new PDE family in this extension.

## Focused baseline correctness audit

Audited the existing G1 Poisson experiment `data/g1_test`, `runs/a_test3k`, `runs/c_test3k` and `runs/g1_base_test3k`. No training or dataset generation was run. No numerical implementation change was needed, and no existing CSV, checkpoint or PNG was rewritten. Only README clarification, these notes and `runs/baseline_audit.json` were written.

The actual G1 equation has known constant conductivity a=1 and zero Dirichlet boundary data. Its source is a four-function expansion with independent N(0,9) amplitudes; geometry and source basis definitions are known to both A/C and the physical-prior baseline. True amplitudes and source/solution labels are stored for supervision but do not enter inverse prediction.

`prior` is E[u | known physical information], ignoring the solution observation set. Forward supplies the full source and legitimately reproduces the numerical reference with a fresh FEM solve. Hybrid supplies only masked source nodal values, but they usually determine the four amplitudes. On `test_poisson_disk_00000.npz`, Forward knew 289/289 source values, Inverse 0/289, and Hybrid 152/289. The Hybrid constraint matrix had rank 4/4. The reconstructed source differed from the stored full source by relative 3.27e-8; the fresh solution's physical RMSE was 4.79e-9. A solver spy confirmed an actual FEM call in each mode. Inverse correctly solved with zero prior-mean source, not the hidden true source.

Small target-invariance checks kept observation packets, known physical values, basis/prior definitions and masks fixed, then deleted `solution`, `velocity` and `amplitudes` and changed only hidden field values (plus the hidden T2 coefficient label). All tested predictions were **exactly unchanged**, maximum difference 0.0:

- G1 prior/RBF in all three modes on the held-out disk case.
- T1 prior/RBF/ridge/Bayes, T2 prior/RBF/scalar-grid, and T4 prior/RBF in inverse mode on one existing held-out case each.
- Actual A/C 3k checkpoints also produced finite predictions with the reference arrays absent, for zero and eight observations. Their assembled features were unchanged by hidden-label mutations.

The T1 Bayesian path constructs H from numerical PDE basis responses at the supplied sensors, with the matched amplitude prior and noise covariance. No-observation, fully source-hidden T1 gave mean zero and covariance 9 I. RBF reads only supplied sensor coordinates/quantity/value/noise: it returns zero for none, the observed mean for fewer than dimension+1, or its Gaussian RBF fit otherwise. Known source/conductivity/boundary values are not used by RBF. The T2 grid evaluates candidate parameters with allowed known fields; mutating the true unknown coefficient did not affect it. T4 never applies the linear-Gaussian posterior formula.

For the saved 3k evaluations, all **13,824 metric keys** matched between A, C, prior and RBF, spanning 384 held-out cases, three modes, six sensor counts, and aggregate/stationary-time rows. All **162 saved-array comparisons** matched exactly for sensor coordinates, quantities, observed values, mesh, physical times and reference fields. These compare the 54 saved A packets with C, prior and RBF; remaining cases use the same deterministic observation path and matching metric keys. The dataset's 3,840 physical-case seeds were distinct across splits.

All methods used displacement scale **s_u=0.17816505847495698**. Recovering it from every saved RMSE/normalized-RMSE pair differed by at most **2.78e-17**. Saved MSE/RMSE values obeyed RMSE²=MSE. The one-case noise sanity check gave observed-minus-reference-at-sensor RMS 0.007532 for the declared Gaussian standard deviation 0.01 (eight measurements); observation construction was also inspected directly.

Existing prior results, mean per-case errors over the 384 held-out cases (unchanged with sensor count):

| Mode | Physical RMSE | Normalized RMSE | Relative L2 |
|---|---:|---:|---:|
| Forward | 3.8415e-9 | 2.1562e-8 | 2.5808e-8 |
| Inverse | 0.1495328 | 0.8392936 | approximately 1 |
| Hybrid | 4.0273e-9 | 2.2604e-8 | 2.7960e-8 |

The near-zero values reflect legitimate known-physics reconstruction plus stored float32 rounding. The inverse 0.83929 is field RMS divided by a fixed training scale and then averaged per case; it is not relative L2 or a theoretical unit-variance guarantee.

**Verdict:** the recorded A/C/prior/RBF numbers remain valid, with no leakage or metric correction requiring reevaluation. Their interpretation must distinguish an analytic physical-prior/FEM estimator from learned A/C reconstruction and observation-only RBF. Calling this an equal-physics-use or observation-only comparison would be misleading. Existing figures need no numerical regeneration; captions should retain this distinction and respect whether the axis is physical or normalized RMSE. Source code, multiprocessing, CLI options, data formats and existing experiments were left unchanged.
