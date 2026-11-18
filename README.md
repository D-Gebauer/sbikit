# sbikit

Simulation-based inference for cosmology with neural likelihood estimation: compress the data vector, learn the
likelihood of the summaries with a normalising flow, sample posteriors with a batched ensemble MCMC, and check
that they are calibrated.

- **Compression**: MOPED, MSE regression, VMIM, hierarchical VMIM, information-maximising neural networks (IMNN).
- **Neural likelihood estimation**: zuko flows (MAF, NSF and others, built with `sbi`), scalers stored with the
  weights, likelihood ensembles.
- **Sampling**: posteriors of hundreds of observations or test simulations in one batched `torchemcee` ensemble on
  CPU or GPU, any torchemcee move, per-target priors, stopped by convergence checks (split R-hat, ESS,
  autocorrelation time).
- **Metrics**: TARP (identical to the `tarp` package), bias and confidence scalars with bootstrap errors,
  marginal coverage, figures of merit, importance reweighting between priors.
- **Hyperparameter searches**: Optuna studies for compressors and flows, by validation loss, by the figure of
  merit subject to calibration constraints, or by your own objective.
- **Utilities**: priors with density terms and support constraints, parameter transforms, parameter-level
  blinding, checkpoints that load with `weights_only=True`, run records.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Conventions](#conventions)
- [User guide](#user-guide)
  - [Data containers and splits](#data-containers-and-splits)
  - [Compression](#compression)
  - [Neural likelihood estimation](#neural-likelihood-estimation)
  - [Priors](#priors)
  - [Sampling](#sampling)
  - [Parameter transforms](#parameter-transforms)
  - [Metrics](#metrics)
  - [Hyperparameter searches](#hyperparameter-searches)
  - [Blinding](#blinding)
  - [Saving, loading and run records](#saving-loading-and-run-records)
- [Extending sbikit](#extending-sbikit)
- [Notes on methods](#notes-on-methods)
- [Tests](#tests)
- [References](#references)
- [License](#license)

Changes between versions are listed in [CHANGELOG.md](CHANGELOG.md).

## Installation

Python 3.11 or newer.

```bash
pip install git+https://github.com/D-Gebauer/sbikit
```

or, to work on the code and run the tests and the example notebook:

```bash
git clone https://github.com/D-Gebauer/sbikit
cd sbikit
pip install -e ".[test,examples]"
```

PyTorch is installed as a dependency of `sbi`. To get a particular build (CPU-only, or a specific CUDA version),
install `torch` first from the matching index (see [pytorch.org](https://pytorch.org/get-started/locally/)), then
sbikit.

Dependencies: `numpy >= 2.0`, `sbi 0.27`, `zuko 1.6`, `optuna >= 4.0` and
[`torchemcee`](https://github.com/D-Gebauer/torchemcee) `>= 0.1.1` (install from GitHub). The `sbi` and `zuko`
versions are pinned because saved flows depend on their layer names.

## Quick start

[`examples/quickstart.ipynb`](examples/quickstart.ipynb) runs the whole chain on a toy problem, with plots.

The shortest path from simulations to posteriors, with every setting at its default:

```python
import sbikit as sk

nle = sk.NLE()                                   # theta (N, P) parameters, s (N, n) summaries
nle.fit(s, theta)                                # holds out 10 % for validation
prior = sk.Prior.box_from_data(theta)
post = sk.sample_posteriors(nle, s_obs, prior)   # (B, 2000, P) samples for the B rows of s_obs
print(post)                                      # Posteriors(targets=..., converged=.../...)
```

A full analysis, with a compression step, your own splits and a calibration test:

```python
import sbikit as sk
from sbikit.compress import MOPED
from sbikit.metrics import evaluate

# simulations: theta (N, P) parameters, x (N, D) data vectors
ds = sk.Dataset(theta, x, param_names=["Om", "S8", "w0"])
splits = sk.resolve_splits(ds)                       # random 80/10/10, or pass your own
tr, va, te = splits["train"], splits["val"], splits["test"]

# compression: MOPED from fiducial simulations and seed-matched derivative simulations
fid = sk.FiducialSet(x_fid, ["Om", "S8", "w0"], theta_fid, x_minus, x_plus, delta)
moped = MOPED(fid)
moped.fit()
s = moped(ds.x)                                      # (N, 3) summaries

# likelihood p(s | theta)
nle = sk.NLE(flow={"model": "nsf", "hidden_features": 64}, device="auto", seed=0)
nle.fit(s[tr], ds.theta[tr], s[va], ds.theta[va])

# posteriors of all test simulations in one batched ensemble
prior = sk.Prior.box_from_data(ds.theta, names=ds.param_names)
post = sk.sample_posteriors(nle, s[te], prior, sk.MCMCConfig(nwalkers=256),
                            init_centre=ds.theta[te], init_sd=ds.theta[tr].std(0))
assert post.converged.all()

# calibration and constraining power
record, arrays = evaluate(post.samples, ds.theta[te], ds.param_names,
                          fom_sets={"om_s8": ["Om", "S8"]}, tarp_sets={"om_s8": ["Om", "S8"]})

# posterior of an observation
obs = sk.sample_posteriors(nle, moped(x_obs), prior, init_centre=theta_fid)

sk.save(nle, "nle.pt")
sk.run_record("run.json", config={"flow": "nsf"}, splits=splits, extra={"metrics": record})
```

## Conventions

Everything below is importable from `sbikit` (conventionally `import sbikit as sk`) or its submodules, and every
public class and function has a docstring with its arguments and shapes.

- **Arrays.** Data arguments take numpy arrays, lists or torch tensors (on any device). Rows are simulations:
  parameters `(N, P)`, data vectors `(N, D)`, summaries `(N, n)`. Results are numpy arrays, except where a tensor
  is the natural type (`NLE.log_prob`, `NLE.log_likelihood`, `Prior.sample`, and `Prior.log_prob` when given a
  tensor). Wrong shapes, mismatched lengths and NaN or inf in training data raise a `ValueError`.
- **Settings.** Every `config`, `flow`, `train` or `mcmc` argument takes `None` (defaults), a dict or the config
  object (`FlowConfig`, `TrainConfig`, `NetConfig`, `IMNNConfig`, `MCMCConfig`). Unknown keys raise a `TypeError`
  with the closest field name, invalid values a `ValueError`.
- **Validation sets are optional.** `fit(x, theta)` holds out `val_fraction` (default 0.1) of the simulations at
  random; `fit(x, theta, x_val, theta_val)` uses yours. Pass your own whenever simulations are correlated (see
  [groups](#data-containers-and-splits)).
- **Seeds.** `seed=` fixes weight initialisation, batch order, hold-out split and sampling, so the same seed and
  data give the same result. A seeded call leaves torch's and numpy's global random state as it was. With
  `seed=None`, training draws from torch's global state.
- **Devices.** `device` is `"cpu"` (default), `"cuda"`, `"cuda:1"`, `"mps"` or `"auto"` (GPU if there is one).
  `sk.resolve_device("auto")` returns the choice.
- **Fitted or not.** Using a model before `fit` raises `sk.NotFittedError`. A fit that never reaches a finite
  validation loss returns `inf`, warns and leaves the model unfitted.
- **Messages.** Summaries go through the `sbikit` logger; `sk.set_verbosity("warning")` silences them,
  `"debug"` shows more. Things that need attention (chains that did not converge, a failed fit, dropped
  parameters) are Python warnings.
- **Callbacks.** Training loops and the sampler take `callback=`, called after every epoch or convergence check
  with a dict of the current state; returning `True` stops the run and keeps the best result so far.

## User guide

### Data containers and splits

| object | holds |
|---|---|
| `Dataset(theta, x, param_names, groups=, blocks=, meta=)` | simulations: parameters `(N, P)`, data vectors `(N, D)`; optional group ids, named column blocks of `x`, and per-simulation arrays that are not parameters |
| `FiducialSet(x, param_names, theta_fid, x_minus, x_plus, delta, dmu=)` | simulations at a fiducial point `(n_fid, D)` and seed-matched simulations at `theta_fid ± step` `(K, n_d, D)`, or the mean derivatives `dmu` `(K, D)` directly |
| `Splits({"train": idx, ...})` | named index arrays (integer indices or boolean masks; stored sorted), with `check`, `hash`, `save`, `load` |

**Splits are your decision.** `resolve_splits(dataset, splits)` checks the splits you pass (indices in range, no
overlap). Without `splits` it draws random ones by fraction (`fractions={"train": 0.8, "val": 0.1, "test": 0.1}`)
and raises if one of them would be empty.

**Groups.** Simulations that are correlated, such as several footprints or noise realisations cut from one N-body
run, must not sit on both sides of a split. Give them a common id in `Dataset(groups=...)`: random splits then keep
each group together, and splits you pass raise an error if they separate a group (`group_strict=False` turns that
off). `FiducialSet.split(groups_fid, groups_deriv, val_groups)` does the same for fiducial and derivative sets.

**Which parameters are labels.** `Dataset.theta` holds the parameters the flow is conditioned on. Parameters that
vary in the simulations but are left out of `theta` are marginalised implicitly; keep them in `meta` if you want
to look at them later.

Both containers reject NaN and inf (`Dataset(..., allow_nonfinite=True)` keeps such rows for you to filter).
`Dataset.x` is stored as float32, `Dataset.theta` as float64.

### Compression

All compressors map `(N, D)` data vectors to `(N, n_out)` summaries (one `(D,)` vector to `(n_out,)`) and share
one interface: `fit(...)` returns a validation loss, calling the object compresses, `sk.save` / `sk.load` store
and restore it.

| class | what it does | trained on |
|---|---|---|
| `MOPED(fiducial, params=, hartlap=True, pivot_scale=, cov=)` | linear compression `b = C⁻¹ ∂μ/∂θ`, Gram-Schmidt orthogonalised; one summary per parameter; degenerate parameters are dropped with a warning | `FiducialSet` |
| `IMNN(fiducial, config, validation=)` | network that maximises the Fisher information at the fiducial point; output is the quasi maximum-likelihood estimate of the parameters | `FiducialSet` with derivative simulations |
| `MSECompressor(n_out, config)` | network regressing the parameters (posterior-mean estimate); the summaries are the z-scored estimates, `predict(x)` returns them in parameter units | `(x, theta)` |
| `VMIMCompressor(n_out, config)` | network trained with a mixture-density head to maximise the mutual information with the parameters | `(x, theta)` |
| `HierarchicalVMIM(blocks, anchor, orders, n_out)` | VMIM stages over blocks of the data vector, each conditioned on the earlier ones, combined by a final VMIM | `(x, theta)` |

```python
from sbikit.compress import MOPED, IMNN, MSECompressor, VMIMCompressor, HierarchicalVMIM

vmim = VMIMCompressor(n_out=3, config={"widths": (256, 128, 64), "epochs": 300}, device="auto", seed=0)
val_loss = vmim.fit(x[tr], theta[tr], x[va], theta[va])      # or vmim.fit(x, theta)
s = vmim(x)

blocks = {"xi_plus": range(0, 20), "xi_minus": range(20, 40), "peaks": range(40, 55)}
h = HierarchicalVMIM(blocks, anchor=["xi_plus"], orders=[["xi_minus", "peaks"], ["peaks", "xi_minus"]], n_out=3)
h.fit(x[tr], theta[tr], x[va], theta[va])

trn, val = fid.split(run_id_fid, run_id_deriv, val_groups=[0, 1])
imnn = IMNN(trn, {"patience": 250, "min_iterations": 1000}, validation=val)
imnn.fit()
```

Details worth knowing:

- **MOPED** can compress for parameters the flow is not conditioned on (nuisance parameters that only have
  derivative simulations). `pivot_scale={name: prior width}` orders the Gram-Schmidt by how well each direction is
  measured instead of by the order of `params`. `MOPED.from_vectors(vectors, params)` wraps a precomputed
  compression matrix, so that an SBI and a likelihood analysis can share their summaries. The covariance is the
  sample covariance of the fiducial simulations, which needs more simulations than data-vector entries
  (`n_fid > D + 2` with the Hartlap factor; `fit` raises otherwise), or the matrix you pass as `cov=`.
- **Network compressors** (`NetConfig`): LeakyReLU MLP with a linear skip connection, AdamW with a one-cycle
  schedule, early stopping, inputs scaled by median/MAD + asinh (robust against the occasional extreme
  simulation). `vicreg > 0` adds VICReg variance and covariance terms to the VMIM training loss. `optimizer`,
  `schedule`, `input_scaling` and `network` select alternatives by name.
- **Atoms.** If part of the training set sits exactly on one parameter value (ΛCDM simulations at `w0 = -1` among
  wCDM ones, say), the density head of a VMIM can place a near-delta spike there, and the loss then rewards the
  spike instead of information. `fit(..., atoms=[(column, value, half_width)])` spreads those labels uniformly over
  `value ± half_width` during training. Keep such rows out of the validation set as well.
- **Hierarchical VMIM**: `configs={"default": {...}, "anchor": {...}, "final": {...}, "<block>": {...}}` sets a
  `NetConfig` per stage; a stage's entry replaces the default as a whole. The final stage uses the VICReg terms
  only if its config sets `vicreg > 0`.
- **IMNN** follows the authors' reference implementation, `imnn` 0.3.2 (`NumericalGradientIMNN`): the same loss,
  regularisation, early stopping and estimator, and the network of its examples. The loss, Fisher matrix and
  estimator are tested against a numpy transcription of that code. Pass a separate validation `FiducialSet`.
  Minibatches, dropout, weight decay, gradient clipping and input scaling are available and off by default; since
  the reference does not scale its inputs, `fit` warns when the data vector is far from order one
  (`input_scaling="zscore"` fixes that).

### Neural likelihood estimation

```python
nle = sk.NLE(flow={"model": "nsf", "hidden_features": 64, "num_transforms": 5}, device="auto", seed=0)
val_loss, epochs = nle.fit(s[tr], theta[tr], s[va], theta[va], train={"lr": 1e-3, "batch_size": 128})
logp = nle.log_likelihood(s[te], theta[te])           # (n_test,) tensor
s_new = nle.sample(theta[te], n=10)                   # (10, n_test, n) summaries drawn from the flow
```

`FlowConfig` sets the architecture: `model` (`"nsf"`, `"maf"`, and the other zuko flows `sbi` builds: `"naf"`,
`"unaf"`, `"nice"`, `"gf"`, `"bpf"`, `"ncsf"`, `"sospf"`), `hidden_features`, `num_blocks`, `num_transforms`,
`num_bins`, and `kwargs` passed on to the builder. `TrainConfig` sets the training: learning rate, batch size,
`optimizer` (`"adam"`, `"adamw"`), `scheduler` (`"plateau"`, `"none"`), early stopping, gradient clipping,
`val_fraction`.

- Summaries and parameters are z-scored inside the `NLE` (`s_scaler=`, `t_scaler=` choose another scaler), and the
  scalers are stored with the weights. `log_prob` and all losses are densities of the raw summaries.
- Every `fit` starts from freshly initialised weights; `fit(..., warm_start=True)` continues from the current ones.
- `fit(..., optimizer=lambda params: ..., scheduler=lambda opt: ..., callback=...)` replace the optimizer and
  scheduler and hook into the loop; `nle.history` and `nle.train_history` hold the losses per epoch.
- `sk.Ensemble([nle_1, nle_2, ...], weights=None)` is the (weighted) mean likelihood of several flows and can be
  used wherever an `NLE` can.

### Priors

A `Prior` is a box, times optional density terms on subsets of the parameters, times optional support constraints.
Parameters without a term are uniform over the box.

```python
import numpy as np
from sbikit import Prior, Gaussian, KDE, Polygon2D, CurveBound2D

prior = Prior(
    low=[0.2, 0.6, -1.5, -0.1], high=[0.4, 1.0, -0.5, 0.1], names=["Om", "s8", "w0", "dz"],
    terms=[Gaussian(dims=["dz"], mean=[0.0], cov=[0.02 ** 2])],            # Gaussian prior on dz
    constraints=[
        # a band along the Om-s8 degeneracy
        Polygon2D(dims=["Om", "s8"], vertices=[[0.2, 1.0], [0.25, 1.0], [0.4, 0.75],
                                               [0.4, 0.6], [0.35, 0.6], [0.2, 0.85]]),
        # w0 above a curve in Om
        CurveBound2D.from_function(dims=["Om", "w0"], fn=lambda om: -0.7 - np.sqrt(om), x_min=0.2, x_max=0.4),
    ],
)
prior.log_prob(points)             # -inf outside the support; arrays in, arrays out (tensors likewise)
prior.sample(1000, seed=0)         # (1000, 4) tensor
```

- `dims` are parameter names (if the prior has `names`) or column indices.
- Give every parameter finite box bounds: by default the sampler works in the logit of the box. For a Gaussian
  parameter use, e.g., mean ± 5σ. Infinite bounds are allowed for parameters with a density term, together with
  `MCMCConfig(logit_box=False)`.
- `KDE(dims, centres)` is a Gaussian kernel density with a leave-one-out bandwidth: use it when the simulations
  follow a design that is not uniform, to sample with the design density as the prior.
- Constraints are callables `theta (..., P) -> bool (...)`; they must index the last axis only. Two are built in:
  `Polygon2D`, a convex polygon in the plane of two parameters (vertices in order around it), for straight edges;
  and `CurveBound2D`, which keeps one parameter above or below a curve in another, given as nodes or tabulated
  from a function with `from_function`. A plain function works for sampling; to save a prior with your own term or
  constraint, [register](#extending-sbikit) its class.
- `Prior.box_from_data(theta, pad=0.0)` takes the box from the range of the simulations.

### Sampling

```python
config = sk.MCMCConfig(nwalkers=256, moves="de_snooker", rhat=1.01, min_ess=2000, n_samples=2000)
post = sk.sample_posteriors(nle, s_obs, prior, config, init_centre=centres, init_sd=theta[tr].std(0))

post                # Posteriors(targets=20, samples=2000, parameters=3, converged=20/20)
post.samples        # (B, n_samples, P)
post.converged      # (B,) bool
post.diagnostics    # per target: rhat_max, ess_min, tau_max, acceptance; per parameter: rhat, ess, tau; steps, seconds
post.mean(), post.std(), post.quantile([0.16, 0.84]), post.summary()
post.save("posteriors.npz"); post = sk.Posteriors.load("posteriors.npz")
```

- **Batched.** Every row of `s_obs` is a target with its own walkers; one step evaluates the flow once for all
  targets. `prior` can be a list with one prior per target. `targets_per_batch` splits very large sets.
- **Logit box.** By default walkers move in `logit((θ − low)/(high − low))`, so they cannot leave the prior box and
  posteriors that pile up at a bound still mix.
- **Moves.** A preset (`"de_snooker"` (default), `"de"`, `"stretch"`, `"walk"`, `"kde"`), a torchemcee `Move`
  object, or a weighted list of names and objects, e.g. `[["de", 0.8], ["gaussian", 0.2, {"cov": 0.01}]]`.
- **Convergence.** `burn_in` steps are discarded. The chain then grows in blocks until, on its second half, every
  target has grouped split R-hat below `rhat`, at least `min_ess` effective samples and a length of `min_tau`
  autocorrelation times in every parameter, or until `max_steps` steps after burn-in. Samples come from that
  second half. Targets that did not converge are returned too, flagged in `post.converged`, with a warning.
- **Initialisation.** With `init_centre`, walkers start in a small ball around it (the truths for test
  simulations; a fiducial point or best fit for data; one `(P,)` point serves all targets). Without it each
  target starts from the `nwalkers` best of 20 × `nwalkers` prior draws, which costs 20 extra likelihood
  evaluations of the ensemble. `init_walkers=` sets the start positions directly.
- **More control.** `keep_chain=True` also returns the thinned chain and its log posterior (`post.chain`,
  `post.log_prob`), from which a run can be continued (`init_walkers=post.chain[-1]`). `callback=` sees the
  diagnostics after every check and can stop the run. Any object with `log_prob(input, condition)` in `sbi`'s
  layout can serve as the likelihood.

### Parameter transforms

The flow is conditioned on the parameters `θ` of the training simulations. To infer other parameters `ψ` with
`θ = f(ψ)`, pass `transform=`: the sampler then moves in `ψ`, the prior is on `ψ`, and `post.derived` holds `f` of
the samples. No retraining is needed.

```python
import torch
from sbikit import Columns

# flow conditioned on (Om, sigma8); sample (Om, S8)
to_sigma8 = Columns(["Om", "S8"], keep=["Om"], derived=["sigma8"],
                    fn=lambda psi: psi[..., 1:2] / torch.sqrt(psi[..., 0:1] / 0.3))
post = sk.sample_posteriors(nle, s_obs, prior_Om_S8, transform=to_sigma8, init_centre=centre)
```

The same mechanism fits a model with fewer parameters than the flow has labels, for example a two-parameter
redshift evolution of an amplitude that the simulations vary freely per tomographic bin. Any callable on
`(..., P)` tensors works as a transform.

### Metrics

`sbikit.metrics` works on `samples (n_sims, n_samples, P)` and `truth (n_sims, P)`.

```python
from sbikit.metrics import evaluate
record, arrays = evaluate(post.samples, theta[te], names,
                          fom_sets={"om_s8": ["Om", "S8"]}, tarp_sets={"om_s8": ["Om", "S8"]})
```

`record` is a JSON-serialisable dict (numbers that are undefined are `None`):

| entry | content |
|---|---|
| `tarp[set]` | `bias` (negative: coverage curve below the diagonal), `confidence` (negative: overconfident), their bootstrap errors, `max_dev`; for `"all"` parameters and each named set |
| `marginal[param]` | fraction of truths in the central 68 % and 95 % intervals with its binomial error, KS distance of the rank statistic and its p-value |
| `fom[set]` | mean of `log FoM = −½ log det Cov` over the simulations, its geometric-mean FoM |
| `z_mean`, `z_rms`, `std_median` | per-parameter z-scores of the posterior mean (`z_mean_sd`: the error of `z_mean`) and median posterior width |

`arrays` holds the TARP curves and per-simulation values. Keyword options: `weights` (importance weights for the
summaries and FoMs), `levels` (coverage levels), `bounds` (see below), `drop_nonfinite` and
`extra={"name": fn(samples, truth)}` for your own statistics in the record. The building blocks are available
separately: `tarp_f`, `ecp_curve`, `tarp_scalars`, `marginal_coverage`, `posterior_summary`, `log_fom`,
`effective_sample_size`, and `importance_weights(samples, from_prior, to_prior)` to turn posteriors under the
sampling prior into posteriors under another one.

Two things to know when reading the record. The TARP `bias` is the position of the coverage curve, not a shift of
the posterior means; a systematic shift shows up in `z_mean`. And TARP scales the parameters by the range of the
test truths: for small test sets, or parameters with the same truth in every simulation, pass the prior box as
`bounds=(low, high)`.

Using the same record for every run is what makes compressions, flows and analyses comparable.

### Hyperparameter searches

```python
from sbikit.hpt import tune_nle, tune_compressor, EvalSet, best_trials, load_trial, SPACES
from sbikit.compress import VMIMCompressor

# flows, by validation loss; without a storage file the study lives in memory
study = tune_nle(s[tr], theta[tr], s[va], theta[va], n_trials=50, storage="hpt/study.log", name="nle",
                 outdir="hpt", device="auto")
best = load_trial(study, outdir="hpt")              # the best flow

# flows, by figure of merit subject to calibration: needs simulations with known truths
ev = EvalSet(s[te], theta[te], prior, names)
study = tune_nle(s[tr], theta[tr], s[va], theta[va], n_trials=50, storage="hpt/study.log", name="nle_fom",
                 outdir="hpt", objective="fom_tarp", evaluation=ev, fom_params=["Om", "S8"], device="auto")
ensemble = sk.Ensemble([load_trial(study, t.number, "hpt") for t in best_trials(study, k=3)])

# compressors
cfg, vmim, study = tune_compressor(lambda c: VMIMCompressor(3, c, "auto"), SPACES["vmim"],
                                   x[tr], theta[tr], x[va], theta[va], n_trials=40,
                                   storage="hpt/comp.log", name="vmim", outdir="hpt")
```

- A search space is a dict `name -> fixed value | {"choices": [...]} | {"low", "high", "log", "int", "step"}`;
  `SPACES` has defaults for `"nle"`, `"mse"`, `"vmim"` and `"imnn"`. Keys that no config has, and values a config
  rejects, raise before the first trial.
- `objective="fom_tarp"` samples the posteriors of the evaluation set for each trial and maximises the mean log
  figure of merit, with the TARP bias and confidence (on all parameters and on `fom_params`) and the convergence of
  every chain as constraints. Flows with a poor validation loss are pruned before the MCMC. `objective` can also
  be your own function `(nle, trial) -> value` or `-> (value, constraints)`.
- `storage` is `None` (in memory), a file (`*.db`: sqlite; anything else: an Optuna journal file, which several
  processes can share), a storage URL or an Optuna storage object. A search called again on an existing study
  resumes it. Give parallel workers different `seed` values.
- With `outdir`, every trial's model and a JSON record are written to `outdir/name/trials/`. `load_trial` returns
  a trial's model from memory or from there.
- `tune_compressor(..., fixed={...})` trains one given configuration (to retrain a tuned stage with another seed,
  say); `sampler=` and `catch=` are passed to Optuna; `fit_kwargs=` to `compressor.fit`.

### Blinding

```python
from sbikit.blinding import make_key, Blinder

make_key("blind.key", {"S8": 0.05, "w0": 0.3})        # once; never overwritten; keep it out of version control
blinder = Blinder("blind.key")
blinded = blinder.blind(post.samples, ["Om", "S8", "w0"])   # shifts S8 and w0 by hidden offsets
blinder.fingerprint                                    # store with results to show which key was used
```

The offset of each parameter is drawn from `scale × U(−1, 1)` with the seed in the key file. `names` gives the
parameter of every column. If a parameter of the key is missing from `names` (a different spelling, say), `blind`
raises instead of leaving it unblinded; pass `missing="ignore"` for arrays that hold only some of the blinded
parameters. Blind only the posteriors of the data; validation on simulations needs no blinding. Derived
parameters are not blinded with their inputs: recompute them from the blinded samples.

### Saving, loading and run records

```python
sk.save(obj, "file.pt", meta={"note": "run 3"})   # NLE, Ensemble, any compressor, Prior, scaler
obj = sk.load("file.pt", device="auto")
sk.read_meta("file.pt")                           # kind, versions it was written with, your meta
sk.run_record("run.json", config=config, data_files=["data.npz"], splits=splits, repo=".")
```

Checkpoints contain only tensors and plain Python values, so they load with `torch.load(..., weights_only=True)`,
and they record the versions of sbikit and its dependencies. `run_record` writes a JSON file with the
configuration, versions, git hashes of the analysis repository and of sbikit (if it runs from a checkout), md5
sums of the data files, the hash of the splits, the host and the GPU. `Posteriors` and `Splits` have their own
`save` / `load` (`.npz`).

## Extending sbikit

Built-in parts are chosen by name; your own are added under a name with a `register_*` function (usable as a
decorator) and then behave like the built-in ones, including `sk.save` / `sk.load`. Import the module that
registers them before loading a checkpoint that uses them.

| to add | register with | then select by |
|---|---|---|
| a flow (any conditional density estimator) | `sk.register_flow(name, builder)` | `FlowConfig(model=name, kwargs={...})` |
| a compression network (a CNN on maps, ...) | `sbikit.compress.register_network(name, factory)` | `NetConfig(network=name, network_kwargs={...})`, `IMNNConfig(network=name)` |
| a scaler | `sk.register_scaler(cls)` | `NetConfig(input_scaling=name)`, `NLE(s_scaler=name)` |
| a prior density term / constraint | `sk.register_term(cls)` / `sk.register_constraint(cls)` | `Prior(terms=[...], constraints=[...])` |
| an MCMC move | `sk.register_move(name, cls)` | `MCMCConfig(moves=[[name, weight, {...}]])` |
| a compressor | `sbikit.compress.register(cls)` on a `Compressor` subclass | use the class |
| any other saveable object | `sbikit.io.register_loader(kind, fn)` | `sk.save` / `sk.load` |

```python
import torch.nn as nn
from sbi.neural_nets.net_builders import build_zuko_maf
from sbikit.compress import VMIMCompressor, register_network

@sk.register_flow("wide_maf")
def wide_maf(batch_x, batch_theta, config):        # the tensors only carry the dimensions
    return build_zuko_maf(batch_x, batch_theta, z_score_x="none", z_score_y="none",
                          hidden_features=[256] * 3, num_transforms=config.num_transforms, **config.kwargs)

nle = sk.NLE(flow={"model": "wide_maf", "num_transforms": 4})

@register_network("tanh_mlp")
def tanh_mlp(n_in, n_out, config):                 # config is the NetConfig; its network_kwargs are yours
    width = config.network_kwargs.get("width", 64)
    return nn.Sequential(nn.Linear(n_in, width), nn.Tanh(), nn.Linear(width, n_out))

vmim = VMIMCompressor(3, {"network": "tanh_mlp", "network_kwargs": {"width": 128}})
```

For a one-off experiment nothing needs registering: `sk.NLE(flow=my_builder)` or `sk.NLE(flow=my_estimator)`,
`VMIMCompressor(3, net=lambda n_in, n_out: my_module)`, a plain function as a prior constraint or as a transform,
a torchemcee `Move` object in `MCMCConfig(moves=...)`. Such objects train and sample normally; only saving them
needs the registered form, and `sk.save` says so. The docstring of each `register_*` function describes what the
registered object must provide.

## Notes on methods

**Likelihood values.** `NLE.log_prob` is the density of the raw summaries: the Jacobian of the summary scaler is
included. Validation losses of flows with different scalers, and the members of an `Ensemble`, are therefore on
one scale.

## Tests

```bash
python -m pytest                    # 307 tests on the CPU
python -m pytest --device cuda      # the same tests, training and sampling on the GPU
python -m pytest -m "not slow"      # skip the longest end-to-end tests
```

The tests check, among other things: priors against analytic densities; MOPED being lossless for a linear Gaussian
model; the IMNN against the reference formulas and the analytic Fisher information; batched posteriors against
analytic Gaussian posteriors; the width of a flat box for every move; the flow learning a linear Gaussian
likelihood; TARP against the `tarp` package; the hyperparameter objectives end to end; save/load round trips
of every object; that checkpoints written by version 0.1.0 still load and give the same outputs; that a seed
reproduces every training and sampling step without touching the global random state; and the error raised for
each kind of wrong input.

## References

- Heavens, Jimenez & Lahav (2000), *Massive lossless data compression and multiple parameter estimation from galaxy spectra* (MOPED)
- Charnock, Lavaux & Wandelt (2018), *Automatic physical inference with information maximizing neural networks*; Makinen et al. (2021), *Lossless, scalable implicit likelihood inference for cosmological fields* (IMNN)
- Jeffrey, Alsing & Lanusse (2021), *Likelihood-free inference with neural compression of DES SV weak lensing map statistics* (VMIM); Williamson et al. (2026) (hierarchical VMIM)
- Bardes, Ponce & LeCun (2022), *VICReg: Variance-invariance-covariance regularization for self-supervised learning*
- Papamakarios et al. (2019), *Sequential neural likelihood*; Alsing et al. (2019), *Fast likelihood-free cosmology with neural density estimators and active learning* (neural likelihood estimation)
- Papamakarios, Pavlakou & Murray (2017), *Masked autoregressive flow for density estimation*; Durkan et al. (2019), *Neural spline flows*
- Foreman-Mackey et al. (2013), *emcee: The MCMC hammer*; ter Braak & Vrugt (2008), *Differential Evolution Markov Chain with snooker updater and fewer chains*
- Lemos et al. (2023), *Sampling-based accuracy testing of posterior estimators for general inference* (TARP)
- Hartlap, Simon & Schneider (2007), *Why your model parameter confidences might be too optimistic*
- Akiba et al. (2019), *Optuna: A next-generation hyperparameter optimization framework*
- Tejero-Cantero et al. (2020), *sbi: A toolkit for simulation-based inference*; Rozet et al., *zuko: Normalizing flows in PyTorch*

## License

MIT, see [LICENSE](LICENSE).
