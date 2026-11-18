# sbikit

Simulation-based inference for cosmology with neural likelihood estimation: compress the data vector, learn the
likelihood of the summaries with a normalising flow, sample posteriors with a batched ensemble MCMC, and test
their calibration.

- **Compression**: MOPED, MSE regression, VMIM, hierarchical VMIM, IMNN.
- **Likelihood**: zuko flows built with `sbi` (MAF, NSF, ...), ensembles of flows.
- **Sampling**: all observations or test simulations in one batched `torchemcee` ensemble on CPU or GPU, stopped
  by convergence checks.
- **Metrics**: TARP, marginal coverage, figures of merit, importance reweighting between priors.
- **Goodness of fit**: posterior maxima, χ² of the learned likelihood, posterior predictive tests calibrated on
  simulations.
- **Hyperparameter searches** with Optuna.
- **Utilities**: priors with density terms and constraints, parameter transforms, blinding, checkpoints, run
  records.

Changes between versions are listed in [CHANGELOG.md](CHANGELOG.md).

## Installation

Python 3.11 or newer.

```bash
pip install git+https://github.com/D-Gebauer/sbikit
```

To work on the code and run the tests and the example notebook:

```bash
git clone https://github.com/D-Gebauer/sbikit
cd sbikit
pip install -e ".[test,examples]"
```

For a particular PyTorch build (CPU-only, a specific CUDA version), install `torch` first.

Dependencies: `numpy >= 2.0`, `sbi 0.27`, `zuko 1.6`, `optuna >= 4.0` and
[`torchemcee`](https://github.com/D-Gebauer/torchemcee) `>= 0.1.1` (install from GitHub). The `sbi` and `zuko`
versions are pinned because saved flows depend on their layer names.

## Quick start

[`examples/quickstart.ipynb`](examples/quickstart.ipynb) runs the whole chain on a toy problem, with plots.

With every setting at its default:

```python
import sbikit as sk

nle = sk.NLE()                                   # theta (N, P) parameters, s (N, n) summaries
nle.fit(s, theta)                                # holds out 10 % for validation
prior = sk.Prior.box_from_data(theta)
post = sk.sample_posteriors(nle, s_obs, prior)   # (B, 2000, P) samples for the B rows of s_obs
```

With compression, your own splits and a calibration test:

```python
import sbikit as sk
from sbikit.compress import MOPED
from sbikit.metrics import evaluate

names = ["a", "b", "c"]
ds = sk.Dataset(theta, x, param_names=names)         # theta (N, P) parameters, x (N, D) data vectors
splits = sk.resolve_splits(ds)                       # random 80/10/10, or pass your own
tr, va, te = splits["train"], splits["val"], splits["test"]

# MOPED from fiducial simulations and seed-matched derivative simulations
fid = sk.FiducialSet(x_fid, names, theta_fid, x_minus, x_plus, delta)
moped = MOPED(fid)
moped.fit()
s = moped(ds.x)                                      # (N, 3) summaries

# likelihood p(s | theta)
nle = sk.NLE(flow={"model": "nsf", "hidden_features": 64}, device="auto", seed=0)
nle.fit(s[tr], ds.theta[tr], s[va], ds.theta[va])

# posteriors of all test simulations
prior = sk.Prior.box_from_data(ds.theta, names=names)
post = sk.sample_posteriors(nle, s[te], prior, sk.MCMCConfig(nwalkers=256),
                            init_centre=ds.theta[te], init_sd=ds.theta[tr].std(0))

# calibration and constraining power
record, arrays = evaluate(post.samples, ds.theta[te], names,
                          fom_sets={"a_b": ["a", "b"]}, tarp_sets={"a_b": ["a", "b"]})

# posterior of an observation
obs = sk.sample_posteriors(nle, moped(x_obs), prior, init_centre=theta_fid)

sk.save(nle, "nle.pt")
sk.run_record("run.json", config={"flow": "nsf"}, splits=splits, extra={"metrics": record})
```

## General behaviour

- Data arguments are numpy arrays, lists or torch tensors with one row per simulation. Results are numpy arrays;
  `NLE.log_prob` and `Prior.sample` return tensors.
- Settings (`config`, `flow`, `train`, `mcmc`) are a dict or the config object (`FlowConfig`, `TrainConfig`,
  `NetConfig`, `IMNNConfig`, `MCMCConfig`). Unknown keys and invalid values raise.
- `fit(x, theta)` holds out `val_fraction` (default 0.1) for validation; `fit(x, theta, x_val, theta_val)` uses
  your validation set.
- `seed=` makes training and sampling reproducible and leaves the global random state unchanged.
- `device` is `"cpu"` (default), `"cuda"`, `"mps"` or `"auto"`.
- Progress is reported through the `sbikit` logger (`sk.set_verbosity("warning")` silences it). Unconverged
  chains and failed fits raise Python warnings.
- Training loops and the sampler take `callback=`, called with the current state after every epoch or convergence
  check; returning `True` stops the run.
- Every public class and function has a docstring with its arguments and shapes.

## User guide

### Data containers and splits

| object | holds |
|---|---|
| `Dataset(theta, x, param_names, groups=, blocks=, meta=)` | parameters `(N, P)` and data vectors `(N, D)`; optional group ids, named column blocks of `x`, extra per-simulation arrays |
| `FiducialSet(x, param_names, theta_fid, x_minus, x_plus, delta, dmu=)` | simulations at a fiducial point `(n_fid, D)` and seed-matched simulations at `theta_fid ± step` `(K, n_d, D)`, or the mean derivatives `dmu` `(K, D)` |
| `Splits({"train": idx, ...})` | named index arrays or boolean masks, with `check`, `hash`, `save`, `load` |

`resolve_splits(dataset, splits)` checks the splits you pass (indices in range, no overlap). Without `splits` it
draws random ones (`fractions={"train": 0.8, "val": 0.1, "test": 0.1}`).

Simulations that are correlated, for example realisations that share initial conditions, should stay on one side
of a split. Give them a common id in `Dataset(groups=...)`: random splits keep each group together, and splits
you pass raise if they separate a group (`group_strict=False` turns that off).
`FiducialSet.split(groups_fid, groups_deriv, val_groups)` does the same for fiducial sets.

### Compression

Compressors map `(N, D)` data vectors to `(N, n_out)` summaries. `fit(...)` returns the validation loss and
calling the object compresses.

| class | method | trained on |
|---|---|---|
| `MOPED(fiducial, params=, hartlap=True, pivot_scale=, cov=)` | linear compression, one summary per parameter | `FiducialSet` |
| `IMNN(fiducial, config, validation=)` | network maximising the Fisher information at the fiducial point | `FiducialSet` with derivative simulations |
| `MSECompressor(n_out, config)` | network regressing the parameters | `(x, theta)` |
| `VMIMCompressor(n_out, config)` | network maximising the mutual information with the parameters | `(x, theta)` |
| `HierarchicalVMIM(blocks, anchor, orders, n_out)` | VMIM stages over blocks of the data vector, each conditioned on the earlier ones | `(x, theta)` |

```python
from sbikit.compress import MOPED, IMNN, MSECompressor, VMIMCompressor, HierarchicalVMIM

vmim = VMIMCompressor(n_out=3, config={"widths": (256, 128, 64), "epochs": 300}, device="auto", seed=0)
val_loss = vmim.fit(x[tr], theta[tr], x[va], theta[va])      # or vmim.fit(x, theta)
s = vmim(x)

blocks = {"first": range(0, 20), "second": range(20, 40), "third": range(40, 55)}    # columns of x
h = HierarchicalVMIM(blocks, anchor=["first"], orders=[["second", "third"], ["third", "second"]], n_out=3)
h.fit(x[tr], theta[tr], x[va], theta[va])

trn, val = fid.split(groups_fid, groups_deriv, val_groups=[0, 1])
imnn = IMNN(trn, {"patience": 250, "min_iterations": 1000}, validation=val)
imnn.fit()
```

- `MOPED` uses the sample covariance of the fiducial simulations (needs `n_fid > D + 2`) or the matrix passed as
  `cov=`. Degenerate parameters are dropped with a warning. `pivot_scale={name: prior width}` sets the order of
  the Gram-Schmidt orthogonalisation. `MOPED.from_vectors(vectors, params)` wraps an existing compression matrix.
- `NetConfig` sets the network compressors: layer `widths`, `epochs`, learning rate, early stopping,
  `input_scaling`, `optimizer`, `schedule`, `network`, and `vicreg` for VICReg terms in the VMIM loss.
- `MSECompressor.predict(x)` returns the parameter estimates in parameter units.
- If many training simulations share exactly one value of a parameter, `fit(..., atoms=[(column, value,
  half_width)])` spreads those labels over `value ± half_width`, so that the VMIM density cannot fit a spike there.
- `HierarchicalVMIM(configs={"default": {...}, "anchor": {...}, "final": {...}, "<block>": {...}})` sets a
  `NetConfig` per stage.
- `IMNN` follows `imnn` 0.3.2 (`NumericalGradientIMNN`), which does not scale its inputs; set
  `input_scaling="zscore"` for data vectors far from order one.

### Neural likelihood estimation

```python
nle = sk.NLE(flow={"model": "nsf", "hidden_features": 64, "num_transforms": 5}, device="auto", seed=0)
val_loss, epochs = nle.fit(s[tr], theta[tr], s[va], theta[va], train={"lr": 1e-3, "batch_size": 128})
logp = nle.log_likelihood(s[te], theta[te])           # (n_test,) tensor
s_new = nle.sample(theta[te], n=10)                   # (10, n_test, n) summaries drawn from the flow
```

- `FlowConfig`: `model` (`"nsf"`, `"maf"`, `"naf"`, `"unaf"`, `"nice"`, `"gf"`, `"bpf"`, `"ncsf"`, `"sospf"`),
  `hidden_features`, `num_blocks`, `num_transforms`, `num_bins`, and `kwargs` for the builder.
- `TrainConfig`: learning rate, batch size, `optimizer` (`"adam"`, `"adamw"`), `scheduler` (`"plateau"`,
  `"none"`), early stopping, gradient clipping, `val_fraction`.
- Summaries and parameters are z-scored inside the `NLE` (`s_scaler=`, `t_scaler=`); the scalers are saved with
  the weights. `log_prob` is the density of the unscaled summaries.
- Every `fit` starts from new weights; `warm_start=True` continues from the current ones.
- `fit(..., optimizer=lambda params: ..., scheduler=lambda opt: ...)` replace the optimizer and scheduler.
  `nle.train_history` holds the losses per epoch.
- `sk.Ensemble([nle_1, nle_2, ...], weights=None)` averages the likelihoods of several flows and is used like an
  `NLE`.

### Priors

A `Prior` is a box, optionally with density terms on some parameters and constraints on the support. Parameters
without a term are uniform over the box.

```python
from sbikit import Prior, Gaussian, KDE, Polygon2D, CurveBound2D

prior = Prior(
    low=[0.0, 0.0, -1.0], high=[1.0, 2.0, 1.0], names=["a", "b", "c"],
    terms=[Gaussian(dims=["c"], mean=[0.0], cov=[0.2 ** 2])],
    constraints=[
        Polygon2D(dims=["a", "b"], vertices=[[0, 0], [1, 0], [1, 1], [0, 2]]),     # convex region in (a, b)
        CurveBound2D.from_function(dims=["a", "c"], fn=lambda a: a ** 2 - 1, x_min=0.0, x_max=1.0),  # c >= a^2 - 1
    ],
)
prior.log_prob(points)             # -inf outside the support
prior.sample(1000, seed=0)         # (1000, 3) tensor
```

- `dims` are parameter names or column indices.
- `Prior.box_from_data(theta, pad=0.0)` takes the box from the range of the simulations.
- `KDE(dims, centres)` is a kernel density estimate, e.g. of a non-uniform simulation design.
- `Polygon2D` takes the vertices of a convex polygon in order. `CurveBound2D` keeps one parameter above or below
  a curve in another, given as nodes or with `from_function`. Any function `theta (..., P) -> bool (...)` also
  works as a constraint.
- The sampler needs finite box bounds unless `MCMCConfig(logit_box=False)` is set.

### Sampling

```python
config = sk.MCMCConfig(nwalkers=256, moves="de_snooker", rhat=1.01, min_ess=2000, n_samples=2000)
post = sk.sample_posteriors(nle, s_obs, prior, config, init_centre=centres, init_sd=theta[tr].std(0))

post.samples        # (B, n_samples, P)
post.converged      # (B,) bool
post.diagnostics    # rhat, ess, tau and acceptance per target, steps, seconds
post.mean(), post.std(), post.quantile([0.16, 0.84]), post.summary()
post.save("posteriors.npz"); post = sk.Posteriors.load("posteriors.npz")
```

- Every row of `s_obs` is a target with its own walkers. `prior` can be a list with one prior per target;
  `targets_per_batch` splits very large sets.
- With `init_centre`, walkers start in a ball of width `init_scale × init_sd` around it. Without it, each target
  starts from its best prior draws. `init_walkers=` sets the positions directly.
- After `burn_in`, the chain grows in blocks of `block` steps until every target has R-hat below `rhat`, at least
  `min_ess` effective samples and a length of `min_tau` autocorrelation times, or until `max_steps`. Targets that
  did not converge are returned, flagged in `post.converged`, with a warning.
- `moves` is a preset (`"de_snooker"`, `"de"`, `"stretch"`, `"walk"`, `"kde"`), a torchemcee `Move`, or a
  weighted list such as `[["de", 0.8], ["gaussian", 0.2, {"cov": 0.01}]]`.
- By default walkers move in the logit of the prior box (`logit_box=False` turns that off).
- `keep_chain=True` also returns `post.chain` and `post.log_prob`; `init_walkers=post.chain[-1]` continues a run.
- `sk.find_map(nle, s_obs, prior, post.samples)` returns the maximum of every posterior and its log density,
  starting from the best of the given points.
- Any object with `log_prob(input, condition)` in `sbi`'s layout can be the likelihood.

### Parameter transforms

To sample parameters `ψ` other than the parameters `θ = f(ψ)` the flow was trained on, pass `transform=`. The
prior and the start positions are then in `ψ`, and `post.derived` holds `f` of the samples.

```python
from sbikit import Columns

# flow conditioned on (a, b); sample (a, r) with b = a * r
to_b = Columns(["a", "r"], keep=["a"], derived=["b"], fn=lambda psi: psi[..., 0:1] * psi[..., 1:2])
post = sk.sample_posteriors(nle, s_obs, prior_a_r, transform=to_b, init_centre=centre)
```

Any callable on `(..., P)` tensors works as a transform.

### Metrics

```python
from sbikit.metrics import evaluate

# samples (n_sims, n_samples, P), truth (n_sims, P)
record, arrays = evaluate(post.samples, theta[te], names,
                          fom_sets={"a_b": ["a", "b"]}, tarp_sets={"a_b": ["a", "b"]})
```

`record` is a JSON-serialisable dict; `arrays` holds the TARP curves and per-simulation values.

| entry | content |
|---|---|
| `tarp[set]` | `bias` (offset of the coverage curve from the diagonal), `confidence` (negative: overconfident), bootstrap errors, `max_dev`; for all parameters and each named set |
| `marginal[param]` | coverage of the central 68 % and 95 % intervals, KS distance of the ranks and its p-value |
| `fom[set]` | mean of `log FoM = −½ log det Cov` over the simulations |
| `z_mean`, `z_rms`, `std_median` | z-scores of the posterior mean and median posterior width, per parameter |

- Options: `weights`, `levels`, `bounds`, `drop_nonfinite`, and `extra={"name": fn(samples, truth)}` for your own
  statistics.
- TARP scales the parameters by the range of the test truths. For small test sets, pass the prior box as
  `bounds=(low, high)`.
- Also available separately: `tarp_f`, `ecp_curve`, `tarp_scalars`, `marginal_coverage`, `posterior_summary`,
  `log_fom`, `effective_sample_size`, `importance_weights(samples, from_prior, to_prior)`.

### Goodness of fit

```python
from sbikit.gof import likelihood_chi2, predictive_statistic, calibrated_pvalue

# the learned likelihood on test simulations at their true parameters
chi2 = likelihood_chi2(nle, s[te], theta[te])            # (n_test,); χ²(n) if the likelihood is Gaussian

# an observation at its posterior maximum
theta_map, log_prob = sk.find_map(nle, s_obs, prior, obs.samples)
chi2_obs = likelihood_chi2(nle, s_obs, theta_map)

# posterior predictive test of the observation against the test simulations
t_obs = predictive_statistic(nle, s_obs, obs.samples)    # −2 ⟨log p(s | θ)⟩ over the posterior samples
t_ref = predictive_statistic(nle, s[te], post.samples)
p = calibrated_pvalue(t_obs, t_ref)                      # fraction of the simulations that fit worse
```

- `likelihood_chi2` is the squared distance of the summaries from the mean of the likelihood at `theta`, in units
  of its covariance; both are estimated from `n_draws` draws of the flow.
- No statistic of a flow has a known distribution: compare the value of an observation with the same statistic of
  simulations (`calibrated_pvalue`). Realisations at one parameter point give a sharper reference than
  simulations from the whole prior.
- With the posterior samples of another analysis (part of the data vector removed, another statistic),
  `predictive_statistic` tests whether the two are consistent.
- `columns=[...]` restricts a statistic to some summaries, e.g. to leave out those of parameters the flow is not
  conditioned on.

### Hyperparameter searches

```python
from sbikit.hpt import tune_nle, tune_compressor, EvalSet, best_trials, load_trial, SPACES
from sbikit.compress import VMIMCompressor

# flows, by validation loss
study = tune_nle(s[tr], theta[tr], s[va], theta[va], n_trials=50, storage="hpt/study.log", name="nle",
                 outdir="hpt", device="auto")
best = load_trial(study, outdir="hpt")

# flows, by figure of merit subject to calibration, on simulations with known truths
ev = EvalSet(s[te], theta[te], prior, names)
study = tune_nle(s[tr], theta[tr], s[va], theta[va], n_trials=50, storage="hpt/study.log", name="nle_fom",
                 outdir="hpt", objective="fom_tarp", evaluation=ev, fom_params=["a", "b"], device="auto")
ensemble = sk.Ensemble([load_trial(study, t.number, "hpt") for t in best_trials(study, k=3)])

# compressors
cfg, vmim, study = tune_compressor(lambda c: VMIMCompressor(3, c, "auto"), SPACES["vmim"],
                                   x[tr], theta[tr], x[va], theta[va], n_trials=40,
                                   storage="hpt/comp.log", name="vmim", outdir="hpt")
```

- A search space is a dict `name -> fixed value | {"choices": [...]} | {"low", "high", "log", "int", "step"}`.
  `SPACES` has defaults for `"nle"`, `"mse"`, `"vmim"` and `"imnn"`.
- `objective` is `"val_loss"`, `"fom_tarp"` (maximises the mean log figure of merit of `fom_params`, with the TARP
  scalars and chain convergence as constraints) or your own function `(nle, trial) -> value`.
- `storage` is `None` (in memory), a file (`*.db`: sqlite; otherwise an Optuna journal file, which several
  processes can share), a URL or an Optuna storage. Calling a search again on an existing study resumes it; give
  parallel workers different `seed` values.
- With `outdir`, every trial's model and a JSON record are written to `outdir/name/trials/`.
- `tune_compressor(..., fixed={...})` trains one given configuration.

### Blinding

```python
from sbikit.blinding import make_key, Blinder

make_key("blind.key", {"a": 0.05, "c": 0.3})          # once; keep it out of version control
blinder = Blinder("blind.key")
blinded = blinder.blind(post.samples, ["a", "b", "c"])     # shifts a and c by hidden offsets
blinder.fingerprint                                    # identifies the key
```

Each offset is drawn from `scale × U(−1, 1)`. `blind` raises if a parameter of the key is missing from the names
(`missing="ignore"` allows it). Derived parameters have to be recomputed from the blinded samples.

### Saving, loading and run records

```python
sk.save(obj, "file.pt", meta={"note": "run 3"})   # NLE, Ensemble, any compressor, Prior, scaler
obj = sk.load("file.pt", device="auto")
sk.read_meta("file.pt")                           # kind, package versions, your meta
sk.run_record("run.json", config=config, data_files=["data.npz"], splits=splits, repo=".")
```

Checkpoints load with `torch.load(..., weights_only=True)`. `run_record` writes the configuration, package
versions, git hashes, checksums of the data files, the hash of the splits, host and GPU. `Posteriors` and `Splits`
have their own `save` / `load`.

## Extending sbikit

Your own components are registered under a name and then selected like the built-in ones, including in
`sk.save` / `sk.load`.

| to add | register with | select by |
|---|---|---|
| a flow | `sk.register_flow(name, builder)` | `FlowConfig(model=name, kwargs={...})` |
| a compression network | `sbikit.compress.register_network(name, factory)` | `NetConfig(network=name, network_kwargs={...})` |
| a scaler | `sk.register_scaler(cls)` | `NetConfig(input_scaling=name)`, `NLE(s_scaler=name)` |
| a prior term / constraint | `sk.register_term(cls)` / `sk.register_constraint(cls)` | `Prior(terms=[...], constraints=[...])` |
| an MCMC move | `sk.register_move(name, cls)` | `MCMCConfig(moves=[[name, weight, {...}]])` |
| a compressor | `sbikit.compress.register(cls)` | the class |
| another saveable object | `sbikit.io.register_loader(kind, fn)` | `sk.save` / `sk.load` |

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
def tanh_mlp(n_in, n_out, config):                 # config is the NetConfig
    width = config.network_kwargs.get("width", 64)
    return nn.Sequential(nn.Linear(n_in, width), nn.Tanh(), nn.Linear(width, n_out))

vmim = VMIMCompressor(3, {"network": "tanh_mlp", "network_kwargs": {"width": 128}})
```

Without registering, objects can be passed directly: `sk.NLE(flow=my_builder)`,
`VMIMCompressor(3, net=lambda n_in, n_out: my_module)`, a function as a prior constraint or transform, a
torchemcee `Move` in `MCMCConfig(moves=...)`. Only saving them needs the registered form.

## Tests

```bash
python -m pytest                    # 325 tests on the CPU
python -m pytest --device cuda      # on the GPU
python -m pytest -m "not slow"      # skip the longest tests
```

## References

- Heavens, Jimenez & Lahav (2000), *Massive lossless data compression and multiple parameter estimation from galaxy spectra* (MOPED)
- Charnock, Lavaux & Wandelt (2018), *Automatic physical inference with information maximizing neural networks*; Makinen et al. (2021), *Lossless, scalable implicit likelihood inference for cosmological fields* (IMNN)
- Jeffrey, Alsing & Lanusse (2021), *Likelihood-free inference with neural compression of DES SV weak lensing map statistics* (VMIM); Williamson et al. (2026) (hierarchical VMIM)
- Bardes, Ponce & LeCun (2022), *VICReg: Variance-invariance-covariance regularization for self-supervised learning*
- Papamakarios et al. (2019), *Sequential neural likelihood*; Alsing et al. (2019), *Fast likelihood-free cosmology with neural density estimators and active learning* (neural likelihood estimation)
- Papamakarios, Pavlakou & Murray (2017), *Masked autoregressive flow for density estimation*; Durkan et al. (2019), *Neural spline flows*
- Foreman-Mackey et al. (2013), *emcee: The MCMC hammer*; ter Braak & Vrugt (2008), *Differential Evolution Markov Chain with snooker updater and fewer chains*
- Lemos et al. (2023), *Sampling-based accuracy testing of posterior estimators for general inference* (TARP)
- Doux et al. (2021), *Dark Energy Survey internal consistency tests of the joint cosmological probe analysis with posterior predictive distributions*
- Hartlap, Simon & Schneider (2007), *Why your model parameter confidences might be too optimistic*
- Akiba et al. (2019), *Optuna: A next-generation hyperparameter optimization framework*
- Tejero-Cantero et al. (2020), *sbi: A toolkit for simulation-based inference*; Rozet et al., *zuko: Normalizing flows in PyTorch*

## License

MIT, see [LICENSE](LICENSE).
