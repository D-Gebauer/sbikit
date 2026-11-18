# sbikit

Simulation-based inference for cosmology with neural likelihood estimation: compress the data vector, learn the
likelihood of the summaries with a normalising flow, sample posteriors with a batched ensemble MCMC, and check
that they are calibrated.

- **Compression**: MOPED, MSE regression, VMIM, hierarchical VMIM, information-maximising neural networks (IMNN).
- **Neural likelihood estimation**: zuko MAF/NSF flows (built with `sbi`), scalers stored with the weights,
  likelihood ensembles.
- **Sampling**: posteriors of hundreds of observations or test simulations in one batched `torchemcee` ensemble on
  CPU or GPU, any emcee move, per-target priors, stopped by convergence checks (split R-hat, ESS, autocorrelation time).
- **Metrics**: TARP (identical to the `tarp` package), bias and confidence scalars with bootstrap errors,
  marginal coverage, figures of merit, importance reweighting between priors.
- **Hyperparameter searches**: Optuna studies for compressors and flows, by validation loss or by the figure of
  merit subject to calibration constraints.
- **Utilities**: priors with density terms and support constraints, parameter transforms, parameter-level
  blinding, checkpoints that load with `weights_only=True`, run records.

Each piece works on its own with numpy arrays in and out. sbikit does not load data, run forward models or decide
how simulations are split: an analysis keeps those and calls sbikit for the inference.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
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
- [Notes on methods](#notes-on-methods)
- [Tests](#tests)
- [References](#references)
- [License](#license)

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
[`torchemcee`](https://github.com/D-Gebauer/torchemcee) `>= 0.1.1` (installed from GitHub). The `sbi` and `zuko`
versions are pinned because saved flows depend on their layer names.

## Quick start

[`examples/quickstart.ipynb`](examples/quickstart.ipynb) runs the whole chain on a toy problem, with plots. In short:

```python
import numpy as np
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
nle = sk.NLE(n_x=s.shape[1], n_theta=3, flow={"model": "nsf", "hidden_features": 64}, device="cuda")
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
obs = sk.sample_posteriors(nle, moped(x_obs[None]), prior, init_centre=theta_fid[None])

sk.save(nle, "nle.pt")
sk.run_record("run.json", config={"flow": "nsf"}, splits=splits, extra={"metrics": record})
```

## User guide

Everything below is importable from `sbikit` (conventionally `import sbikit as sk`) or its submodules, and every
public class and function has a docstring with its arguments and shapes.

### Data containers and splits

| object | holds |
|---|---|
| `Dataset(theta, x, param_names, groups=, blocks=, meta=)` | simulations: parameters `(N, P)`, data vectors `(N, D)`; optional group ids, named column blocks of `x`, and per-simulation arrays that are not parameters |
| `FiducialSet(x, param_names, theta_fid, x_minus, x_plus, delta, dmu=)` | simulations at a fiducial point `(n_fid, D)` and seed-matched simulations at `theta_fid ± step` `(K, n_d, D)`, or the mean derivatives `dmu` `(K, D)` directly |
| `Splits({"train": idx, ...})` | named index arrays, with `check`, `hash`, `save`, `load` |

**Splits are your decision.** `resolve_splits(dataset, splits)` checks the splits you pass (indices in range, no
overlap). Without `splits` it draws random ones by fraction (`fractions={"train": 0.8, "val": 0.1, "test": 0.1}`).

**Groups.** Simulations that are correlated, such as several footprints or noise realisations cut from one N-body
run, must not sit on both sides of a split. Give them a common id in `Dataset(groups=...)`: random splits then keep
each group together, and splits you pass raise an error if they separate a group (`group_strict=False` turns that
off). `FiducialSet.split(groups_fid, groups_deriv, val_groups)` does the same for fiducial and derivative sets.

**Which parameters are labels.** `Dataset.theta` holds the parameters the flow is conditioned on. Parameters that
vary in the simulations but are left out of `theta` are marginalised implicitly; keep them in `meta` if you want
to look at them later.

### Compression

All compressors map `(N, D)` data vectors to `(N, n_out)` summaries and share one interface: `fit(...)` returns a
validation loss, calling the object compresses, `sk.save` / `sk.load` store and restore it.

| class | what it does | trained on |
|---|---|---|
| `MOPED(fiducial, params=, hartlap=True, pivot_scale=)` | linear compression `b = C⁻¹ ∂μ/∂θ`, Gram-Schmidt orthogonalised; one summary per parameter; degenerate parameters are dropped with a warning | `FiducialSet` |
| `IMNN(fiducial, config, validation=)` | network that maximises the Fisher information at the fiducial point; output is the quasi maximum-likelihood estimate of the parameters | `FiducialSet` with derivative simulations |
| `MSECompressor(n_out, config)` | network regressing the parameters (posterior-mean estimate) | `(x, theta)` |
| `VMIMCompressor(n_out, config)` | network trained with a mixture-density head to maximise the mutual information with the parameters | `(x, theta)` |
| `HierarchicalVMIM(blocks, anchor, orders, n_out)` | VMIM stages over blocks of the data vector, each conditioned on the earlier ones, combined by a final VMIM | `(x, theta)` |

```python
from sbikit.compress import MOPED, IMNN, MSECompressor, VMIMCompressor, HierarchicalVMIM

vmim = VMIMCompressor(n_out=3, config={"widths": (256, 128, 64), "epochs": 300}, device="cuda", seed=0)
val_loss = vmim.fit(x[tr], theta[tr], x[va], theta[va])
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
  compression matrix, so that an SBI and a likelihood analysis can share their summaries.
- **Network compressors** (`NetConfig`): LeakyReLU MLP with a linear skip connection, AdamW with a one-cycle
  schedule, early stopping, inputs scaled by median/MAD + asinh (robust against the occasional extreme
  simulation). `vicreg > 0` adds VICReg variance and covariance terms to the VMIM training loss.
- **Atoms.** If part of the training set sits exactly on one parameter value (ΛCDM simulations at `w0 = -1` among
  wCDM ones, say), the density head of a VMIM can place a near-delta spike there, and the loss then rewards the
  spike instead of information. `fit(..., atoms=[(column, value, half_width)])` spreads those labels uniformly over
  `value ± half_width` during training. Keep such rows out of the validation set as well.
- **IMNN** follows the authors' reference implementation, `imnn` 0.3.2 (`NumericalGradientIMNN`): the same loss,
  regularisation, early stopping and estimator, and the network of its examples. The loss, Fisher matrix and
  estimator are tested against a numpy transcription of that code. Pass a separate validation `FiducialSet`.
  Minibatches, dropout, weight decay, gradient clipping and input scaling are available and off by default.
- **Your own compressor**: subclass `sbikit.compress.Compressor`, set `kind`, implement `fit`, `__call__`, `state`,
  `from_state`, and decorate it with `sbikit.compress.base.register`.

### Neural likelihood estimation

```python
nle = sk.NLE(n_x=3, n_theta=3, flow={"model": "nsf", "hidden_features": 64, "num_transforms": 5}, device="cuda")
val_loss, epochs = nle.fit(s[tr], theta[tr], s[va], theta[va], train={"lr": 1e-3, "batch_size": 128}, seed=0)
logp = nle.log_likelihood(s[te], theta[te])           # (n_test,) tensor
```

`FlowConfig` sets the architecture (`model` `"nsf"` or `"maf"`, `hidden_features`, `num_blocks`,
`num_transforms`, `num_bins`), `TrainConfig` the training (Adam, learning-rate decay on plateau, early stopping,
gradient clipping). Both can be given as dicts. Summaries and parameters are z-scored inside the `NLE`.

`sk.Ensemble([nle_1, nle_2, ...])` is the average likelihood of several flows and can be used wherever an `NLE`
can.

### Priors

A `Prior` is a box, times optional density terms on subsets of the parameters, times optional support constraints.
Parameters without a term are uniform over the box.

```python
import numpy as np
from sbikit import Prior, Gaussian, KDE, Polygon2D, CurveBound2D

prior = Prior(
    low=[0.2, 0.6, -1.5, -0.1], high=[0.4, 1.0, -0.5, 0.1], names=["Om", "s8", "w0", "dz"],
    terms=[Gaussian(dims=[3], mean=[0.0], cov=[[0.02 ** 2]])],             # Gaussian prior on dz
    constraints=[
        # a band along the Om-s8 degeneracy
        Polygon2D(dims=[0, 1], vertices=[[0.2, 1.0], [0.25, 1.0], [0.4, 0.75],
                                         [0.4, 0.6], [0.35, 0.6], [0.2, 0.85]]),
        # w0 above a curve in Om
        CurveBound2D.from_function(dims=[0, 2], fn=lambda om: -0.7 - np.sqrt(om), x_min=0.2, x_max=0.4),
    ],
)
prior.log_prob(theta_tensor)       # -inf outside the support
prior.sample(1000)
```

- Every parameter needs finite box bounds, because the sampler works in the logit of the box. For a Gaussian
  parameter use, e.g., mean ± 5σ.
- `KDE(dims, centres)` is a Gaussian kernel density with a leave-one-out bandwidth: use it when the simulations
  follow a design that is not uniform, to sample with the design density as the prior.
- Constraints are callables `theta (..., P) -> bool`. Two are built in and can be saved: `Polygon2D`, a convex
  polygon in the plane of two parameters, for straight edges; and `CurveBound2D`, which keeps one parameter above
  or below a curve in another, given as nodes or tabulated from a function with `from_function`. Custom callables
  work for sampling but not for `sk.save`.
- `Prior.box_from_data(theta)` takes the box from the range of the simulations.

### Sampling

```python
config = sk.MCMCConfig(nwalkers=256, moves="de_snooker", rhat=1.01, min_ess=2000, n_samples=2000)
post = sk.sample_posteriors(nle, s_obs, prior, config, init_centre=centres, init_sd=theta[tr].std(0))

post.samples        # (B, n_samples, P)
post.converged      # (B,) bool
post.diagnostics    # rhat_max, ess_min, tau_max, acceptance per target; steps, seconds
```

- **Batched.** Every row of `s_obs` is a target with its own walkers; one step evaluates the flow once for all
  targets. `prior` can be a list with one prior per target. `targets_per_batch` splits very large sets.
- **Logit box.** By default walkers move in `logit((θ − low)/(high − low))`, so they cannot leave the prior box and
  posteriors that pile up at a bound still mix.
- **Moves.** A preset (`"de_snooker"` (default), `"de"`, `"stretch"`, `"walk"`, `"kde"`) or a weighted list of
  torchemcee moves, e.g. `[["de", 0.8, {}], ["gaussian", 0.2, {"cov": 0.01}]]`.
- **Convergence.** After `burn_in`, the chain grows in blocks until, on its second half, every target has grouped
  split R-hat below `rhat`, at least `min_ess` effective samples and a length of `min_tau` autocorrelation times in
  every parameter, or until `max_steps`. Samples come from that second half. Targets that did not converge are
  returned too, flagged in `post.converged`.
- **Initialisation.** `init="ball"` starts walkers in a small ball around `init_centre` (the truths for test
  simulations; a fiducial point or best fit for data); `init="prior"` draws them from the prior.

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
redshift evolution of an amplitude that the simulations vary freely per tomographic bin.

### Metrics

`sbikit.metrics` works on `samples (n_sims, n_samples, P)` and `truth (n_sims, P)`.

```python
from sbikit.metrics import evaluate
record, arrays = evaluate(post.samples, theta[te], names,
                          fom_sets={"om_s8": ["Om", "S8"]}, tarp_sets={"om_s8": ["Om", "S8"]})
```

`record` is a JSON-serialisable dict:

| entry | content |
|---|---|
| `tarp[set]` | `bias` (negative: coverage curve below the diagonal), `confidence` (negative: overconfident), their bootstrap errors, `max_dev`; for `"all"` parameters and each named set |
| `marginal[param]` | fraction of truths in the central 68 % and 95 % intervals, KS distance of the rank statistic |
| `fom[set]` | mean of `log FoM = −½ log det Cov` over the simulations, its geometric-mean FoM |
| `z_mean`, `z_rms`, `std_median` | per-parameter z-scores of the posterior mean and median posterior width |

`arrays` holds the TARP curves and per-simulation values. The building blocks are available separately: `tarp_f`,
`ecp_curve`, `tarp_scalars`, `marginal_coverage`, `posterior_summary`, `log_fom`, and
`importance_weights(samples, from_prior, to_prior)` to turn posteriors under the sampling prior into posteriors
under another one.

Using the same record for every run is what makes compressions, flows and analyses comparable.

### Hyperparameter searches

```python
from sbikit.hpt import tune_nle, tune_compressor, EvalSet, best_trials, SPACES

# flows, by validation loss
study = tune_nle(s[tr], theta[tr], s[va], theta[va], n_trials=50, storage="hpt/study.log", name="nle",
                 outdir="hpt", device="cuda")

# flows, by figure of merit subject to calibration: needs simulations with known truths
ev = EvalSet(s[te], theta[te], prior, names)
study = tune_nle(s[tr], theta[tr], s[va], theta[va], n_trials=50, storage="hpt/study.log", name="nle_fom",
                 outdir="hpt", objective="fom_tarp", evaluation=ev, fom_params=["Om", "S8"], device="cuda")
best = best_trials(study, k=3)

# compressors
cfg, vmim, study = tune_compressor(lambda c: VMIMCompressor(3, c, "cuda"), SPACES["vmim"],
                                   x[tr], theta[tr], x[va], theta[va], n_trials=40,
                                   storage="hpt/comp.log", name="vmim", outdir="hpt")
```

- A search space is a dict `name -> fixed value | {"choices": [...]} | {"low", "high", "log", "int"}`;
  `SPACES` has defaults for `"nle"`, `"mse"`, `"vmim"` and `"imnn"`.
- `objective="fom_tarp"` samples the posteriors of the evaluation set for each trial and maximises the mean log
  figure of merit, with the TARP bias and confidence (on all parameters and on `fom_params`) and the convergence of
  every chain as constraints. Flows with a poor validation loss are pruned before the MCMC.
- Studies are stored in an Optuna journal file (several processes can work on one study) or, for `*.db`, in sqlite,
  and resume where they stopped. Every trial's model and record are written to `outdir/name/trials/`.

### Blinding

```python
from sbikit.blinding import make_key, Blinder

make_key("blind.key", {"S8": 0.05, "w0": 0.3})        # once; never overwritten; keep it out of version control
blinder = Blinder("blind.key")
blinded = blinder.blind(post.samples, names)           # shifts S8 and w0 by hidden offsets
blinder.fingerprint                                    # store with results to show which key was used
```

The offset of each parameter is drawn from `scale × U(−1, 1)` with the seed in the key file. Blind only the
posteriors of the data; validation on simulations needs no blinding.

### Saving, loading and run records

```python
sk.save(obj, "file.pt")          # NLE, Ensemble, any compressor, Prior
obj = sk.load("file.pt", device="cuda")
sk.run_record("run.json", config=config, data_files=["data.npz"], splits=splits, repo=".")
```

Checkpoints contain only tensors and plain Python values, so they load with `torch.load(..., weights_only=True)`,
and they record the versions of sbikit and its dependencies. `run_record` writes a JSON file with the
configuration, versions, git hashes of sbikit and of the analysis repository, md5 sums of the data files, the hash
of the splits, the host and the GPU.

## Notes on methods

**TARP.** `tarp_f` implements the definition of Lemos et al. (2023) as the `tarp` package does with `norm=True`
and random references, and reproduces its result exactly for every reference seed. On top of that, sbikit
averages 50 reference draws by default (a single draw is noisy), computes the scalars from the statistic
directly instead of from the coarse histogram, and bootstraps the simulations for errors. Two properties of the
test to keep in mind: the normalisation uses the range of the test truths, so compare scalars rather than curves
between test sets with different ranges; and uniform references are mostly far from the posteriors in many
dimensions, which makes the test insensitive to miscalibration in weakly constrained directions. This is why
`evaluate` also runs TARP on named subsets and reports marginal coverage.

**Snooker move.** The default moves are differential evolution (weight 0.8) and the snooker update (0.2) of
ter Braak & Vrugt (2008). The `DESnookerMove` of emcee 3.1.6 and of torchemcee 0.1.0 differs from that paper (it
scales the proposal by `(x − z)/√|x − z|` instead of the unit vector, and uses the exponent `(d − 1)/2` instead of
`d − 1`) and gives posteriors that are too narrow: on a flat 9-dimensional box, the standard deviation comes out
16 % too small with the snooker move alone and 3 % too small in the 0.8/0.2 mix. torchemcee 0.1.1 has the
corrected move, sbikit requires it, and the test suite checks every move and preset on that box to 1.5 %.

**MOPED and IMNN are local.** Both are built at one fiducial point and are optimal only near it. Over a wide prior,
compare them with a compressor trained on the whole parameter range.

**Likelihood normalisation.** `NLE.log_prob` is the density of the z-scored summaries; the constant Jacobian of the
scaling is left out. Posteriors are unaffected; add it if you compare likelihood values between models with
different scalers.

## Tests

```bash
python -m pytest                    # 39 tests on the CPU
python -m pytest --device cuda      # the same tests, training and sampling on the GPU
python -m pytest -m "not slow"      # skip the longest end-to-end tests
```

The tests check, among other things: priors against analytic densities; MOPED being lossless for a linear Gaussian
model; the IMNN against the reference formulas and the analytic Fisher information; batched posteriors against
analytic Gaussian posteriors; the width of a flat box for every move; the flow learning a linear Gaussian
likelihood; TARP against the `tarp` package; both hyperparameter objectives end to end; and save/load round trips
of every object.

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
