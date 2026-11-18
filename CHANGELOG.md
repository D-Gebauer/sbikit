# Changelog

## 0.2.0

Checkpoints written by 0.1.0 load unchanged. Results for a given seed differ from 0.1.0 wherever noted.

### Changed behaviour

- **Likelihood values include the scaler Jacobian.** `NLE.log_prob`, `log_likelihood`, `val_loss` and the losses
  returned by `fit` are densities of the raw summaries (0.1.0: of the z-scored ones; new = old − Σ log std).
  Posteriors of a single flow are unchanged. An `Ensemble` of flows with different scalers is now the mean of
  proper densities; before, each member was weighted by the product of its scaler's standard deviations.
- **`MCMCConfig.max_steps` counts steps after burn-in**, also with `adaptive=True` (0.1.0 stopped at
  `burn_in + steps >= max_steps`). `diagnostics["steps"]` excludes burn-in. `block` and `max_steps` are rounded to
  multiples of `thin`, so thinning is regular across convergence checks.
- **Seeds.** `fit(seed=)` and `NLE(seed=)` fix the weight initialisation as well as the batch order, and every
  `fit` starts from fresh weights (`warm_start=True` continues). No function changes torch's global random state
  any more: models trained in a loop with different seeds differ, and a seeded call does not alter the caller's
  random stream. Hyperparameter-search trials are reproducible for a given seed.
- **Unknown settings raise.** Unknown keys in any config dict (`flow`, `train`, compressor configs, `MCMCConfig`,
  search spaces, per-stage configs of `HierarchicalVMIM`) raise a `TypeError` naming the closest field; invalid
  values (unknown `model`, `init`, `output`, `input_scaling`, out-of-range numbers) raise a `ValueError`.
- **Inputs are checked.** Wrong shapes, mismatched lengths and NaN or inf in simulations raise a `ValueError`
  instead of being broadcast or trained on. Using a model before `fit` raises `NotFittedError`; a fit without a
  finite validation loss warns and leaves the model unfitted.
- **Blinding.** `Blinder.blind` raises if a parameter of the key is missing from `names` (0.1.0 left it
  unblinded silently); `missing="ignore"` restores the permissive behaviour.
- **MOPED** raises when the fiducial set is too small for the covariance (`n_fid <= D + 2` with the Hartlap
  factor) instead of dropping every parameter or returning vectors from a singular matrix.
- **Sampling** warns about targets that did not converge and about `n_samples` exceeding the chain; samples are
  drawn without replacement; a single `(P,)` `init_centre` serves all targets; the default ball width comes from
  each target's own prior; `init` defaults to `"auto"`, which without a centre starts each
  target from its best prior draws (`init="prior"`: the best `nwalkers` of 20 × `nwalkers`; 0.1.0 took the
  first `nwalkers` draws).
- **Metrics.** TARP `max_dev` is computed from the statistic directly (0.1.0: from the binned curve, which is 0
  below 20 simulations); a parameter with the same truth in every simulation needs `bounds=` instead of silently
  giving `bias = 1`; ranks use mid-ranks; undefined numbers in the record are `None`; a numerically singular
  covariance gives a nan figure of merit.
- **Hyperparameter searches.** A resumed study draws new configurations (0.1.0 repeated the earlier ones);
  `n_trials` is the total over all workers; the `fom_tarp` pruning threshold uses all finished trials;
  `tune_compressor(fixed=...)` trains exactly the given configuration and never returns `None` as the compressor.
- **KDE prior.** The leave-one-out bandwidth no longer collapses when centres are repeated, and is found on a
  subset for more than 2000 centres.
- **Network compressors.** A last minibatch of one row is skipped (it made the VICReg terms NaN);
  `grad_clip=0` or `None` turns clipping off; a flat `orders` list of `HierarchicalVMIM` is one repeat.
- `fit` keyword names are `x, theta, x_val, theta_val` (`s, theta, s_val, theta_val` for `NLE`); the positional
  order is unchanged.
- Messages go through the `sbikit` logger instead of `print`; importing sbikit no longer changes the level of
  torchemcee's logger.

### Added

- Optional validation sets (`fit(x, theta)` holds out `val_fraction`); `NLE()` infers its dimensions from the data.
- `device="auto"`, `sk.resolve_device`, `sk.set_verbosity`, `sk.NotFittedError`.
- Registries with `register_flow`, `register_network`, `register_scaler`, `register_term`,
  `register_constraint`, `register_move`, `io.register_loader`; all zuko flows that `sbi` builds;
  `FlowConfig.kwargs`; custom flow builders, estimators and networks passed directly.
- `fit(callback=, verbose=)` everywhere; `NLE.fit(optimizer=, scheduler=, warm_start=)`; `train_history`,
  `best_val_loss`; training settings stored in checkpoints.
- `NLE.sample`, `NLE(s_scaler=, t_scaler=)`, `Ensemble(weights=)`, `Ensemble.log_likelihood` / `val_loss`.
- `MSECompressor.predict`, `MOPED(cov=)`, `Compressor.to`, single-vector inputs to every compressor.
- `Prior`: `dims` by name, numpy in and out, `sample(seed=)`, `box_from_data(pad=)`, infinite bounds for
  parameters with a density term, subclasses preserved by `to()`, `__repr__`.
- `sample_posteriors(init_walkers=, keep_chain=, callback=)`; `Move` objects in `MCMCConfig.moves`;
  `n_samples=None`; `Posteriors.mean/std/quantile/summary/save/load`, `names`, per-parameter diagnostics.
- `evaluate(weights=, levels=, bounds=, extra=, drop_nonfinite=)`, `effective_sample_size`, `tarp_f(bounds=,
  references=)`.
- In-memory and URL study storage, `load_trial`, callable objectives, `sampler=`, `catch=`, `fit_kwargs=`.
- `sk.read_meta`, `load(return_meta=True)`, saveable scalers.

## 0.1.0

First public version.
