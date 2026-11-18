"""sbikit: simulation-based inference for cosmology.

The pieces of a neural-likelihood analysis, each usable on its own:

- :mod:`sbikit.data`: containers for simulations, fiducial/derivative sets and splits;
- :mod:`sbikit.compress`: MOPED, MSE, VMIM, hierarchical VMIM and IMNN compression;
- :mod:`sbikit.nle`: neural likelihood estimation with normalising flows, and ensembles;
- :mod:`sbikit.priors`: box priors with density terms and support constraints;
- :mod:`sbikit.sampling`: batched, convergence-checked ensemble MCMC on CPU or GPU;
- :mod:`sbikit.transforms`: parameter maps between the sampled and the learned parameters;
- :mod:`sbikit.metrics`: TARP, marginal coverage, figures of merit, importance reweighting;
- :mod:`sbikit.hpt`: Optuna hyperparameter searches for compressors and flows;
- :mod:`sbikit.blinding`: parameter-level blinding of posterior samples;
- :mod:`sbikit.io`: checkpoints and run records.

The most used classes and functions are re-exported here (``import sbikit as sk``).
"""
__version__ = "0.1.0"

import sys as _sys
import types as _types

# sbi imports torch.utils.tensorboard, whose compat layer imports TensorFlow whenever it is installed. In environments
# that have both, TensorFlow loaded before triton (imported lazily by torch.optim -> torch._dynamo) can segfault the
# process (a native-library clash seen with TF 2.21 + triton 3.8), and the import alone can take a minute on a network
# file system. tensorboard's own opt-out is an importable `tensorboard.compat.notf`: it then uses its TF stub, which
# is all the SummaryWriter needs. Nothing is changed if TensorFlow is already imported.
if "tensorflow" not in _sys.modules:
    _sys.modules.setdefault("tensorboard.compat.notf", _types.ModuleType("tensorboard.compat.notf"))

from . import blinding, compress, hpt, io, metrics, transforms
from .data import Dataset, FiducialSet, Splits, random_splits, resolve_splits
from .io import load, run_record, save
from .nle import NLE, Ensemble, FlowConfig, TrainConfig
from .priors import KDE, CurveBound2D, Gaussian, Polygon2D, Prior
from .sampling import MCMCConfig, Posteriors, sample_posteriors
from .scaling import RobustAsinh, ZScore
from .transforms import Columns, Identity, Transform

__all__ = ["Dataset", "FiducialSet", "Splits", "random_splits", "resolve_splits", "NLE", "Ensemble", "FlowConfig",
           "TrainConfig", "Prior", "Gaussian", "KDE", "Polygon2D", "CurveBound2D", "MCMCConfig", "Posteriors",
           "sample_posteriors", "ZScore", "RobustAsinh", "compress", "hpt", "io", "metrics", "load", "save",
           "run_record", "blinding", "transforms", "Transform", "Identity", "Columns"]
