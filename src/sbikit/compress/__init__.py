"""Compressors: maps from data vectors (N, D) to low-dimensional summaries (N, n_out).

==========================  =========================================================  ==============
class                       method                                                     trained on
==========================  =========================================================  ==============
:class:`MOPED`              linear, optimal for a Gaussian likelihood near a fiducial  fiducial set
:class:`IMNN`               network maximising the Fisher information at a fiducial    fiducial set
:class:`MSECompressor`      network regressing the parameters (posterior mean)         (x, theta)
:class:`VMIMCompressor`     network maximising the mutual information with theta       (x, theta)
:class:`HierarchicalVMIM`   VMIM stages over blocks of the data vector                 (x, theta)
==========================  =========================================================  ==============

All share the :class:`Compressor` interface: ``fit(...)`` returns a validation loss, calling the
compressor maps data vectors to summaries, and ``state()`` / :func:`compressor_from_state`
(or :func:`sbikit.io.save` / :func:`sbikit.io.load`) store and restore it.
"""
from .base import NETWORKS, REGISTRY, Compressor, compressor_from_state, register, register_network
from .hierarchical import HierarchicalVMIM
from .imnn import IMNN, IMNNConfig
from .moped import MOPED, hartlap_factor, moped_vectors, moped_vectors_pivoted
from .nets import MDN, MSECompressor, Net, NetConfig, VMIMCompressor

__all__ = ["Compressor", "REGISTRY", "register", "compressor_from_state", "NETWORKS", "register_network", "MOPED", "moped_vectors", "moped_vectors_pivoted", "hartlap_factor",
           "MSECompressor", "VMIMCompressor", "NetConfig", "Net", "MDN", "HierarchicalVMIM", "IMNN", "IMNNConfig"]
