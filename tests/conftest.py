"""Shared fixtures. ``pytest --device cuda`` runs every test that trains or samples on the GPU."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--device", default="cpu", help="torch device for training and sampling (cpu, cuda, cuda:1, mps)")


@pytest.fixture(scope="session")
def device(request):
    name = request.config.getoption("--device")
    if name.startswith("cuda") and not torch.cuda.is_available():
        pytest.exit(f"--device {name} requested but CUDA is not available", returncode=2)
    return name


def pytest_report_header(config):
    name = config.getoption("--device")
    gpu = f" ({torch.cuda.get_device_name(0)})" if name.startswith("cuda") and torch.cuda.is_available() else ""
    return f"sbikit test device: {name}{gpu}, torch {torch.__version__}"


class GaussianLikelihood:
    """Exact linear-Gaussian likelihood x = A theta + b + N(0, sigma^2), in sbi's estimator layout."""

    def __init__(self, A, b, sigma, device="cpu"):
        self.A = torch.as_tensor(A, dtype=torch.float32, device=device)
        self.b = torch.as_tensor(b, dtype=torch.float32, device=device)
        self.sigma = float(sigma)
        self.device = device

    def log_prob(self, input, condition):
        mu = condition @ self.A.T + self.b                     # (M, D)
        x = input.reshape(-1, mu.shape[-1])
        return (-0.5 * ((x - mu) / self.sigma) ** 2).sum(-1).reshape(1, -1)

    def to(self, device):
        return self


@pytest.fixture
def linear_model(device):
    rng = np.random.default_rng(1)
    P, D, sigma = 2, 4, 0.05
    A = rng.normal(size=(D, P))
    b = rng.normal(size=D)
    return {"P": P, "D": D, "A": A, "b": b, "sigma": sigma, "lik": GaussianLikelihood(A, b, sigma, device)}


def simulate(model, theta, rng):
    return theta @ model["A"].T + model["b"] + model["sigma"] * rng.normal(size=(len(theta), model["D"]))
