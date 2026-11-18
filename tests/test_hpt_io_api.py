"""Checkpoints, run records and hyperparameter searches: inputs, errors, resuming and extension points."""
import dataclasses
import json
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch
from optuna.trial import TrialState

import sbikit as sk
from sbikit import hpt, io
from sbikit.compress import Compressor, register
from sbikit.hpt import EvalSet, best_trials, load_trial, make_study, run_study, suggest, tune_compressor, tune_nle

optuna.logging.set_verbosity(optuna.logging.WARNING)


@register
class Toy(Compressor):
    """A compressor without training: its "loss" is a function of the config plus seeded noise."""
    kind = "toy_hpt"
    n_out = 1

    def __init__(self, config=None):
        self.config, self.noise, self.fit_kwargs = dict(config or {}), None, None

    def fit(self, x_trn=None, t_trn=None, x_val=None, t_val=None, **kwargs):
        self.noise, self.fit_kwargs = float(torch.rand(1)), kwargs
        if self.config.get("fail"):
            return float("inf")
        return (self.config.get("a", 0.0) - 0.3) ** 2 + self.config.get("b", 0.0) + 1e-3 * self.noise

    def __call__(self, x):
        return np.asarray(x)[..., :1] + self.noise

    def state(self):
        return {"kind": self.kind, "config": self.config, "noise": self.noise, "shape": (1, 2), "path": Path("a") / "b",
                "n": np.int64(3), "scale": np.float32(0.5), "array": np.arange(3.0)}

    @classmethod
    def from_state(cls, state, device="cpu"):
        out = cls(state["config"])
        out.noise, out.extra = state["noise"], state
        return out


def toy_factory(cfg):
    unknown = set(cfg) - {"a", "b", "fail", "widths"}
    if unknown:
        raise TypeError(f"Toy has no field {sorted(unknown)}")
    return Toy(cfg)


TOY_SPACE = {"a": {"low": 0.0, "high": 1.0}, "b": 0.5}


def _sims(n=400, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-1, 1, size=(n, 2))
    s = (theta + 0.1 * rng.normal(size=(n, 2))).astype(np.float32)
    return s, theta


NLE_SPACE = {"model": "maf", "hidden_features": {"choices": [8, 16]}, "num_transforms": 1, "num_blocks": 1,
             "lr": {"low": 1e-3, "high": 1e-2, "log": True}}
NLE_TRAIN = {"max_epochs": 3, "batch_size": 128}


# ------------------------------------------------------------------------------ checkpoints
def test_numpy_values_round_trip(tmp_path):
    toy = Toy({"a": np.float64(0.25), "widths": (8, 4)})
    toy.noise = 0.5
    meta = {"cut": np.float32(1.5), "n": np.int64(7), "bins": np.arange(4), "file": Path("data") / "x.npz",
            "shape": (2, 3), "nested": {"flag": np.bool_(True)}}
    io.save(toy, tmp_path / "sub" / "toy.pt", meta=meta)
    loaded, info = io.load(tmp_path / "sub" / "toy.pt", return_meta=True)
    assert loaded.config == {"a": 0.25, "widths": [8, 4]} and type(loaded.config["a"]) is float
    assert loaded.extra["shape"] == [1, 2] and loaded.extra["path"] == str(Path("a") / "b")
    assert loaded.extra["n"] == 3 and torch.equal(loaded.extra["array"], torch.arange(3.0, dtype=torch.float64))
    m = info["meta"]
    assert m["cut"] == 1.5 and m["n"] == 7 and m["file"] == str(Path("data") / "x.npz") and m["shape"] == [2, 3]
    assert torch.equal(m["bins"], torch.arange(4)) and m["nested"]["flag"] is True
    assert not list(tmp_path.glob("sub/*.tmp"))


def test_read_meta_and_return_meta(tmp_path):
    prior = sk.Prior([0.0, 1.0], [1.0, 2.0], names=["a", "b"])
    io.save(prior, tmp_path / "prior.pt", meta={"run": "x1"})
    info = io.read_meta(tmp_path / "prior.pt")
    assert set(info) == {"kind", "format", "versions", "meta"}
    assert info["kind"] == "prior" and info["format"] == io.FORMAT and info["meta"] == {"run": "x1"}
    assert info["versions"]["sbikit"] == sk.__version__ and info["versions"]["torch"] == str(torch.__version__)
    obj, info2 = io.load(tmp_path / "prior.pt", return_meta=True)
    assert info2 == info and obj.names == ["a", "b"]
    assert isinstance(io.load(tmp_path / "prior.pt"), sk.Prior)
    io.save(prior, tmp_path / "bare.pt")
    assert io.read_meta(tmp_path / "bare.pt")["meta"] == {}


def test_save_rejects_what_load_cannot_read(tmp_path):
    with pytest.raises(TypeError, match="NLE, Ensemble, Prior"):
        io.save(object(), tmp_path / "x.pt")
    with pytest.raises(TypeError, match="NLE, Ensemble, Prior"):
        io.save({"kind": "nle"}, tmp_path / "x.pt")

    class NoKind:
        def state(self):
            return {"weights": [1.0]}

    with pytest.raises(TypeError, match="'kind'"):
        io.save(NoKind(), tmp_path / "x.pt")
    with pytest.raises(TypeError, match=r"meta\['model'\]\['fn'\]"):
        io.save(Toy({"a": 1.0}), tmp_path / "x.pt", meta={"model": {"fn": object()}})
    with pytest.raises(TypeError, match="meta must be a dict"):
        io.save(Toy({"a": 1.0}), tmp_path / "x.pt", meta=["a"])
    assert not list(tmp_path.iterdir())                      # nothing half-written is left behind

    class Unregistered:
        def state(self):
            return {"kind": "nobody_loads_this", "v": 1}

    with pytest.warns(UserWarning, match="register_loader"):
        io.save(Unregistered(), tmp_path / "u.pt")
    with pytest.raises(ValueError, match="import the module"):
        io.load(tmp_path / "u.pt")


def test_load_errors(tmp_path):
    torch.save([1, 2, 3], tmp_path / "list.pt")
    torch.save({"model": torch.zeros(2)}, tmp_path / "foreign.pt")
    (tmp_path / "text.pt").write_text("not a checkpoint")
    for name in ("list.pt", "foreign.pt", "text.pt"):
        with pytest.raises(ValueError, match="an sbikit checkpoint"):
            io.load(tmp_path / name)
        with pytest.raises(ValueError, match="an sbikit checkpoint"):
            io.read_meta(tmp_path / name)
    raw = {"format": io.FORMAT, "kind": "prior", "state": {}, "meta": {"cut": np.float64(1.5)}}
    torch.save(raw, tmp_path / "np.pt")
    with pytest.raises(ValueError, match="objects of the types.*numpy.*weights_only=False"):
        io.load(tmp_path / "np.pt")                          # numpy values written without save()
    with pytest.raises(FileNotFoundError):
        io.load(tmp_path / "missing.pt")

    io.save(sk.Prior([0.0], [1.0]), tmp_path / "prior.pt")
    ck = torch.load(tmp_path / "prior.pt", weights_only=True)
    torch.save({**ck, "kind": "priro"}, tmp_path / "unknown.pt")
    with pytest.raises(ValueError, match=r"kind 'priro'.*did you mean 'prior'.*import the module"):
        io.load(tmp_path / "unknown.pt")
    torch.save({**ck, "format": "sbikit/2"}, tmp_path / "newer.pt")
    with pytest.raises(ValueError, match="sbikit/2.*Update sbikit"):
        io.load(tmp_path / "newer.pt")
    with pytest.raises(RuntimeError, match="CUDA is not available|cuda"):
        if torch.cuda.is_available():
            raise RuntimeError("cuda")
        io.load(tmp_path / "prior.pt", device="cuda")
    assert isinstance(io.load(tmp_path / "prior.pt", device="auto"), sk.Prior)


def test_rebuild_failure_names_versions(tmp_path, device):
    s, theta = _sims(200)
    nle = sk.NLE(2, 2, {"model": "maf", "hidden_features": 8, "num_transforms": 1, "num_blocks": 1}, device)
    nle.fit(s, theta, train={"max_epochs": 1})
    io.save(nle, tmp_path / "nle.pt")
    ck = torch.load(tmp_path / "nle.pt", weights_only=True)
    ck["state"]["flow"] = {k + "_renamed": v for k, v in ck["state"]["flow"].items()}
    ck["versions"]["zuko"] = "0.0.1"
    torch.save(ck, tmp_path / "other.pt")
    message = r"(?s)could not rebuild the 'nle' object.*written with zuko 0\.0\.1; installed: zuko"
    with pytest.raises(RuntimeError, match=message):
        io.load(tmp_path / "other.pt", device)


def test_register_loader_and_scalers(tmp_path):
    class Calibration:
        def __init__(self, shift):
            self.shift = shift

        def state(self):
            return {"kind": "toy_calibration", "shift": self.shift}

    io.register_loader("toy_calibration", lambda state, device: Calibration(state["shift"]), overwrite=True)
    io.save(Calibration(np.float64(0.1)), tmp_path / "cal.pt")
    assert io.load(tmp_path / "cal.pt").shift == 0.1
    with pytest.raises(ValueError, match="already registered"):
        io.register_loader("toy_calibration", lambda state, device: None)
    assert {"nle", "ensemble", "prior"} <= set(io.LOADERS)

    x = np.random.default_rng(0).normal(2.0, 3.0, size=(50, 3))
    for scaler in (sk.ZScore().fit(x), sk.RobustAsinh().fit(x)):
        io.save(scaler, tmp_path / "scaler.pt")
        assert np.allclose(io.load(tmp_path / "scaler.pt")(x), scaler(x))


# ------------------------------------------------------------------------------ run records
def test_run_record_inputs(tmp_path):
    @dataclasses.dataclass
    class Settings:
        lr: float = 1e-3
        widths: tuple = (8, 4)
        best: float = float("nan")

    data = tmp_path / "data.npz"
    np.savez(data, a=np.arange(3))
    splits = {"train": [0, 1, 2], "val": np.array([3, 4])}
    rec = io.run_record(tmp_path / "out" / "run.json", config=Settings(), data_files=data, splits=splits,
                        extra={"metrics": {"loss": np.nan, "fom": np.float32(np.inf), "curve": np.arange(3) / 2,
                                           "t": torch.ones(2)}})
    text = (tmp_path / "out" / "run.json").read_text()

    def strict(constant):
        raise AssertionError(f"{constant} in a JSON record")

    assert json.loads(text, parse_constant=strict) == rec
    assert rec["data_files"] == {str(data): io.file_md5(data)}                # one file, not one per character
    assert rec["config"] == {"lr": 1e-3, "widths": [8, 4], "best": None}
    assert rec["splits"] == sk.Splits(splits).hash()
    assert rec["metrics"] == {"loss": None, "fom": None, "curve": [0.0, 0.5, 1.0], "t": [1.0, 1.0]}

    rec = io.run_record(tmp_path / "run2.json", config={"flow": sk.FlowConfig(model="maf"), "mcmc": sk.MCMCConfig()},
                        data_files=[data, str(data)], splits=sk.Splits(splits))
    assert rec["config"]["flow"]["model"] == "maf" and rec["config"]["mcmc"]["nwalkers"] == sk.MCMCConfig().nwalkers
    assert list(rec["data_files"]) == [str(data)] and rec["splits"] == sk.Splits(splits).hash()
    with pytest.raises(ValueError, match="config"):
        io.run_record(tmp_path / "run3.json", extra={"config": {"a": 1}})
    with pytest.raises(FileNotFoundError):
        io.run_record(tmp_path / "run3.json", data_files=tmp_path / "missing.npz")
    assert not (tmp_path / "run3.json").exists()


def test_git_hash_only_for_own_checkout(tmp_path):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    project = tmp_path / "project"
    package = project / ".venv" / "site-packages" / "sbikit"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (project / "analysis.py").write_text("x = 1\n")
    (project / ".gitignore").write_text(".venv\n")
    try:
        for cmd in (["init", "-q"], ["add", "analysis.py", ".gitignore"], ["commit", "-q", "-m", "x"]):
            subprocess.run(["git", "-C", str(project), *cmd], check=True, env=env, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not usable here")
    assert io._git(project) is not None                                       # the analysis repository has a hash
    assert io._git(package, package="sbikit") is None                         # ... which is not sbikit's
    (project / "pyproject.toml").write_text('[project]\nname = "sbikit"\n')   # right name, package not tracked
    assert io._git(package, package="sbikit") is None
    assert io._git(tmp_path / "nowhere", package="sbikit") is None


# ------------------------------------------------------------------------------ search spaces
def test_suggest_specs():
    study = make_study()
    cfg = suggest(study.ask(), {"lr": {"low": 1e-4, "high": 1e-2, "log": True}, "n": {"low": 2, "high": 6, "int": True},
                                "g": {"low": 0.0, "high": 1.0, "step": 0.25},
                                "k": {"low": 2, "high": 10, "int": True, "step": 4},
                                "widths": {"choices": ["16,8", "8"]}, "act": {"choices": ["relu", None]}, "fixed": 3,
                                "kwargs": {"value": {"residual": True}}, "tuple": (1, 2)})
    assert 1e-4 <= cfg["lr"] <= 1e-2 and cfg["n"] in range(2, 7) and isinstance(cfg["n"], int)
    assert cfg["g"] in (0.0, 0.25, 0.5, 0.75, 1.0) and cfg["k"] in (2, 6, 10)
    assert cfg["widths"] in ((16, 8), (8,)) and cfg["act"] in ("relu", None)
    assert cfg["fixed"] == 3 and cfg["kwargs"] == {"residual": True} and cfg["tuple"] == (1, 2)


@pytest.mark.parametrize("spec, message", [
    ({"low": 1e-4, "high": 1e-2, "logg": True}, "did you mean 'log'"),
    ({"min": 0, "max": 1}, "not a valid spec"),
    ({"low": 0.0}, "'low' and 'high' are both needed"),
    ({"choices": [1, 2], "log": True}, "cannot be combined"),
    ({"choices": []}, "non-empty list"),
    ({"choices": "abc"}, "non-empty list"),
    ({"choices": [(8, 4), (4,)]}, "write network widths as strings"),
    ({"low": 2.5, "high": 6, "int": True}, "integer range"),
    ({"low": 0.0, "high": 1.0, "log": True}, "low > 0"),
    ({"low": 2.0, "high": 1.0}, "low <= high"),
    ({"low": 1.0, "high": 2.0, "step": 0.5, "log": True}, "'step'"),
    ({"residual": True}, "not a valid spec"),
    ({"value": 1, "x": 2}, r"'value' cannot be combined with \['x'\]"),
])
def test_malformed_spec_raises(spec, message):
    study = make_study()
    with pytest.raises(ValueError, match=message):
        suggest(study.ask(), {"x": spec})
    with pytest.raises(ValueError, match="search space entry 'x'"):
        tune_compressor(toy_factory, {"x": spec}, n_trials=1)


# ------------------------------------------------------------------------------ studies
def test_make_study_storages(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                        # no ExperimentalWarning either
        studies = [make_study(), make_study(None, "a", constrained=True), make_study(tmp_path / "sub" / "s.db", "a"),
                   make_study(str(tmp_path / "j" / "s.log"), "a"), make_study(f"sqlite:///{tmp_path / 'url.db'}", "a"),
                   make_study(optuna.storages.InMemoryStorage(), "a"),
                   make_study(None, "a", sampler=optuna.samplers.RandomSampler(seed=1))]
        for study in studies:
            run_study(study, lambda t: t.suggest_float("x", 0, 1), 2)
    assert all(len(s.trials) == 2 for s in studies)
    assert isinstance(studies[-1].sampler, optuna.samplers.RandomSampler)
    assert all((tmp_path / f).exists() for f in ("sub/s.db", "j/s.log", "url.db"))
    assert not Path("sqlite:").exists()
    with pytest.raises(TypeError, match="storage"):
        make_study(3, "a")
    with pytest.raises(ValueError, match="direction"):
        make_study(None, "a", "minimise")


def test_direction_mismatch_raises(tmp_path, device):
    make_study(tmp_path / "s.log", "search", "minimize")
    assert make_study(tmp_path / "s.log", "search", "minimize").direction.name == "MINIMIZE"
    with pytest.raises(ValueError, match="exists with direction.*minimize.*not 'maximize'"):
        make_study(tmp_path / "s.log", "search", "maximize")
    s, theta = _sims(100)
    tune_nle(s, theta, n_trials=1, storage=tmp_path / "n.log", name="nle", space=NLE_SPACE, train=NLE_TRAIN,
             device=device)
    with pytest.raises(ValueError, match="exists with direction"):
        tune_nle(s, theta, n_trials=2, storage=tmp_path / "n.log", name="nle", space=NLE_SPACE, train=NLE_TRAIN,
                 objective="fom_tarp", evaluation=EvalSet(s[:4], theta[:4], sk.Prior([-1, -1], [1, 1]), ["a", "b"]),
                 fom_params=["a"])


def test_resume_draws_new_configurations_and_respects_n_trials(tmp_path):
    def objective(trial):
        return trial.suggest_float("x", 0, 1) ** 2

    for i in range(3):                                       # three runs of two trials each, all in the start-up phase
        study = run_study(make_study(tmp_path / "s.log", "r", n_startup=10, seed=0), objective, 2 * (i + 1))
    xs = [t.params["x"] for t in study.trials]
    assert len(xs) == 6 and len(set(xs)) == 6
    run_study(make_study(tmp_path / "s.log", "r"), objective, 6)              # already there: nothing runs
    assert len(make_study(tmp_path / "s.log", "r").trials) == 6

    one = run_study(make_study(tmp_path / "one.log", "r", seed=0), objective, 6)
    two = run_study(make_study(tmp_path / "two.log", "r", seed=0), objective, 6)
    assert [t.params for t in one.trials] == [t.params for t in two.trials]   # an uninterrupted search is reproducible
    other = run_study(make_study(tmp_path / "three.log", "r", seed=1), objective, 6)
    assert [t.params for t in one.trials] != [t.params for t in other.trials]


def test_run_study_counts_other_workers_and_catches(tmp_path):
    study = make_study(tmp_path / "s.log", "w")
    twin = make_study(tmp_path / "s.log", "w", seed=1)       # a second worker on the same study

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        if trial.number < 4:                                 # meanwhile the other worker finishes a trial
            t = twin.ask()
            twin.tell(t, 1.0 + x)
        return x

    run_study(study, objective, 6)
    assert len(study.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED))) == 6

    def flaky(trial):
        if trial.number % 2 == 0:
            raise FloatingPointError("diverged")
        return trial.suggest_float("x", 0, 1)

    with pytest.raises(FloatingPointError):
        run_study(make_study(), flaky, 3)
    study = run_study(make_study(), flaky, 3, catch=(FloatingPointError,))
    assert [t.state.name for t in study.trials].count("COMPLETE") == 3 and len(study.trials) == 6

    def broken(trial):
        raise FloatingPointError("always")

    with pytest.warns(RuntimeWarning, match="failed in a row"):
        study = run_study(make_study(), broken, 3, catch=(FloatingPointError,))
    assert len(study.trials) == hpt._MAX_FAILURES


def test_prune_threshold_uses_all_finished_trials():
    rng = np.random.default_rng(0)
    study = make_study()
    assert hpt._prune_threshold(study, 0.75, 10) == np.inf
    pruned = []
    for i in range(400):
        val = float(rng.normal())
        prune = val > hpt._prune_threshold(study, 0.75, 10)
        trial = study.ask()
        trial.set_user_attr("val_loss", val)
        study.tell(trial, state=TrialState.PRUNED) if prune else study.tell(trial, -val)
        pruned.append(prune)
    assert not any(pruned[:10])
    assert 0.17 < np.mean(pruned[10:]) < 0.33                # about 1 - prune_quantile, not far more
    assert hpt._prune_threshold(study, None, 10) == np.inf
    trial = study.ask()
    trial.set_user_attr("val_loss", None)                    # a failed fit does not enter the reference
    study.tell(trial, state=TrialState.PRUNED)
    assert np.isfinite(hpt._prune_threshold(study, 0.75, 10))


# ------------------------------------------------------------------------------ compressors
def test_tune_compressor_in_memory_and_reproducible():
    state = torch.random.get_rng_state()
    cfg, comp, study = tune_compressor(toy_factory, TOY_SPACE, n_trials=6, n_startup=3, seed=3)
    assert torch.equal(torch.random.get_rng_state(), state)                  # the caller's random stream is untouched
    assert len(study.trials) == 6 and cfg == study.best_trial.user_attrs["config"] and cfg["b"] == 0.5
    assert comp.config == cfg and load_trial(study) is comp
    assert list(hpt._MODELS[study]) == [study.best_trial.number]             # only the best compressor is kept
    cfg2, comp2, study2 = tune_compressor(toy_factory, TOY_SPACE, n_trials=6, n_startup=3, seed=3)
    assert [t.value for t in study.trials] == [t.value for t in study2.trials] and comp2.noise == comp.noise
    _, comp3, study3 = tune_compressor(toy_factory, TOY_SPACE, n_trials=6, n_startup=3, seed=4)
    assert [t.value for t in study.trials] != [t.value for t in study3.trials]
    _, comp4, _ = tune_compressor(toy_factory, TOY_SPACE, n_trials=2, fit_kwargs={"atoms": [1, 2]})
    assert comp4.fit_kwargs == {"atoms": [1, 2]}
    _, comp5, study5 = tune_compressor(toy_factory, TOY_SPACE, n_trials=2, seed=None)       # unseeded searches run too
    assert comp5 is not None and len(study5.trials) == 2


def test_tune_compressor_validates_before_training(tmp_path):
    with pytest.raises(TypeError, match="no field"):
        tune_compressor(toy_factory, {"a": {"low": 0.0, "high": 1.0}, "bb": 1.0}, n_trials=2,
                        storage=tmp_path / "s.log")
    with pytest.raises(TypeError, match="no field"):
        tune_compressor(toy_factory, TOY_SPACE, n_trials=2, storage=tmp_path / "s.log", fixed={"c": 1.0})
    assert not (tmp_path / "s.log").exists()                                  # no study, no trial
    with pytest.raises(ValueError, match="finite validation loss"):
        tune_compressor(toy_factory, {"fail": True}, n_trials=2)

    def picky(cfg):                                          # every value an entry can take is tried, not just one
        if cfg["act"] not in ("relu", "tanh") or not 0.0 <= cfg["a"] < 1.0:
            raise ValueError(f"bad config {cfg}")
        return Toy(cfg)

    made = []
    spaces = (({"a": 0.5, "act": {"choices": ["relu", "tanh", "rleu"]}}, "search space entry 'act' = 'rleu'"),
              ({"a": {"low": 0.0, "high": 1.0}, "act": "relu"}, "search space entry 'a' = 1.0: bad config"))
    for space, message in spaces:
        with pytest.raises(ValueError, match=message):
            tune_compressor(lambda c: made.append(c) or picky(c), space, n_trials=4, storage=tmp_path / "p.log")
    assert not (tmp_path / "p.log").exists() and [c["act"] for c in made[:3]] == ["relu", "tanh", "rleu"]


def test_tune_compressor_fixed(tmp_path):
    kw = dict(storage=tmp_path / "s.log", name="toy", outdir=tmp_path)
    cfg, comp, study = tune_compressor(toy_factory, TOY_SPACE, n_trials=3, n_startup=3, **kw)
    best = study.best_trial.number
    # on a study that has trials already: one new trial with exactly this configuration
    fixed = {"a": 0.9, "b": 2.0, "widths": "8,4"}            # b is not searched, widths is not in the space at all
    cfg_f, comp_f, study = tune_compressor(toy_factory, TOY_SPACE, n_trials=3, fixed=fixed, **kw)
    assert cfg_f == {"a": 0.9, "b": 2.0, "widths": [8, 4]} and comp_f.config == {"a": 0.9, "b": 2.0, "widths": (8, 4)}
    assert len(study.trials) == 4 and study.trials[3].params == {"a": 0.9} and study.trials[3].user_attrs["fixed"]
    assert study.trials[3].value == pytest.approx(0.36 + 2.0, abs=2e-3) and study.best_trial.number == best
    rec = json.loads((tmp_path / "toy" / "trials" / "0003.json").read_text())
    assert rec["config"] == cfg_f and rec["fixed"] is True and rec["seed"] == 3
    assert io.load(tmp_path / "toy" / "trials" / "0003.pt").noise == comp_f.noise
    # constants of the space are part of the configuration; values outside the ranges are honoured too
    cfg_o, comp_o, study = tune_compressor(toy_factory, TOY_SPACE, fixed={"a": 5.0}, **kw)
    assert comp_o.config == {"a": 5.0, "b": 0.5} and study.trials[4].params == {} and len(study.trials) == 5
    with pytest.warns(UserWarning, match=r"no value for the searched keys \['a'\]"):
        _, comp_d, _ = tune_compressor(toy_factory, TOY_SPACE, fixed={"b": 0.0})
    assert comp_d.config == {"b": 0.0}
    with pytest.raises(TypeError, match="fixed must be a dict"):
        tune_compressor(toy_factory, TOY_SPACE, fixed=["a"])
    # a first trial on a new study, and the same seed gives the same training
    _, one, _ = tune_compressor(toy_factory, TOY_SPACE, fixed={"a": 0.1}, seed=5)
    _, two, _ = tune_compressor(toy_factory, TOY_SPACE, fixed={"a": 0.1}, seed=5)
    _, three, _ = tune_compressor(toy_factory, TOY_SPACE, fixed={"a": 0.1}, seed=6)
    assert one.noise == two.noise != three.noise


def test_best_compressor_is_never_none(tmp_path):
    kw = dict(n_trials=4, n_startup=4, storage=tmp_path / "s.db", name="toy")
    cfg, comp, study = tune_compressor(toy_factory, TOY_SPACE, **kw)
    # resumed without an output folder, nothing new to run: the best configuration is trained again with its seed
    cfg2, comp2, study2 = tune_compressor(toy_factory, TOY_SPACE, **kw)
    assert comp2 is not None and comp2 is not comp and cfg2 == cfg
    assert comp2.config == cfg and comp2.noise == comp.noise
    # with an output folder it comes from disk, and every trial has a record
    kw = dict(storage=tmp_path / "t.db", name="toy", outdir=tmp_path / "out")
    cfg, comp, study = tune_compressor(toy_factory, TOY_SPACE, n_trials=3, n_startup=3, **kw)
    cfg2, comp2, study2 = tune_compressor(toy_factory, TOY_SPACE, n_trials=3, n_startup=3, **kw)
    assert comp2 is not comp and comp2.noise == comp.noise and cfg2 == cfg
    records = sorted((tmp_path / "out" / "toy" / "trials").glob("*.json"))
    assert len(records) == 3
    assert {"trial", "config", "val_loss", "seconds", "seed"} <= set(json.loads(records[0].read_text()))
    assert load_trial(study2, outdir=tmp_path / "out").noise == comp.noise


def test_best_compressor_is_retrained_with_its_trial_seed(tmp_path, device):
    rng = np.random.default_rng(3)
    t = rng.uniform(-1, 1, size=(300, 2))
    x = np.hstack([t, t ** 2]).astype(np.float32)
    space = {"widths": {"choices": ["8", "8,8"]}, "lr": {"low": 1e-3, "high": 1e-2, "log": True}, "epochs": 3,
             "patience": 2}
    tol = dict(rtol=1e-6 if device == "cpu" else 1e-2, atol=0)

    def factory(cfg):
        return sk.compress.MSECompressor(2, cfg, device)     # no seed: the hold-out split follows the trial's seed

    # without a validation set the compressor holds one out; the same seed gives the same trials
    one = tune_compressor(factory, space, x, t, n_trials=3, n_startup=3, seed=4, storage=tmp_path / "s.log")
    two = tune_compressor(factory, space, x, t, n_trials=3, n_startup=3, seed=4)
    assert np.allclose([tr.value for tr in one[2].trials], [tr.value for tr in two[2].trials], **tol)
    assert one[1].seed == 4000 + one[2].best_trial.number
    assert [tr.user_attrs["seed"] for tr in one[2].trials] == [4000, 4001, 4002]
    other = tune_compressor(factory, space, x, t, n_trials=3, n_startup=3, seed=5)
    assert not np.allclose([tr.value for tr in one[2].trials], [tr.value for tr in other[2].trials], **tol)

    # a second worker (another seed, no output folder) gets the compressor of the best trial, not another one
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg, comp, study = tune_compressor(factory, space, x, t, n_trials=3, n_startup=3, seed=9,
                                           storage=tmp_path / "s.log")
    assert comp is not one[1] and cfg == one[0] and len(study.trials) == 3
    assert comp.best_val_loss == pytest.approx(study.best_value, rel=tol["rtol"])
    assert np.allclose(comp(x[:5]), one[1](x[:5]), rtol=tol["rtol"], atol=1e-6)

    # a configuration that no longer trains to the trial's loss is returned with a warning and not written to disk
    with pytest.warns(RuntimeWarning, match="is not the one the search found"):
        _, changed, _ = tune_compressor(factory, space, x[:200], t[:200], n_trials=3, storage=tmp_path / "s.log",
                                        outdir=tmp_path / "out")
    assert changed is not None and not list((tmp_path / "out").rglob("*.pt"))


def test_load_trial_leaves_models_in_memory_on_their_device(tmp_path):
    class Movable(Toy):
        kind = "toy_hpt"
        device = "elsewhere"

        def to(self, device):
            self.device = device
            return self

    cfg, comp, study = tune_compressor(Movable, {"a": {"low": 0.0, "high": 1.0}}, n_trials=2, outdir=tmp_path)
    assert load_trial(study) is comp and comp.device == "elsewhere"          # the default moves nothing
    assert load_trial(study, device="cpu") is comp and comp.device == "cpu"  # an explicit device moves it in place
    again = make_study()                                                     # a study this process holds no models for
    again.add_trials(study.trials)
    assert load_trial(again, outdir=tmp_path, name=study.study_name).noise == comp.noise


def test_tune_compressor_real_network(device):
    rng = np.random.default_rng(2)
    t = rng.uniform(-1, 1, size=(300, 2))
    x = np.hstack([t, t ** 2]).astype(np.float32)
    space = {"widths": {"choices": ["8", "8,8"]}, "lr": 3e-3, "epochs": 3, "patience": 2}
    cfg, comp, study = tune_compressor(lambda c: sk.compress.MSECompressor(2, c, device), space, x, t, n_trials=2,
                                       n_startup=2)
    assert comp(x[:3]).shape == (3, 2) and len(study.trials) == 2
    cfg_f, comp_f, _ = tune_compressor(lambda c: sk.compress.MSECompressor(2, c, device), space, x, t, x[:50], t[:50],
                                       fixed={"widths": [8, 8], "dropout": 0.1})
    assert tuple(comp_f.config.widths) == (8, 8) and comp_f.config.dropout == 0.1 and comp_f.config.epochs == 3
    with pytest.raises(TypeError, match="did you mean 'dropout'"):
        tune_compressor(lambda c: sk.compress.MSECompressor(2, c, device), {**space, "dropuot": 0.1}, x, t, n_trials=2)


# ------------------------------------------------------------------------------ NLE searches
def test_tune_nle_rejects_bad_arguments_before_training(tmp_path, monkeypatch):
    s, theta = _sims(100)

    def no_training(*args, **kwargs):
        raise AssertionError("a trial was started")

    monkeypatch.setattr(sk.NLE, "fit", no_training)
    kw = dict(n_trials=2, storage=tmp_path / "s.log", train=NLE_TRAIN)
    with pytest.raises(TypeError, match="'hiden_features'.*did you mean 'hidden_features'"):
        tune_nle(s, theta, space={"hiden_features": {"choices": [8, 16]}, "lr": 1e-3}, **kw)
    with pytest.raises(TypeError, match="did you mean 'lr'"):
        tune_nle(s, theta, space={"lrr": {"low": 1e-4, "high": 1e-3}}, **kw)
    with pytest.raises(TypeError, match="max_epoch"):
        tune_nle(s, theta, space=NLE_SPACE, n_trials=2, storage=tmp_path / "s.log", train={"max_epoch": 3})
    with pytest.raises(ValueError, match="search space entry 'model' = 'nfs'"):
        tune_nle(s, theta, space={"model": {"choices": ["maf", "nfs"]}}, **kw)
    with pytest.raises(ValueError, match="search space entry 'lr' = -1"):
        tune_nle(s, theta, space={"lr": {"low": -1.0, "high": 1e-3}}, **kw)
    with pytest.raises(ValueError, match="search space entry 'lr' = -1.0: TrainConfig"):     # not blamed on 'model'
        tune_nle(s, theta, space={"model": "maf", "num_transforms": 1, "lr": {"low": -1.0, "high": 1.0}}, **kw)
    with pytest.raises(ValueError, match="search space entry 'lr' = 0.0: TrainConfig"):
        tune_nle(s, theta, space={"model": {"choices": ["maf", "nsf"]}, "lr": {"choices": [1e-3, 0.0]}}, **kw)
    with pytest.raises(ValueError, match=r"'choices'"):
        tune_nle(s, theta, space={"hidden_features": [8, 16]}, **kw)
    with pytest.raises(ValueError, match="objective"):
        tune_nle(s, theta, space=NLE_SPACE, objective="fom", **kw)
    with pytest.raises(ValueError, match="t_trn must be 2D"):
        tune_nle(s, theta[:, 0], space=NLE_SPACE, **kw)
    with pytest.raises(ValueError, match="both s_val and t_val"):
        tune_nle(s, theta, s[:10], space=NLE_SPACE, **kw)
    with pytest.raises(ValueError, match="s_val has 3 columns"):
        tune_nle(s, theta, np.zeros((10, 3)), theta[:10], space=NLE_SPACE, **kw)

    prior = sk.Prior([-1, -1], [1, 1])
    fom = dict(space=NLE_SPACE, objective="fom_tarp", **kw)
    with pytest.raises(ValueError, match="needs evaluation"):
        tune_nle(s, theta, fom_params=["a"], **fom)
    with pytest.raises(ValueError, match="needs fom_params"):
        tune_nle(s, theta, evaluation=EvalSet(s[:5], theta[:5], prior, ["a", "b"]), **fom)
    with pytest.raises(ValueError, match=r"fom_params 'A' \(did you mean 'a'\?\); valid: \['a', 'b'\]"):
        tune_nle(s, theta, evaluation=EvalSet(s[:5], theta[:5], prior, ["a", "b"]), fom_params=["A"], **fom)
    with pytest.raises(ValueError, match="tarp_params 'c'"):
        tune_nle(s, theta, evaluation=EvalSet(s[:5], theta[:5], prior, ["a", "b"]), fom_params="a", tarp_params=["c"],
                 **fom)
    with pytest.raises(ValueError, match="evaluation.s has 3 columns"):
        tune_nle(s, theta, evaluation=EvalSet(np.zeros((5, 3)), theta[:5], prior, ["a", "b"]), fom_params=["a"], **fom)
    with pytest.raises(ValueError, match="evaluation.s has 5 rows but evaluation.truth has 4"):
        tune_nle(s, theta, evaluation=EvalSet(s[:5], theta[:4], prior, ["a", "b"]), fom_params=["a"], **fom)
    with pytest.raises(ValueError, match="param_names"):
        tune_nle(s, theta, evaluation=EvalSet(s[:5], theta[:5], prior, ["a"]), fom_params=["a"], **fom)
    with pytest.raises(ValueError, match=r"same value in all 1 simulation\(s\) for the parameter\(s\) \['a', 'b'\]"):
        tune_nle(s, theta, evaluation=EvalSet(s[:1], theta[:1], prior, ["a", "b"]), fom_params=["a"], **fom)
    with pytest.raises(ValueError, match=r"same value in all 5 simulation\(s\) for the parameter\(s\) \['b'\]"):
        tune_nle(s, theta, evaluation=EvalSet(s[:5], np.column_stack([theta[:5, 0], np.zeros(5)]), prior, ["a", "b"]),
                 fom_params=["a"], **fom)
    with pytest.raises(ValueError, match="2 priors for 5 simulations"):
        tune_nle(s, theta, evaluation=EvalSet(s[:5], theta[:5], [prior, prior], ["a", "b"]), fom_params=["a"], **fom)
    with pytest.raises(ValueError, match="single sampling prior"):
        tune_nle(s, theta, evaluation=EvalSet(s[:5], theta[:5], [prior] * 5, ["a", "b"], fom_prior=prior),
                 fom_params=["a"], **fom)
    with pytest.raises(ValueError, match="init_sd"):
        tune_nle(s, theta, evaluation=EvalSet(s[:5], theta[:5], prior, ["a", "b"], init_sd=np.ones(3)),
                 fom_params=["a"], **fom)
    with pytest.raises(TypeError, match="MCMCConfig has no field"):
        tune_nle(s, theta, evaluation=EvalSet(s[:5], theta[:5], prior, ["a", "b"]), fom_params=["a"],
                 mcmc={"nwalker": 8}, **fom)
    with pytest.raises(ValueError, match="prune_quantile"):
        tune_nle(s, theta, evaluation=EvalSet(s[:5], theta[:5], prior, ["a", "b"]), fom_params=["a"],
                 prune_quantile=75, **fom)
    assert not (tmp_path / "s.log").exists()                                  # none of them got as far as a study


def test_tune_nle_reproducible_in_memory(tmp_path, device):
    s, theta = _sims()
    kw = dict(n_trials=3, space=NLE_SPACE, train=NLE_TRAIN, n_startup=3, device=device)
    torch.manual_seed(11)
    state = torch.random.get_rng_state()
    one = tune_nle(s[:300], theta[:300], s[300:], theta[300:], seed=2, **kw)
    assert torch.equal(torch.random.get_rng_state(), state)                  # the caller's random stream is untouched
    torch.manual_seed(12)
    two = tune_nle(s[:300], theta[:300], s[300:], theta[300:], seed=2, **kw)
    rtol = 1e-6 if device == "cpu" else 1e-2
    assert [t.params for t in one.trials] == [t.params for t in two.trials]
    assert np.allclose([t.value for t in one.trials], [t.value for t in two.trials], rtol=rtol, atol=0)
    three = tune_nle(s[:300], theta[:300], s[300:], theta[300:], seed=3, **kw)
    assert not np.allclose([t.value for t in one.trials], [t.value for t in three.trials], rtol=rtol, atol=0)
    assert not list(tmp_path.iterdir()) and len(one.trials) == 3            # nothing is written without outdir

    # the flows are in memory; without a validation set every flow holds one out
    best = load_trial(one, device=device)
    assert best.val_loss(s[300:], theta[300:]) == pytest.approx(one.best_value, rel=1e-3)
    assert load_trial(one, 1, device=device) is not best or one.best_trial.number == 1
    with pytest.raises(ValueError, match="no trial 7"):
        load_trial(one, 7)
    auto = tune_nle(s, theta, n_trials=1, space=NLE_SPACE, train=NLE_TRAIN, device=device)
    assert np.isfinite(auto.best_value)
    with pytest.warns(UserWarning, match="only by objective='fom_tarp'"):
        tune_nle(s, theta, n_trials=1, space=NLE_SPACE, train=NLE_TRAIN, fom_params=["a"], device=device)


def test_tune_nle_outdir_resume_and_load_trial(tmp_path, device, monkeypatch):
    s, theta = _sims()
    kw = dict(storage=tmp_path / "s.log", name="vl", outdir=tmp_path, space=NLE_SPACE, train=NLE_TRAIN, n_startup=4,
              device=device)
    study = tune_nle(s[:300], theta[:300], s[300:], theta[300:], 2, **kw)
    rec = json.loads((tmp_path / "vl" / "trials" / "0001.json").read_text())
    assert rec["val_loss"] == pytest.approx(study.trials[1].value) and rec["seed"] == 1
    assert rec["train"]["max_epochs"] == 3
    study = tune_nle(s[:300], theta[:300], s[300:], theta[300:], 4, **kw)    # resumes: two more trials, new configs
    assert len(study.trials) == 4 and len({t.params["lr"] for t in study.trials}) == 4
    assert sorted(p.name for p in (tmp_path / "vl" / "trials").glob("*.pt")) == [f"{i:04d}.pt" for i in range(4)]
    assert list(hpt._MODELS[study]) == [2, 3]                                # this call trained trials 2 and 3

    number = best_trials(study, 1)[0].number
    from_memory_or_disk = load_trial(study, outdir=tmp_path, device=device)
    from_disk = io.load(tmp_path / "vl" / "trials" / f"{number:04d}.pt", device)
    ll = from_disk.log_likelihood(s[:5], theta[:5]).cpu().numpy()
    assert np.allclose(from_memory_or_disk.log_likelihood(s[:5], theta[:5]).cpu().numpy(), ll, rtol=1e-5)
    first = load_trial(study, 0, tmp_path, device=device)                    # trained by the first call: from disk
    assert first.val_loss(s[300:], theta[300:]) == pytest.approx(study.trials[0].value, rel=1e-3)
    with pytest.raises(ValueError, match="pass the search's outdir"):
        load_trial(study, 0)
    with pytest.raises(FileNotFoundError, match="neither in memory nor at"):
        load_trial(study, 0, tmp_path / "elsewhere")

    monkeypatch.setattr(hpt, "KEEP_IN_MEMORY", 2)                            # only the best flows stay in memory
    study = tune_nle(s[:300], theta[:300], s[300:], theta[300:], n_trials=4, space=NLE_SPACE, train=NLE_TRAIN,
                     device=device)
    assert sorted(hpt._MODELS[study]) == sorted(t.number for t in best_trials(study, 2))


def test_tune_nle_callable_objective(tmp_path, device):
    s, theta = _sims()
    seen = []

    def objective(nle, trial):
        assert isinstance(nle, sk.NLE) and nle.fitted
        seen.append(trial.number)
        value = nle.val_loss(s[300:350], theta[300:350])
        return value, [0.5 if trial.number == 0 else -0.5]                   # the first trial is infeasible

    study = tune_nle(s[:300], theta[:300], s[300:], theta[300:], 3, outdir=tmp_path, name="own", space=NLE_SPACE,
                     train=NLE_TRAIN, objective=objective, n_startup=3, device=device)
    assert seen == [0, 1, 2] and study.direction.name == "MINIMIZE"
    assert 0 not in [t.number for t in best_trials(study, 3)] and len(best_trials(study, 3, feasible_only=False)) == 3
    assert study.trials[0].user_attrs["constraints"] == [0.5]
    rec = json.loads((tmp_path / "own" / "trials" / "0001.json").read_text())
    assert rec["objective"] == pytest.approx(study.trials[1].value) and rec["constraints"] == [-0.5]

    plain = tune_nle(s[:300], theta[:300], n_trials=2, space=NLE_SPACE, train=NLE_TRAIN, device=device,
                     objective=lambda nle, trial: float(trial.number))       # a plain value: no constraints
    assert [t.value for t in plain.trials] == [0.0, 1.0] and len(best_trials(plain, 2)) == 2

    def sometimes_nan(nle, trial):
        return float("nan") if trial.number == 0 else 1.0

    study = tune_nle(s[:300], theta[:300], n_trials=2, space=NLE_SPACE, train=NLE_TRAIN, device=device,
                     objective=sometimes_nan, outdir=tmp_path, name="nan")
    assert [t.state.name for t in study.trials] == ["PRUNED", "COMPLETE"]
    text = (tmp_path / "nan" / "trials" / "0000.json").read_text()
    assert "NaN" not in text and json.loads(text)["objective"] is None      # records stay valid JSON


def test_tune_nle_fom_tarp_small(tmp_path, device):
    s, theta = _sims(500)
    ev = EvalSet(s[400:412], theta[400:412], sk.Prior([-1.5, -1.5], [1.5, 1.5]), ["a", "b"])
    mcmc = {"nwalkers": 16, "burn_in": 100, "block": 100, "max_steps": 300, "thin": 2, "min_ess": 50, "n_samples": 200}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                      # short chains may not converge
        study = tune_nle(s[:300], theta[:300], s[300:400], theta[300:400], 2, outdir=tmp_path, name="ft",
                         space=NLE_SPACE, train={"max_epochs": 5}, objective="fom_tarp", evaluation=ev, mcmc=mcmc,
                         fom_params="a", n_references=5, n_bootstrap=10, n_startup=2, device=device)
    assert study.direction.name == "MAXIMIZE" and len(study.trials) == 2
    text = (tmp_path / "ft" / "trials" / "0001.json").read_text()
    rec = json.loads(text, parse_constant=lambda c: pytest.fail(f"{c} in a JSON record"))
    assert {"objective", "constraints", "metrics", "converged", "mcmc_steps"} <= set(rec)
    assert len(rec["constraints"]) == 5
    assert (tmp_path / "ft" / "trials" / "0001_posteriors.npz").exists()
    assert study.trials[1].user_attrs["constraints"] == rec["constraints"]
