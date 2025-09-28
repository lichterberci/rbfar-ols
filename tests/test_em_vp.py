import importlib.util
import sys
from pathlib import Path
from typing import Tuple

import pytest
import torch


@pytest.fixture(scope="module")
def em_vp_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "em-vp.py"
    spec = importlib.util.spec_from_file_location("em_vp_module", str(mod_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _make_synthetic_data(
    n_samples: int = 80, seed: int = 123, return_params: bool = False
) -> (
    Tuple[torch.Tensor, torch.Tensor]
    | Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
):
    torch.manual_seed(seed)
    p = 3

    X = torch.randn(n_samples, p)

    centres_true = torch.tensor(
        [[1.0, -0.7, 0.2], [-1.2, 0.9, -0.4]], dtype=torch.float32
    )
    widths_true = torch.tensor([0.6, 0.9], dtype=torch.float32)
    weights_true = torch.tensor(
        [[0.8, -0.4, 0.3], [-0.5, 0.7, -0.2]], dtype=torch.float32
    )
    noise_std = torch.tensor([0.05, 0.08], dtype=torch.float32)
    log_pi = torch.log(torch.tensor([0.55, 0.45], dtype=torch.float32))

    diff = X.unsqueeze(1) - centres_true.unsqueeze(0)
    sq_norm = (diff * diff).sum(dim=2)
    widths_sq = widths_true.pow(2)
    features = torch.exp(-sq_norm / (2.0 * widths_sq.unsqueeze(0)))
    logits = log_pi.unsqueeze(0) + torch.log(features + 1e-9)

    assignments = torch.distributions.Categorical(logits=logits).sample()

    y = torch.empty(n_samples)
    for i in range(n_samples):
        comp = assignments[i]
        mean = X[i] @ weights_true[comp]
        y[i] = mean + noise_std[comp] * torch.randn(())

    if return_params:
        return (
            X,
            y.unsqueeze(1),
            {
                "centres": centres_true,
                "widths": widths_true,
                "weights": weights_true,
            },
        )

    return X, y.unsqueeze(1)


def test_em_vp_converges_and_predicts(em_vp_module):
    EMVPConfig = em_vp_module.EMVPConfig
    EMVPTrainer = em_vp_module.EMVPTrainer

    X, y, params = _make_synthetic_data(return_params=True)

    cfg = EMVPConfig(
        num_components=2,
        max_iters=75,
        tol_param=1e-4,
        tol_loglik=1e-4,
        device="cpu",
        responsibility_floor=1e-6,
    )
    trainer = EMVPTrainer(cfg)

    model = trainer.fit(
        X,
        y,
        centres_init=params["centres"],
        widths_init=params["widths"],
    )
    state = model.state

    assert state.responsibilities.shape == (X.shape[0], cfg.num_components)
    row_sums = state.responsibilities.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    loglik = trainer.diagnostics.log_likelihood
    assert len(loglik) >= 2
    assert loglik[-1] >= loglik[0] - 1e-3

    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)

    mse_model = torch.mean((preds - y.squeeze(1)) ** 2)
    mse_baseline = torch.mean((y.squeeze(1) - y.mean()) ** 2)
    assert mse_model < mse_baseline


def test_em_vp_generalises_on_holdout(em_vp_module):
    EMVPConfig = em_vp_module.EMVPConfig
    EMVPTrainer = em_vp_module.EMVPTrainer

    X, y = _make_synthetic_data(n_samples=100, seed=321)

    X_train, X_test = X[:70], X[70:]
    y_train, y_test = y[:70], y[70:]

    cfg = EMVPConfig(num_components=2, max_iters=60, tol_param=1e-4, tol_loglik=1e-4)
    trainer = EMVPTrainer(cfg)

    model = trainer.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape[0] == X_test.shape[0]
    assert torch.isfinite(preds).all()

    mse_test = torch.mean((preds - y_test.squeeze(1)) ** 2)
    mse_baseline = torch.mean((y_test.squeeze(1) - y_train.mean()) ** 2)
    assert mse_test < mse_baseline
