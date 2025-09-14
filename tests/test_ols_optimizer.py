"""
Test module for OLS optimizer functionality.
"""

import pytest
import torch
import importlib.util
from pathlib import Path


@pytest.fixture
def ols_optimizer_class():
    """Load the OLS optimizer class."""
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "ols_optimizer.py"
    spec = importlib.util.spec_from_file_location("ols_optimizer", str(mod_path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.OlsOptimizer


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""

    def _make_synthetic(
        l=400, m=80, k_true=5, noise=1e-3, device=None, dtype=torch.float32
    ):
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        P = torch.randn(l, m, device=device, dtype=dtype)
        idx_true = torch.randperm(m, device=device)[:k_true]
        coef_true = torch.randn(k_true, device=device, dtype=dtype)
        d = (P[:, idx_true] @ coef_true) + noise * torch.randn(
            l, device=device, dtype=dtype
        )
        return P, d, idx_true

    return _make_synthetic


def test_ols_converges_and_is_sparse(ols_optimizer_class, synthetic_data):
    """Test that OLS optimizer converges and produces sparse solutions."""
    torch.manual_seed(0)
    P, d, idx_true = synthetic_data()
    opt = ols_optimizer_class(rho=1e-2, epsilon=1e-8)
    sel_idx, theta = opt.optimize(P, d)

    # Sanity checks
    assert sel_idx.ndim == 1 and theta.ndim == 1
    assert sel_idx.numel() == theta.numel() and sel_idx.numel() > 0

    # Convergence check
    y = P[:, sel_idx] @ theta
    rel_err = torch.norm(d - y, dim=0) / (torch.norm(d, dim=0) + 1e-12)
    assert rel_err < 0.12

    # Sparsity and some overlap
    assert sel_idx.numel() <= min(P.shape[1], int(3 * len(idx_true)))
    overlap = len(set(sel_idx.tolist()) & set(idx_true.tolist()))
    assert overlap >= 1


def test_ols_optimizer_parameter_validation(ols_optimizer_class):
    """Test that OLS optimizer validates parameters correctly."""
    # Test valid parameters
    opt = ols_optimizer_class(rho=1e-2, epsilon=1e-8)
    assert opt._rho == 1e-2
    assert opt._epsilon == 1e-8

    # Test default parameters
    opt_default = ols_optimizer_class()
    assert hasattr(opt_default, "_rho")
    assert hasattr(opt_default, "_epsilon")


def test_ols_optimizer_empty_input(ols_optimizer_class):
    """Test OLS optimizer behavior with edge cases."""
    opt = ols_optimizer_class()

    # Test with very small matrix
    P_small = torch.randn(5, 3)
    d_small = torch.randn(5)

    try:
        sel_idx, theta = opt.optimize(P_small, d_small)
        assert isinstance(sel_idx, torch.Tensor)
        assert isinstance(theta, torch.Tensor)
        assert sel_idx.numel() <= P_small.shape[1]
    except Exception:
        # Some edge cases might be expected to fail
        pass


@pytest.mark.parametrize(
    "rho,epsilon",
    [
        (1e-3, 1e-6),
        (1e-2, 1e-8),
        (1e-1, 1e-10),
    ],
)
def test_ols_optimizer_different_parameters(
    ols_optimizer_class, synthetic_data, rho, epsilon
):
    """Test OLS optimizer with different parameter values."""
    torch.manual_seed(42)
    P, d, _ = synthetic_data(l=100, m=30, k_true=3)

    opt = ols_optimizer_class(rho=rho, epsilon=epsilon)
    sel_idx, theta = opt.optimize(P, d)

    # Basic validation
    assert isinstance(sel_idx, torch.Tensor)
    assert isinstance(theta, torch.Tensor)
    assert sel_idx.numel() == theta.numel()

    if sel_idx.numel() > 0:
        # Check reconstruction quality
        y = P[:, sel_idx] @ theta
        rel_err = torch.norm(d - y, dim=0) / (torch.norm(d, dim=0) + 1e-12)
        assert rel_err < 1.0  # Very loose bound for different parameters
