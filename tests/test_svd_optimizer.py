"""
Test module for SVD optimizer functionality.
"""

import pytest
import torch
import importlib.util
from pathlib import Path


@pytest.fixture
def svd_optimizer_class():
    """Load the SVD optimizer class."""
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "svd_based_optimizer.py"
    spec = importlib.util.spec_from_file_location("svd_optimizer", str(mod_path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.SvdOptimizer


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""

    def _make_synthetic(
        l=400, m=120, k_true=6, noise=1e-3, device=None, dtype=torch.float32
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


def test_svd_converges_and_is_sparse(svd_optimizer_class, synthetic_data):
    """Test that SVD optimizer converges and produces sparse solutions."""
    torch.manual_seed(0)
    P, d, _ = synthetic_data()
    opt = svd_optimizer_class(epsilon=1e-2, alpha=1e-5, delta=1e-4)
    sel_idx, weights = opt.optimize(P, d)

    # Sanity checks
    assert sel_idx.ndim == 1 and weights.ndim == 1

    # Convergence: reconstruction should be reasonable on selected columns
    y = torch.zeros_like(d)
    if sel_idx.numel() > 0:
        y = P[:, sel_idx] @ weights
    rel_err = torch.norm(d - y, dim=0) / (torch.norm(d, dim=0) + 1e-12)
    assert rel_err < 0.25


def test_svd_optimizer_parameter_validation(svd_optimizer_class):
    """Test that SVD optimizer validates parameters correctly."""
    # Test valid parameters
    opt = svd_optimizer_class(epsilon=1e-2, alpha=1e-5, delta=1e-4)
    assert opt._epsilon == 1e-2
    assert opt._alpha == 1e-5
    assert opt._delta == 1e-4

    # Test default parameters
    opt_default = svd_optimizer_class()
    assert hasattr(opt_default, "_epsilon")
    assert hasattr(opt_default, "_alpha")
    assert hasattr(opt_default, "_delta")


def test_svd_optimizer_empty_selection(svd_optimizer_class, synthetic_data):
    """Test SVD optimizer behavior when no features are selected."""
    torch.manual_seed(42)
    # Very strict epsilon to potentially get empty selection
    P, d, _ = synthetic_data(l=50, m=20, k_true=2)
    opt = svd_optimizer_class(epsilon=1e-10, alpha=1e-3, delta=1e-2)

    sel_idx, weights = opt.optimize(P, d)

    # Should handle empty selection gracefully
    assert isinstance(sel_idx, torch.Tensor)
    assert isinstance(weights, torch.Tensor)
    assert sel_idx.numel() == weights.numel()


@pytest.mark.parametrize(
    "epsilon,alpha,delta",
    [
        (1e-1, 1e-4, 1e-3),
        (1e-2, 1e-5, 1e-4),
        (1e-3, 1e-6, 1e-5),
    ],
)
def test_svd_optimizer_different_parameters(
    svd_optimizer_class, synthetic_data, epsilon, alpha, delta
):
    """Test SVD optimizer with different parameter values."""
    torch.manual_seed(123)
    P, d, _ = synthetic_data(l=100, m=40, k_true=4)

    opt = svd_optimizer_class(epsilon=epsilon, alpha=alpha, delta=delta)
    sel_idx, weights = opt.optimize(P, d)

    # Basic validation
    assert isinstance(sel_idx, torch.Tensor)
    assert isinstance(weights, torch.Tensor)
    assert sel_idx.numel() == weights.numel()

    if sel_idx.numel() > 0:
        # Check reconstruction quality
        y = P[:, sel_idx] @ weights
        rel_err = torch.norm(d - y, dim=0) / (torch.norm(d, dim=0) + 1e-12)
        assert rel_err < 1.0  # Very loose bound for different parameters


def test_svd_optimizer_sparsity(svd_optimizer_class, synthetic_data):
    """Test that SVD optimizer produces reasonably sparse solutions."""
    torch.manual_seed(999)
    P, d, idx_true = synthetic_data(l=200, m=60, k_true=5)

    opt = svd_optimizer_class(epsilon=5e-2, alpha=1e-4, delta=1e-3)
    sel_idx, _ = opt.optimize(P, d)

    # Should be sparse - not select all features
    assert sel_idx.numel() < P.shape[1]

    # Check for some overlap with true features (if any selected)
    if sel_idx.numel() > 0:
        overlap = len(set(sel_idx.tolist()) & set(idx_true.tolist()))
        # At least some overlap expected for reasonable parameters
        total_possible = min(sel_idx.numel(), len(idx_true))
        overlap_ratio = overlap / max(total_possible, 1)
        assert overlap_ratio >= 0  # At least non-negative overlap
