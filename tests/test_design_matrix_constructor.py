"""
Test module for design matrix constructor functionality.
"""

import pytest
import importlib.util
from pathlib import Path

import numpy as np
import torch


@pytest.fixture
def design_matrix_module():
    """Load the design matrix constructor module."""
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "design_matrix_constructor.py"
    spec = importlib.util.spec_from_file_location(
        "design_matrix_constructor", str(mod_path)
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_types_and_shapes_torch_and_numpy(design_matrix_module):
    """Test that function handles both torch and numpy inputs correctly."""
    mod = design_matrix_module
    f = mod.construct_design_matrix_with_no_pretraining
    RBF = mod.RadialBasisFunction

    # Torch inputs
    l, n, m = 32, 4, 7
    torch.manual_seed(0)
    X = torch.randn(l, n)
    d = torch.randn(l)
    cand = torch.randperm(l)[:m]
    centres = X[cand]  # Convert indices to actual centres
    P = f(X, d, centres=centres, radial_basis_function=RBF.GAUSSIAN, sigma=1.0)
    assert isinstance(P, torch.Tensor)
    assert P.shape == (l, m * n)

    # NumPy inputs
    rng = np.random.default_rng(0)
    Xn = rng.standard_normal((l, n))  # float64
    dn = rng.standard_normal(l)
    candn = np.arange(m, dtype=np.int64)
    centres_n = Xn[candn]  # Convert indices to actual centres
    Pn = f(Xn, dn, centres=centres_n, radial_basis_function=RBF.GAUSSIAN, sigma=1.0)
    assert isinstance(Pn, np.ndarray)
    assert Pn.shape == (l, m * n)
    assert Pn.dtype in (np.float32, np.float64)


def test_gaussian_with_large_sigma_replicates_X_blocks(design_matrix_module):
    """Test that large sigma in Gaussian RBF replicates X blocks."""
    mod = design_matrix_module
    f = mod.construct_design_matrix_with_no_pretraining
    RBF = mod.RadialBasisFunction

    l, n, m = 20, 3, 5
    torch.manual_seed(1)
    X = torch.randn(l, n)
    d = torch.randn(l)
    cand = torch.arange(m)
    centres = X[cand]  # Convert indices to actual centres
    # Very large sigma -> phi ~ 1, so P should be [X | X | ... | X]
    P = f(X, d, centres=centres, radial_basis_function=RBF.GAUSSIAN, sigma=1e9)
    expected = torch.cat([X] * m, dim=1)
    assert torch.allclose(P, expected, rtol=1e-6, atol=1e-6)


def test_laplacian_differs_from_gaussian_for_finite_sigma(design_matrix_module):
    """Test that Laplacian RBF produces different results from Gaussian."""
    mod = design_matrix_module
    f = mod.construct_design_matrix_with_no_pretraining
    RBF = mod.RadialBasisFunction

    l, n, m = 25, 3, 6
    torch.manual_seed(2)
    X = torch.randn(l, n)
    d = torch.randn(l)
    cand = torch.randperm(l)[:m]
    centres = X[cand]  # Convert indices to actual centres
    sigma = 0.5
    P_g = f(X, d, centres=centres, radial_basis_function=RBF.GAUSSIAN, sigma=sigma)
    P_l = f(X, d, centres=centres, radial_basis_function=RBF.LAPLACIAN, sigma=sigma)
    assert not torch.allclose(P_g, P_l)


def test_sigma_heuristic_returns_finite_values(design_matrix_module):
    """Test that automatic sigma estimation returns finite values."""
    mod = design_matrix_module
    f = mod.construct_design_matrix_with_no_pretraining
    RBF = mod.RadialBasisFunction

    l, n, m = 16, 4, 8
    torch.manual_seed(3)
    X = torch.randn(l, n)
    d = torch.randn(l)
    cand = torch.randperm(l)[:m]
    centres = X[cand]  # Convert indices to actual centres
    P = f(X, d, centres=centres, radial_basis_function=RBF.GAUSSIAN, sigma=None)
    assert torch.isfinite(P).all()


def test_local_pretraining_types_and_shapes(design_matrix_module):
    """Test basic functionality and shapes for local pretraining."""
    mod = design_matrix_module
    f = mod.construct_design_matrix_with_local_pretraining
    RBF = mod.RadialBasisFunction

    l, n, m = 30, 5, 6
    torch.manual_seed(4)
    X = torch.randn(l, n)
    d = torch.randn(l)
    centres = X[torch.randperm(l)[:m]]
    P = f(X, d, centres=centres, radial_basis_function=RBF.GAUSSIAN, sigma=1.0)
    assert isinstance(P, torch.Tensor)
    assert P.shape == (l, m)


def test_local_pretraining_large_sigma_columns_equal_to_ridge_prediction(
    design_matrix_module,
):
    """Test that large sigma in local pretraining produces ridge-like predictions."""
    mod = design_matrix_module
    f = mod.construct_design_matrix_with_local_pretraining
    RBF = mod.RadialBasisFunction

    l, n, m = 40, 6, 8
    ridge = 1e-6
    torch.manual_seed(5)
    X = torch.randn(l, n)
    d = torch.randn(l)
    cand = torch.arange(m)
    centres = X[cand]  # Convert indices to actual centres
    # With very large sigma, all weights ~1 -> same local model for all centres
    P = f(
        X,
        d,
        centres=centres,
        radial_basis_function=RBF.GAUSSIAN,
        sigma=1e9,
        ridge=ridge,
    )
    # Closed-form ridge solution
    eye = torch.eye(n)
    a = X.T @ X + ridge * eye
    b = X.T @ d
    coef = torch.linalg.lstsq(a, b).solution
    z = X @ coef
    # Each column should equal z
    assert torch.allclose(P, z.unsqueeze(1).expand(-1, m), rtol=1e-5, atol=1e-5)


def test_local_pretraining_laplacian_differs_from_gaussian(design_matrix_module):
    """Test that Laplacian RBF differs from Gaussian in local pretraining."""
    mod = design_matrix_module
    f = mod.construct_design_matrix_with_local_pretraining
    RBF = mod.RadialBasisFunction

    l, n, m = 35, 4, 7
    torch.manual_seed(6)
    X = torch.randn(l, n)
    d = torch.randn(l)
    cand = torch.randperm(l)[:m]
    centres = X[cand]  # Convert indices to actual centres
    sigma = 0.7
    P_g = f(X, d, centres=centres, radial_basis_function=RBF.GAUSSIAN, sigma=sigma)
    P_l = f(X, d, centres=centres, radial_basis_function=RBF.LAPLACIAN, sigma=sigma)
    assert not torch.allclose(P_g, P_l)


def test_local_pretraining_numpy_inputs(design_matrix_module):
    """Test local pretraining function with numpy inputs."""
    mod = design_matrix_module
    f = mod.construct_design_matrix_with_local_pretraining
    RBF = mod.RadialBasisFunction

    l, n, m = 28, 3, 5
    rng = np.random.default_rng(7)
    Xn = rng.standard_normal((l, n))
    dn = rng.standard_normal(l)
    cand = np.arange(m)
    centres_n = Xn[cand]  # Convert indices to actual centres
    Pn = f(Xn, dn, centres=centres_n, radial_basis_function=RBF.GAUSSIAN, sigma=1.0)
    assert isinstance(Pn, np.ndarray)
    assert Pn.shape == (l, m)
