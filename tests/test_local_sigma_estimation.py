"""
Test module for local sigma estimation functionality and integration tests.
"""

import pytest
import numpy as np
import torch

from design_matrix_constructor import (
    estimate_local_sigma_knn,
    construct_design_matrix_with_local_pretraining,
    RadialBasisFunction,
)


class TestLocalSigmaEstimation:
    """Test local sigma estimation functionality."""

    def test_local_sigma_basic_functionality(self):
        """Test basic local sigma estimation."""
        torch.manual_seed(42)
        centres = torch.randn(10, 3)
        k = 3

        local_sigmas = estimate_local_sigma_knn(centres, k)

        assert isinstance(local_sigmas, torch.Tensor)
        assert local_sigmas.shape == (10,)
        assert torch.isfinite(local_sigmas).all()
        assert (local_sigmas > 0).all()

    def test_local_sigma_numpy_input(self):
        """Test local sigma with NumPy input."""
        np.random.seed(42)
        centres = np.random.randn(8, 2)
        k = 2

        local_sigmas = estimate_local_sigma_knn(centres, k)

        assert isinstance(local_sigmas, np.ndarray)
        assert local_sigmas.shape == (8,)
        assert np.isfinite(local_sigmas).all()
        assert (local_sigmas > 0).all()

    def test_local_sigma_single_centre(self):
        """Test local sigma with single centre."""
        centres = torch.tensor([[1.0, 2.0]])

        local_sigmas = estimate_local_sigma_knn(centres, k=1)

        assert local_sigmas.shape == (1,)
        # With single centre, should get a small but positive sigma
        assert local_sigmas[0] > 0
        assert torch.isfinite(local_sigmas).all()

    def test_local_sigma_formula_validation(self):
        """Test that local sigma follows the expected formula."""
        # Create centres with known distances
        centres = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
        k = 2

        local_sigmas = estimate_local_sigma_knn(centres, k)

        # For centre [0,0], nearest neighbors are [1,0] and [0,1] with distance 1.0 each
        # σ = (1/√2) * (1/k) * Σ||distances|| = (1/√2) * (1/2) * (1 + 1) = 1/√2 ≈ 0.707
        expected_sigma_0 = (1.0 / np.sqrt(2.0)) * (1.0 / 2.0) * 2.0
        assert torch.allclose(
            local_sigmas[0],
            torch.tensor(expected_sigma_0, dtype=local_sigmas.dtype),
            rtol=1e-4,
        )

    def test_local_sigma_k_larger_than_centres(self):
        """Test local sigma when k is larger than available centres."""
        centres = torch.randn(3, 2)
        k = 5  # More than 3-1 = 2 available neighbors

        # Should not raise error, should use available neighbours
        local_sigmas = estimate_local_sigma_knn(centres, k)

        assert local_sigmas.shape == (3,)
        assert (local_sigmas > 0).all()

    def test_local_sigma_zero_k_raises_error(self):
        """Test that k=0 raises an appropriate error."""
        centres = torch.randn(5, 2)

        with pytest.raises(ValueError, match="k must be positive"):
            estimate_local_sigma_knn(centres, k=0)

    def test_local_sigma_negative_k_raises_error(self):
        """Test that negative k raises an appropriate error."""
        centres = torch.randn(5, 2)

        with pytest.raises(ValueError, match="k must be positive"):
            estimate_local_sigma_knn(centres, k=-1)

    def test_local_sigma_different_dimensions(self):
        """Test local sigma estimation with different input dimensions."""
        torch.manual_seed(42)

        # Test with 1D centres
        centres_1d = torch.randn(6, 1)
        sigmas_1d = estimate_local_sigma_knn(centres_1d, k=2)
        assert sigmas_1d.shape == (6,)
        assert (sigmas_1d > 0).all()

        # Test with high-dimensional centres
        centres_5d = torch.randn(8, 5)
        sigmas_5d = estimate_local_sigma_knn(centres_5d, k=3)
        assert sigmas_5d.shape == (8,)
        assert (sigmas_5d > 0).all()

    def test_local_sigma_consistency_across_dtypes(self):
        """Test that local sigma estimation is consistent across different data types."""
        centres_f32 = torch.randn(5, 2, dtype=torch.float32)
        centres_f64 = centres_f32.double()

        sigmas_f32 = estimate_local_sigma_knn(centres_f32, k=2)
        sigmas_f64 = estimate_local_sigma_knn(centres_f64, k=2)

        # Results should be very close despite different dtypes
        assert torch.allclose(sigmas_f32, sigmas_f64.float(), rtol=1e-6)


class TestIntegrationWithDesignMatrix:
    """Test integration of local sigma estimation with design matrix construction."""

    def test_local_sigma_with_design_matrix(self):
        """Test local sigma estimation integrated with design matrix construction."""
        torch.manual_seed(42)
        l, n, m = 20, 3, 5
        X = torch.randn(l, n)
        d = torch.randn(l)
        centres = torch.randn(m, n)

        # Test with local sigma
        P_local = construct_design_matrix_with_local_pretraining(
            X,
            d,
            centres,
            use_local_sigma=True,
            local_sigma_k=3,
            radial_basis_function=RadialBasisFunction.GAUSSIAN,
        )

        # Test with global sigma
        P_global = construct_design_matrix_with_local_pretraining(
            X,
            d,
            centres,
            use_local_sigma=False,
            sigma=1.0,
            radial_basis_function=RadialBasisFunction.GAUSSIAN,
        )

        # Both should have the same shape
        assert P_local.shape == P_global.shape == (l, m + 1)

        # But should produce different results
        assert not torch.allclose(P_local, P_global, rtol=1e-4)

    def test_new_features_dont_break_existing_functionality(self):
        """Test that new features don't break existing functionality."""
        torch.manual_seed(42)
        l, n, m = 15, 2, 4
        X = torch.randn(l, n)
        d = torch.randn(l)
        centres = torch.randn(m, n)

        # Original call (should still work)
        P_original = construct_design_matrix_with_local_pretraining(
            X,
            d,
            centres,
            radial_basis_function=RadialBasisFunction.GAUSSIAN,
            sigma=0.5,
        )

        # Verify it has the expected shape and properties
        assert P_original.shape == (l, m + 1)
        assert torch.isfinite(P_original).all()

        # Test that it produces reasonable values (not all zeros)
        assert not (P_original == 0).all()

        # Test that both old and new parameter sets work and produce finite results
        P_new = construct_design_matrix_with_local_pretraining(
            X,
            d,
            centres,
            radial_basis_function=RadialBasisFunction.GAUSSIAN,
            sigma=0.5,
            use_local_sigma=False,
            local_sigma_k=5,
        )

        assert P_new.shape == (l, m + 1)
        assert torch.isfinite(P_new).all()

    def test_laplacian_rbf_with_local_sigma(self):
        """Test local sigma with Laplacian RBF."""
        torch.manual_seed(42)
        l, n, m = 12, 2, 3
        X = torch.randn(l, n)
        d = torch.randn(l)
        centres = torch.randn(m, n)

        P = construct_design_matrix_with_local_pretraining(
            X,
            d,
            centres,
            use_local_sigma=True,
            local_sigma_k=2,
            radial_basis_function=RadialBasisFunction.LAPLACIAN,
        )

        assert P.shape == (l, m + 1)
        assert torch.isfinite(P).all()
        assert not (P == 0).all()  # Should not be all zeros

    def test_local_sigma_different_k_values(self):
        """Test local sigma with different k values in design matrix construction."""
        torch.manual_seed(42)
        l, n, m = 15, 2, 6
        X = torch.randn(l, n)
        d = torch.randn(l)
        centres = torch.randn(m, n)

        # Test with different k values
        P_k2 = construct_design_matrix_with_local_pretraining(
            X,
            d,
            centres,
            use_local_sigma=True,
            local_sigma_k=2,
            radial_basis_function=RadialBasisFunction.GAUSSIAN,
        )

        P_k4 = construct_design_matrix_with_local_pretraining(
            X,
            d,
            centres,
            use_local_sigma=True,
            local_sigma_k=4,
            radial_basis_function=RadialBasisFunction.GAUSSIAN,
        )

        # Should have same shape but different values
        assert P_k2.shape == P_k4.shape == (l, m + 1)
        assert not torch.allclose(P_k2, P_k4, rtol=1e-4)

    def test_local_sigma_vs_provided_sigma(self):
        """Test that local sigma overrides provided sigma when enabled."""
        torch.manual_seed(42)
        l, n, m = 10, 2, 4
        X = torch.randn(l, n)
        d = torch.randn(l)
        centres = torch.randn(m, n)

        # When use_local_sigma=True, provided sigma should be ignored
        P_local = construct_design_matrix_with_local_pretraining(
            X,
            d,
            centres,
            use_local_sigma=True,
            local_sigma_k=2,
            sigma=999.0,  # This large value should be ignored
            radial_basis_function=RadialBasisFunction.GAUSSIAN,
        )

        # When use_local_sigma=False, provided sigma should be used
        P_global = construct_design_matrix_with_local_pretraining(
            X,
            d,
            centres,
            use_local_sigma=False,
            sigma=999.0,
            radial_basis_function=RadialBasisFunction.GAUSSIAN,
        )

        # Results should be different (local sigma vs very large global sigma)
        assert P_local.shape == P_global.shape == (l, m + 1)
        assert not torch.allclose(P_local, P_global, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
