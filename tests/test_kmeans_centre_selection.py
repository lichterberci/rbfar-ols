"""
Test module for K-means centre selection functionality.
"""

import pytest
import numpy as np
import torch

from design_matrix_constructor import select_centres_kmeans


class TestKMeansCentreSelection:
    """Test K-means centre selection functionality."""

    def test_kmeans_basic_functionality(self):
        """Test basic K-means centre selection."""
        torch.manual_seed(42)
        X = torch.randn(100, 3)
        m = 10

        centres = select_centres_kmeans(X, m, max_iters=50)

        assert isinstance(centres, torch.Tensor)
        assert centres.shape == (m, 3)
        assert torch.isfinite(centres).all()

    def test_kmeans_numpy_input(self):
        """Test K-means with NumPy input."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        m = 8

        centres = select_centres_kmeans(X, m, max_iters=30)

        assert isinstance(centres, np.ndarray)
        assert centres.shape == (m, 2)
        assert np.isfinite(centres).all()

    def test_kmeans_single_centre(self):
        """Test K-means with single centre."""
        torch.manual_seed(42)
        X = torch.randn(20, 2)

        centres = select_centres_kmeans(X, m=1)

        assert centres.shape == (1, 2)
        # Single centre should be close to the mean
        expected_centre = X.mean(dim=0, keepdim=True)
        assert torch.allclose(centres, expected_centre, rtol=0.1, atol=0.1)

    def test_kmeans_more_centres_than_points_raises_error(self):
        """Test that requesting more centres than points raises an error."""
        X = torch.randn(5, 2)

        with pytest.raises(ValueError, match="Cannot select 10 centres from 5 points"):
            select_centres_kmeans(X, m=10)

    def test_kmeans_convergence(self):
        """Test that K-means converges with tight tolerance."""
        torch.manual_seed(42)
        # Create clustered data
        cluster1 = torch.randn(20, 2) + torch.tensor([2.0, 2.0])
        cluster2 = torch.randn(20, 2) + torch.tensor([-2.0, -2.0])
        X = torch.cat([cluster1, cluster2], dim=0)

        centres = select_centres_kmeans(X, m=2, max_iters=100, tol=1e-6)

        # Should find centres near the true cluster centres
        assert centres.shape == (2, 2)
        true_centres = torch.tensor([[2.0, 2.0], [-2.0, -2.0]])

        # Match centres to closest true centres
        dists = torch.cdist(centres, true_centres)
        min_dists = dists.min(dim=1)[0]
        assert (min_dists < 1.0).all()  # Should be reasonably close

    def test_kmeans_deterministic_with_seed(self):
        """Test that K-means produces consistent results with the same seed."""
        X = torch.randn(50, 3)
        m = 5

        # Run twice with same seed
        torch.manual_seed(123)
        centres1 = select_centres_kmeans(X, m, max_iters=50)

        torch.manual_seed(123)
        centres2 = select_centres_kmeans(X, m, max_iters=50)

        # Should produce identical results
        assert torch.allclose(centres1, centres2)

    def test_kmeans_different_tolerances(self):
        """Test K-means behavior with different convergence tolerances."""
        torch.manual_seed(42)
        X = torch.randn(30, 2)
        m = 4

        # Very loose tolerance
        centres_loose = select_centres_kmeans(X, m, max_iters=100, tol=1e-1)

        # Very tight tolerance
        centres_tight = select_centres_kmeans(X, m, max_iters=100, tol=1e-8)

        # Both should produce valid results
        assert centres_loose.shape == centres_tight.shape == (m, 2)
        assert torch.isfinite(centres_loose).all()
        assert torch.isfinite(centres_tight).all()

    def test_kmeans_max_iterations_limit(self):
        """Test that K-means respects the maximum iterations limit."""
        torch.manual_seed(42)
        X = torch.randn(100, 2)
        m = 10

        # Very few iterations
        centres = select_centres_kmeans(X, m, max_iters=1, tol=1e-10)

        # Should still produce a valid result even with limited iterations
        assert centres.shape == (m, 2)
        assert torch.isfinite(centres).all()


if __name__ == "__main__":
    pytest.main([__file__])
