"""
Test module for cross-validation embedding estimation functionality.
"""

import pytest
import numpy as np
import importlib.util
from pathlib import Path


@pytest.fixture
def cv_embedding_module():
    """Load the cross-validation embedding estimation module."""
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "cv_embedding_estimation.py"
    spec = importlib.util.spec_from_file_location(
        "cv_embedding_estimation", str(mod_path)
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def estimator_class(cv_embedding_module):
    """Get the CrossValidationEmbeddingEstimator class."""
    return cv_embedding_module.CrossValidationEmbeddingEstimator


@pytest.fixture
def synthetic_time_series():
    """Generate synthetic time series for testing."""

    def _make_time_series(n=200, noise_level=0.1, series_type="lorenz"):
        """Create different types of synthetic time series."""
        t = np.linspace(0, 20, n)

        if series_type == "lorenz":
            # Simple Lorenz-like chaotic series (simplified)
            x = np.zeros(n)
            x[0] = 1.0
            dt = t[1] - t[0]
            for i in range(1, n):
                # Simplified Lorenz equation for x component
                dx = 10 * (np.sin(t[i - 1]) - x[i - 1])
                x[i] = x[i - 1] + dx * dt
            x += noise_level * np.random.randn(n)

        elif series_type == "ar":
            # AR(2) process
            x = np.zeros(n)
            x[0] = np.random.randn()
            x[1] = np.random.randn()
            for i in range(2, n):
                x[i] = 0.7 * x[i - 1] - 0.3 * x[i - 2] + noise_level * np.random.randn()

        elif series_type == "sine":
            # Sine wave with noise
            x = np.sin(2 * np.pi * t / 4) + 0.5 * np.sin(2 * np.pi * t / 7)
            x += noise_level * np.random.randn(n)

        else:
            # Random walk
            x = np.cumsum(np.random.randn(n) * noise_level)

        return x

    return _make_time_series


class TestCrossValidationEmbeddingEstimator:
    """Test suite for CrossValidationEmbeddingEstimator."""

    def test_initialization_default_params(self, estimator_class):
        """Test estimator initialization with default parameters."""
        estimator = estimator_class()

        assert estimator.tau_range == (1, 10)
        assert estimator.dim_range == (2, 10)
        assert estimator.n_splits == 5
        assert estimator.prediction_horizon == 1
        assert estimator.scoring_metric == "mse"
        assert estimator.verbose is False
        assert estimator.random_state is None

    def test_initialization_custom_params(self, estimator_class):
        """Test estimator initialization with custom parameters."""
        estimator = estimator_class(
            tau_range=(1, 5),
            dim_range=(2, 6),
            n_splits=3,
            prediction_horizon=2,
            scoring_metric="mae",
            verbose=True,
            random_state=42,
        )

        assert estimator.tau_range == (1, 5)
        assert estimator.dim_range == (2, 6)
        assert estimator.n_splits == 3
        assert estimator.prediction_horizon == 2
        assert estimator.scoring_metric == "mae"
        assert estimator.verbose is True
        assert estimator.random_state == 42

    def test_invalid_parameter_validation(self, estimator_class):
        """Test parameter validation during initialization."""
        # Invalid tau_range
        with pytest.raises(ValueError, match="tau_range must satisfy"):
            estimator_class(tau_range=(0, 5))

        with pytest.raises(ValueError, match="tau_range must satisfy"):
            estimator_class(tau_range=(5, 3))

        # Invalid dim_range
        with pytest.raises(ValueError, match="dim_range must satisfy"):
            estimator_class(dim_range=(0, 5))

        with pytest.raises(ValueError, match="dim_range must satisfy"):
            estimator_class(dim_range=(5, 3))

        # Invalid n_splits
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            estimator_class(n_splits=1)

        # Invalid prediction_horizon
        with pytest.raises(ValueError, match="prediction_horizon must be at least 1"):
            estimator_class(prediction_horizon=0)

        # Invalid scoring_metric
        with pytest.raises(ValueError, match="scoring_metric must be one of"):
            estimator_class(scoring_metric="invalid_metric")

    def test_create_embedding(self, estimator_class):
        """Test time-delay embedding creation."""
        estimator = estimator_class()
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Test basic embedding
        embedding = estimator._create_embedding(x, tau=1, dim=3)
        expected = np.array(
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
                [6, 7, 8],
                [7, 8, 9],
                [8, 9, 10],
            ]
        )
        np.testing.assert_array_equal(embedding, expected)

        # Test with different tau
        embedding2 = estimator._create_embedding(x, tau=2, dim=3)
        expected2 = np.array(
            [[1, 3, 5], [2, 4, 6], [3, 5, 7], [4, 6, 8], [5, 7, 9], [6, 8, 10]]
        )
        np.testing.assert_array_equal(embedding2, expected2)

    def test_create_embedding_insufficient_data(self, estimator_class):
        """Test embedding creation with insufficient data."""
        estimator = estimator_class()
        x = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="Not enough data points for embedding"):
            estimator._create_embedding(x, tau=2, dim=3)

    def test_prepare_prediction_data(self, estimator_class):
        """Test preparation of prediction data."""
        estimator = estimator_class()
        embedding = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])

        X, y = estimator._prepare_prediction_data(embedding)

        expected_X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
        expected_y = np.array([2, 3, 4, 5])

        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(y, expected_y)

    def test_prepare_prediction_data_insufficient(self, estimator_class):
        """Test prediction data preparation with insufficient data."""
        estimator = estimator_class()
        embedding = np.array([[1, 2, 3]])  # Only one row

        with pytest.raises(
            ValueError, match="Embedding too short for prediction horizon"
        ):
            estimator._prepare_prediction_data(embedding)

    def test_scoring_metrics(self, estimator_class):
        """Test different scoring metrics."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        # Test MSE
        estimator_mse = estimator_class(scoring_metric="mse")
        mse_score = estimator_mse._score_prediction(y_true, y_pred)
        expected_mse = np.mean((y_true - y_pred) ** 2)
        assert abs(mse_score - expected_mse) < 1e-10

        # Test MAE
        estimator_mae = estimator_class(scoring_metric="mae")
        mae_score = estimator_mae._score_prediction(y_true, y_pred)
        expected_mae = np.mean(np.abs(y_true - y_pred))
        assert abs(mae_score - expected_mae) < 1e-10

        # Test R²
        estimator_r2 = estimator_class(scoring_metric="r2")
        r2_score = estimator_r2._score_prediction(y_true, y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        expected_r2 = 1 - (ss_res / ss_tot)
        assert abs(r2_score - expected_r2) < 1e-10

    def test_estimate_with_valid_input(self, estimator_class, synthetic_time_series):
        """Test estimation with valid synthetic time series."""
        # Use a smaller parameter space for faster testing
        estimator = estimator_class(
            tau_range=(1, 3), dim_range=(2, 4), n_splits=3, verbose=False
        )

        # Test with AR process (should work well)
        x = synthetic_time_series(n=100, series_type="ar", noise_level=0.1)
        tau, dim = estimator.estimate(x)

        # Check that returned values are within expected ranges
        assert 1 <= tau <= 3
        assert 2 <= dim <= 4
        assert isinstance(tau, int)
        assert isinstance(dim, int)

    def test_estimate_with_different_series_types(
        self, estimator_class, synthetic_time_series
    ):
        """Test estimation with different types of time series."""
        estimator = estimator_class(
            tau_range=(1, 3), dim_range=(2, 4), n_splits=3, verbose=False
        )

        series_types = ["ar", "sine", "lorenz"]

        for series_type in series_types:
            x = synthetic_time_series(n=100, series_type=series_type, noise_level=0.05)
            tau, dim = estimator.estimate(x)

            # Basic sanity checks
            assert 1 <= tau <= 3
            assert 2 <= dim <= 4

    def test_estimate_invalid_input(self, estimator_class):
        """Test estimation with invalid inputs."""
        estimator = estimator_class()

        # Test with None input
        with pytest.raises(ValueError, match="Input array must not be None"):
            estimator.estimate(None)

        # Test with multidimensional array
        with pytest.raises(ValueError, match="Input must be a 1D array"):
            estimator.estimate(np.array([[1, 2], [3, 4]]))

        # Test with too short time series
        with pytest.raises(ValueError, match="Time series too short"):
            estimator.estimate(np.array([1, 2, 3]))

    def test_evaluate_parameters_edge_cases(
        self, estimator_class, synthetic_time_series
    ):
        """Test parameter evaluation with edge cases."""
        estimator = estimator_class(n_splits=3)
        x = synthetic_time_series(n=50, series_type="ar")

        # Test with parameters that require too much data
        score = estimator._evaluate_parameters(x, tau=10, dim=10)
        assert score == float("inf")  # Should return worst score

        # Test with valid parameters
        score = estimator._evaluate_parameters(x, tau=1, dim=2)
        assert isinstance(score, float)
        assert score != float("inf")

    def test_different_scoring_metrics_estimation(
        self, estimator_class, synthetic_time_series
    ):
        """Test estimation with different scoring metrics."""
        x = synthetic_time_series(n=80, series_type="ar", noise_level=0.1)

        metrics = ["mse", "mae", "r2"]
        results = {}

        for metric in metrics:
            estimator = estimator_class(
                tau_range=(1, 3),
                dim_range=(2, 4),
                n_splits=3,
                scoring_metric=metric,
                verbose=False,
            )
            tau, dim = estimator.estimate(x)
            results[metric] = (tau, dim)

            # Verify valid ranges
            assert 1 <= tau <= 3
            assert 2 <= dim <= 4

        # Results might be different but should all be valid
        assert len(results) == 3


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_estimate_embedding_cv(self, cv_embedding_module, synthetic_time_series):
        """Test the convenience function estimate_embedding_cv."""
        x = synthetic_time_series(n=80, series_type="ar")
        estimator = cv_embedding_module.CrossValidationEmbeddingEstimator(
            tau_range=(1, 3), dim_range=(2, 4), n_splits=3, verbose=False
        )

        tau, dim = cv_embedding_module.estimate_embedding_cv(x, estimator)

        assert 1 <= tau <= 3
        assert 2 <= dim <= 4
        assert isinstance(tau, int)
        assert isinstance(dim, int)


class TestIntegrationWithRealData:
    """Integration tests with real-world-like data patterns."""

    def test_with_lynx_like_data(self, estimator_class):
        """Test with data similar to the lynx dataset (chaotic time series)."""
        # Generate a more complex chaotic-like series
        np.random.seed(42)  # For reproducibility
        n = 100
        x = np.zeros(n)
        x[0] = 1.0

        # Simple chaotic map
        for i in range(1, n):
            x[i] = 3.8 * x[i - 1] * (1 - x[i - 1]) + 0.01 * np.random.randn()

        estimator = estimator_class(
            tau_range=(1, 4), dim_range=(2, 5), n_splits=3, verbose=False
        )

        tau, dim = estimator.estimate(x)

        # Should find reasonable parameters
        assert 1 <= tau <= 4
        assert 2 <= dim <= 5

    def test_performance_comparison_hint(self, estimator_class, synthetic_time_series):
        """Test that gives insight into performance vs traditional methods."""
        # This test demonstrates the approach but doesn't validate against
        # traditional methods since they're not implemented here

        x = synthetic_time_series(n=120, series_type="ar", noise_level=0.05)

        # CV-based estimation
        cv_estimator = estimator_class(
            tau_range=(1, 5),
            dim_range=(2, 6),
            n_splits=3,
            scoring_metric="mse",
            verbose=False,
        )

        tau_cv, dim_cv = cv_estimator.estimate(x)

        # For AR(2) process, we might expect:
        # - tau around 1 (since AR processes have immediate dependencies)
        # - dim around 2-3 (since it's an AR(2) process)

        assert 1 <= tau_cv <= 5
        assert 2 <= dim_cv <= 6

        # The exact values depend on the noise and specific realization,
        # but this demonstrates the method works
