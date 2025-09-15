"""
Modified multifold cross-validation criterion for embedding parameter estimation.

This module implements a performance-based approach to estimate optimal embedding
parameters (delay τ and dimension m) by directly optimizing prediction performance
using cross-validation rather than relying on geometric or information-theoretic
properties of the reconstructed phase space.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, List


class EmbeddingEstimator(ABC):
    """Abstract base class for embedding parameter estimation."""

    @abstractmethod
    def estimate(self, x: np.ndarray) -> Tuple[int, int]:
        """Estimate optimal embedding parameters.

        Args:
            x: Time series data as 1D numpy array

        Returns:
            Tuple of (tau, dimension) where tau is the delay and dimension is the embedding dimension
        """
        pass


def _create_time_series_splits(
    n_samples: int, n_splits: int, min_train_size: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create time series cross-validation splits.

    Args:
        n_samples: Total number of samples
        n_splits: Number of CV splits
        min_train_size: Minimum training set size

    Returns:
        List of (train_indices, validation_indices) tuples
    """
    if n_samples < min_train_size + n_splits:
        return []

    splits = []
    # Calculate step size to create roughly equal validation sets
    remaining_samples = n_samples - min_train_size
    val_size = max(1, remaining_samples // n_splits)

    for i in range(n_splits):
        train_end = min_train_size + i * val_size
        val_start = train_end
        val_end = min(val_start + val_size, n_samples)

        # Skip if validation set would be empty
        if val_start >= val_end:
            break

        train_indices = np.arange(0, train_end)
        val_indices = np.arange(val_start, val_end)
        splits.append((train_indices, val_indices))

    return splits


class _SimpleLinearRegression:
    """Simple linear regression implementation using least squares."""

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit linear regression model."""
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

        try:
            # Solve normal equation: θ = (X^T X)^(-1) X^T y
            XtX = X_with_bias.T @ X_with_bias
            Xty = X_with_bias.T @ y

            # Add small regularization for numerical stability
            XtX += 1e-8 * np.eye(XtX.shape[0])

            params = np.linalg.solve(XtX, Xty)
            self.intercept_ = params[0]
            self.coef_ = params[1:] if len(params) > 1 else np.array([params[1]])

        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            params = np.linalg.pinv(X_with_bias) @ y
            self.intercept_ = params[0]
            self.coef_ = params[1:] if len(params) > 1 else np.array([params[1]])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model not fitted yet")
        return X @ self.coef_ + self.intercept_


@dataclass
class CrossValidationEmbeddingEstimator(EmbeddingEstimator):
    """
    Modified multifold cross-validation criterion for embedding parameter estimation.

    This estimator optimizes embedding parameters by directly minimizing prediction error
    across multiple validation folds, providing a task-specific approach that considers
    the entire pipeline: embedding → model → prediction.
    """

    tau_range: Tuple[int, int] = (1, 10)
    dim_range: Tuple[int, int] = (2, 10)
    n_splits: int = 5
    prediction_horizon: int = 1
    min_train_size: Optional[int] = None
    scoring_metric: str = "mse"  # "mse", "mae", "r2"
    verbose: bool = False
    random_state: Optional[int] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.tau_range[0] < 1 or self.tau_range[0] >= self.tau_range[1]:
            raise ValueError("tau_range must satisfy 1 <= tau_min < tau_max")
        if self.dim_range[0] < 1 or self.dim_range[0] >= self.dim_range[1]:
            raise ValueError("dim_range must satisfy 1 <= dim_min < dim_max")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1")
        if self.scoring_metric not in ["mse", "mae", "r2"]:
            raise ValueError("scoring_metric must be one of: 'mse', 'mae', 'r2'")

    def _create_embedding(self, x: np.ndarray, tau: int, dim: int) -> np.ndarray:
        """Create time-delay embedding of the time series.

        Args:
            x: 1D time series
            tau: Time delay
            dim: Embedding dimension

        Returns:
            Embedded matrix of shape (n_vectors, dim)
        """
        n = len(x)
        n_vectors = n - (dim - 1) * tau

        if n_vectors <= 0:
            raise ValueError(
                f"Not enough data points for embedding with tau={tau}, dim={dim}"
            )

        embedding = np.zeros((n_vectors, dim))
        for i in range(dim):
            start_idx = i * tau
            end_idx = start_idx + n_vectors
            embedding[:, i] = x[start_idx:end_idx]

        return embedding

    def _prepare_prediction_data(
        self, embedding: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for prediction.

        Args:
            embedding: Time-delay embedding matrix

        Returns:
            Tuple of (features, targets) for supervised learning
        """
        if len(embedding) <= self.prediction_horizon:
            raise ValueError("Embedding too short for prediction horizon")

        # Features are the embedding vectors, targets are future values of the first component
        X = embedding[: -self.prediction_horizon]
        y = embedding[self.prediction_horizon :, 0]  # Predict first component

        return X, y

    def _score_prediction(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate prediction score based on the chosen metric.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Score (lower is better for mse/mae, higher is better for r2)
        """
        if self.scoring_metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif self.scoring_metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        elif self.scoring_metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (
                ss_res / (ss_tot + 1e-10)
            )  # Add small epsilon to avoid division by zero
        else:
            raise ValueError(f"Unknown scoring metric: {self.scoring_metric}")

    def _evaluate_parameters(self, x: np.ndarray, tau: int, dim: int) -> float:
        """Evaluate a specific (tau, dim) combination using cross-validation.

        Args:
            x: Time series data
            tau: Time delay parameter
            dim: Embedding dimension

        Returns:
            Average cross-validation score
        """
        try:
            # Create embedding
            embedding = self._create_embedding(x, tau, dim)
            X, y = self._prepare_prediction_data(embedding)

            # Ensure we have enough data for cross-validation
            if len(X) < self.n_splits * 2:
                return float("inf") if self.scoring_metric != "r2" else float("-inf")

            # Set up time series cross-validation
            min_train_size = self.min_train_size or max(
                10, len(X) // (self.n_splits + 1)
            )
            splits = _create_time_series_splits(len(X), self.n_splits, min_train_size)

            if not splits:
                return float("inf") if self.scoring_metric != "r2" else float("-inf")

            scores = []
            for train_idx, val_idx in splits:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Fit simple linear model
                model = _SimpleLinearRegression()
                model.fit(X_train, y_train)

                # Predict and score
                y_pred = model.predict(X_val)
                score = self._score_prediction(y_val, y_pred)
                scores.append(score)

            if not scores:
                return float("inf") if self.scoring_metric != "r2" else float("-inf")

            return np.mean(scores)

        except (ValueError, np.linalg.LinAlgError):
            # Return worst possible score for invalid parameter combinations
            return float("inf") if self.scoring_metric != "r2" else float("-inf")

    def estimate(self, x: np.ndarray) -> Tuple[int, int]:
        """Estimate optimal embedding parameters using cross-validation.

        Args:
            x: Time series data as 1D numpy array

        Returns:
            Tuple of (tau, dimension) with optimal embedding parameters
        """
        if x is None:
            raise ValueError("Input array must not be None")

        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("Input must be a 1D array")
        if len(x) < 10:
            raise ValueError("Time series too short for meaningful cross-validation")

        # Generate parameter combinations to test
        tau_values = range(self.tau_range[0], self.tau_range[1] + 1)
        dim_values = range(self.dim_range[0], self.dim_range[1] + 1)

        best_score = float("inf") if self.scoring_metric != "r2" else float("-inf")
        best_tau = self.tau_range[0]
        best_dim = self.dim_range[0]

        # Grid search with optional progress reporting
        total_combinations = len(tau_values) * len(dim_values)
        completed = 0

        for tau in tau_values:
            for dim in dim_values:
                # Check if we have enough data for this combination
                min_length_needed = (
                    (dim - 1) * tau + self.prediction_horizon + self.n_splits * 2
                )
                if len(x) < min_length_needed:
                    completed += 1
                    continue

                score = self._evaluate_parameters(x, tau, dim)

                # Update best parameters based on scoring metric
                is_better = (self.scoring_metric != "r2" and score < best_score) or (
                    self.scoring_metric == "r2" and score > best_score
                )

                if is_better:
                    best_score = score
                    best_tau = tau
                    best_dim = dim

                completed += 1
                if self.verbose and completed % max(1, total_combinations // 10) == 0:
                    print(
                        f"Cross-validation progress: {completed}/{total_combinations} combinations tested"
                    )

        if self.verbose:
            print(
                f"Best parameters: tau={best_tau}, dim={best_dim}, score={best_score:.6f}"
            )

        return best_tau, best_dim


def estimate_embedding_cv(
    x: np.ndarray, estimator: CrossValidationEmbeddingEstimator
) -> Tuple[int, int]:
    """Convenience function for cross-validation-based embedding parameter estimation.

    Args:
        x: Time series data
        estimator: Configured cross-validation estimator

    Returns:
        Tuple of (tau, dimension) with optimal embedding parameters
    """
    return estimator.estimate(x)
