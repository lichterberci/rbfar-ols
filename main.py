"""
Example usage of the experiment runner with separate config classes.
"""

from typing import List, Optional
import pandas as pd
import torch
import numpy as np
import time
from experiment_runner import (
    ExperimentResult,
    ProposedMethodConfig,
    ControlConfig,
    run_experiment,
    run_comparison_experiments,
    create_default_proposed_config,
    create_default_control_config,
)
from svd_based_optimizer import SvdOptimizer
from ols_optimizer import OlsOptimizer
from design_matrix_constructor import RadialBasisFunction
from lag_estimation import AmiEstimator, OptimalLagSelectionMethod
from dimension_estimation import CaoEstimator, OptimalDimensionSelectionMethod
from permetrics import RegressionMetric


def make_synthetic_series(length=2000, noise=0.4, seed=42):
    """Create a synthetic time series for testing."""
    np.random.seed(seed)
    t = np.linspace(0, 750 * np.pi, length)
    y = (
        0.6 * np.sin(0.6 * t)
        + 0.3 * np.sin(1.3 * t + 0.5)
        + 0.1 * np.sin(2.1 * t + 1.2)
        + np.exp(-0.1 * t) * np.sin(3.7 * t + 0.7)
        + 0.6
        * np.exp(-np.abs(375 - 0.01 * t))
        * np.sin(0.4 * t + 2.1)
        * np.cos(1.8 * t + 0.3)
    )
    return y.astype(np.float32) + noise * np.random.standard_normal(length).astype(
        np.float32
    )


def construct_takens_embedding(
    x: np.ndarray, dim: int, tau: int, fill: Optional[float] = None
) -> np.ndarray:
    """Construct Takens embedding of time series x with dimension dim and lag tau."""
    n = x.shape[0]

    if n - (dim - 1) * tau <= 0:
        raise ValueError("Time series is too short for the given dimension and lag.")

    num_vectors = n - (dim - 1) * tau

    embedding = np.full((num_vectors, dim), fill, dtype=x.dtype)
    for i in range(dim):
        embedding[:, i] = x[i * tau : i * tau + num_vectors]

    return embedding[:num_vectors] if fill is None else embedding


if __name__ == "__main__":
    time_series = make_synthetic_series(length=2000)

    configs = [
        *[
            ProposedMethodConfig(
                approach="no_pretraining",
                optimizer=SvdOptimizer(epsilon=0.01, alpha=alpha),
                m=m,
                post_tune=post_tune,
            )
            for m in [10, 20, 30, 50, 100, 200, 500, 1000]
            for alpha in [1e-2, 1e-1, 5e-1, 1, 5]
            for post_tune in [True, False]
        ]
    ]

    control_config = ControlConfig(m=20)

    proposed_results, control_result = run_comparison_experiments(
        time_series,
        configs,
        control_config,
        train_ratio=0.7,
        device="cuda" if torch.cuda.is_available() else "cpu",
        run_parallel=False,
        show_progress=True,
    )

    if not isinstance(proposed_results, list):
        proposed_results = [proposed_results]

    results_df = pd.DataFrame(
        {
            "Type": [],
            "Name": [],
            "Test R": [],
            "Test R^2": [],
            "Test NRMSE": [],
        }
    )

    for res in proposed_results:
        test_metrics = RegressionMetric(
            y_pred=res.test_predictions,
            y_true=res.test_targets,
        )

        new_row = pd.DataFrame(
            {
                "Type": ["Proposed"],
                "Name": [
                    f"Proposed (m={res.metadata['total_centers']}, alpha={res.metadata['optimizer_params']['_alpha']})"
                ],
                "Test R": [test_metrics.pearson_correlation_coefficient()],
                "Test R^2": [test_metrics.R2()],
                "Test NRMSE": [test_metrics.normalized_root_mean_square_error()],
            }
        )
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    control_test_metrics = RegressionMetric(
        y_pred=control_result.test_predictions,
        y_true=control_result.test_targets,
    )

    new_row = pd.DataFrame(
        {
            "Type": ["Control"],
            "Name": ["Control (m=20)"],
            "Test R": [control_test_metrics.pearson_correlation_coefficient()],
            "Test R^2": [control_test_metrics.R2()],
            "Test NRMSE": [control_test_metrics.normalized_root_mean_square_error()],
        }
    )
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    print(results_df.sort_values(by=["Type", "Test R^2"], ascending=[True, False]))
