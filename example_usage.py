"""
Example usage of the experiment runner with separate config classes.
"""

import numpy as np
import time
from experiment_runner import (
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


def example_separate_configs():
    """Example showing how to use separate config classes."""

    # Generate test time series
    series = make_synthetic_series()
    print(f"Generated time series with {series.shape} shape")

    # Create proposed method configuration
    proposed_config = ProposedMethodConfig(
        approach="local_pretraining",
        optimizer=SvdOptimizer(epsilon=0.05, alpha=1),
        m=800,  # Reduced for faster execution
        post_tune=True,
        tuning_lr=1e-3,
        tuning_max_epochs=100,
    )

    # Create Control configuration
    control_config = ControlConfig(
        m=12,  # Number of centers
        epochs=500,  # Reduced for faster execution
        lr=3e-2,
        train_sigma=True,
        sigma_global=True,
        patience=8,
    )

    print("\n=== Running Proposed Method Experiment ===")
    proposed_result = run_experiment(series, proposed_config, method_type="proposed")
    print(f"Proposed method: {proposed_result.method_name}")
    print(f"Train predictions shape: {proposed_result.train_predictions.shape}")
    print(f"Test predictions shape: {proposed_result.test_predictions.shape}")
    print(
        f"Selected centers: {proposed_result.metadata.get('selected_centers', 'N/A')}"
    )

    print("\n=== Running Control Experiment ===")
    control_result = run_experiment(series, control_config, method_type="control")
    print(f"Control method: {control_result.method_name}")
    print(f"Train predictions shape: {control_result.train_predictions.shape}")
    print(f"Test predictions shape: {control_result.test_predictions.shape}")
    print(f"Training epochs: {control_result.metadata.get('epochs_trained', 'N/A')}")

    print("\n=== Running Comparison Experiments ===")
    proposed_result2, control_result2 = run_comparison_experiments(
        series, proposed_config, control_config
    )
    print(f"Comparison completed:")
    print(f"  Proposed: {proposed_result2.method_name}")
    print(f"  Control: {control_result2.method_name}")

    print("\n=== Running Parallel Comparison ===")
    proposed_result3, control_result3 = run_comparison_experiments(
        series, proposed_config, control_config, run_parallel=True
    )
    print("Parallel comparison completed:")
    print(f"  Proposed: {proposed_result3.method_name}")
    print(f"  Control: {control_result3.method_name}")

    # Note: For proper evaluation, you'd need the clean target series
    # Here we just show the structure
    print("\nResults have consistent structure:")
    print(
        f"  Both have train_predictions: {hasattr(proposed_result, 'train_predictions') and hasattr(control_result, 'train_predictions')}"
    )
    print(
        f"  Both have test_predictions: {hasattr(proposed_result, 'test_predictions') and hasattr(control_result, 'test_predictions')}"
    )
    print(
        f"  Both have method_name: {hasattr(proposed_result, 'method_name') and hasattr(control_result, 'method_name')}"
    )
    print(
        f"  Both have metadata: {hasattr(proposed_result, 'metadata') and hasattr(control_result, 'metadata')}"
    )


def example_convenience_functions():
    """Example showing convenience functions for creating configs."""

    # Using convenience functions with overrides
    proposed_config = create_default_proposed_config(
        optimizer=OlsOptimizer(rho=0.6), approach="no_pretraining", m=400
    )

    control_config = create_default_control_config(m=8, epochs=200, lr=1e-2)

    print("Created configs using convenience functions:")
    print(f"Proposed approach: {proposed_config.approach}")
    print(f"Proposed optimizer: {type(proposed_config.optimizer).__name__}")
    print(f"Proposed centers: {proposed_config.m}")
    print(f"Control centers: {control_config.m}")
    print(f"Control epochs: {control_config.epochs}")


def example_multiple_configs():
    """Example showing multiple proposed configurations comparison."""

    series = make_synthetic_series(length=800)  # Smaller for faster execution

    # Create multiple proposed method configurations
    configs = [
        ProposedMethodConfig(
            approach="no_pretraining",
            optimizer=SvdOptimizer(epsilon=0.01, alpha=0.5),
            m=200,
            post_tune=False,  # Faster
        ),
        ProposedMethodConfig(
            approach="local_pretraining",
            optimizer=SvdOptimizer(epsilon=0.05, alpha=1.0),
            m=300,
            post_tune=False,  # Faster
        ),
        ProposedMethodConfig(
            approach="no_pretraining",
            optimizer=OlsOptimizer(rho=0.6),
            m=250,
            post_tune=False,  # Faster
        ),
    ]

    control_config = ControlConfig(m=10, epochs=100)  # Faster settings

    print("\n=== Sequential Multiple Configs ===")
    proposed_results_seq, control_result_seq = run_comparison_experiments(
        series, configs, control_config, run_parallel=False
    )
    print(f"Sequential execution completed:")
    print(f"  Control: {control_result_seq.method_name}")
    for i, result in enumerate(proposed_results_seq):
        print(f"  Proposed {i+1}: {result.method_name}")

    print("\n=== Parallel Multiple Configs ===")
    start_time = time.time()

    proposed_results_par, control_result_par = run_comparison_experiments(
        series, configs, control_config, run_parallel=True
    )

    parallel_time = time.time() - start_time

    print(f"Parallel execution completed in {parallel_time:.2f} seconds:")
    print(f"  Control: {control_result_par.method_name}")
    for i, result in enumerate(proposed_results_par):
        print(f"  Proposed {i+1}: {result.method_name}")

    # Compare some basic metrics
    print("\nComparison of results:")
    print(f"  All proposed configs returned: {len(proposed_results_par)} results")
    print(
        f"  Control method consistent: {control_result_seq.method_name == control_result_par.method_name}"
    )

    return proposed_results_par, control_result_par


def example_different_rbf_types():
    """Example showing different RBF types."""

    series = make_synthetic_series(length=800)  # Smaller for quick test

    configs = [
        ProposedMethodConfig(
            rbf=RadialBasisFunction.GAUSSIAN,
            m=200,
            approach="no_pretraining",
            post_tune=False,  # Faster
        ),
        ProposedMethodConfig(
            rbf=RadialBasisFunction.LAPLACIAN,
            m=200,
            approach="no_pretraining",
            post_tune=False,  # Faster
        ),
    ]

    print("\n=== Testing Different RBF Types ===")
    for i, config in enumerate(configs):
        result = run_experiment(series, config, method_type="proposed")
        print(f"Config {i+1} ({config.rbf.name}): {result.method_name}")
        print(f"  Metadata: {list(result.metadata.keys())}")


if __name__ == "__main__":
    print("=== Proposed Method Experiment Runner Examples ===")

    example_separate_configs()
    print("\n" + "=" * 50 + "\n")

    example_convenience_functions()
    print("\n" + "=" * 50 + "\n")

    example_multiple_configs()
    print("\n" + "=" * 50 + "\n")

    example_different_rbf_types()

    print("\n=== All Examples Completed ===")
