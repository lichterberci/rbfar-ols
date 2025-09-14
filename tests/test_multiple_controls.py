"""
Test module for multiple control configurations functionality.
"""

import pytest
import numpy as np
from experiment_runner import (
    run_comparison_experiments,
    create_default_proposed_config,
    create_default_control_config,
    ProposedMethodConfig,
    ControlConfig,
    ExperimentResult,
)
from ols_optimizer import OlsOptimizer
from svd_based_optimizer import SvdOptimizer


@pytest.fixture
def sample_time_series():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    return np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))


@pytest.fixture
def simple_configs():
    """Create simple configurations for testing."""
    proposed_config = create_default_proposed_config(
        approach="local_pretraining",
        optimizer=SvdOptimizer(epsilon=0.05),
        m=10,  # Reduced for faster testing
    )

    control_configs = [
        create_default_control_config(
            m=5, epochs=10, lr=1e-2
        ),  # Reduced for faster testing
        create_default_control_config(m=7, epochs=15, lr=5e-3),
    ]

    return proposed_config, control_configs


@pytest.mark.slow
def test_single_proposed_multiple_controls(sample_time_series, simple_configs):
    """Test single proposed config with multiple control configs."""
    proposed_config, control_configs = simple_configs

    try:
        proposed_results, control_results = run_comparison_experiments(
            series=sample_time_series,
            proposed_config=proposed_config,
            ctrl_config=control_configs,
            train_ratio=0.7,
            run_parallel=False,
            show_progress=False,
        )

        # Check result types
        assert isinstance(proposed_results, ExperimentResult)
        assert isinstance(control_results, list)
        assert len(control_results) == 2
        assert all(isinstance(result, ExperimentResult) for result in control_results)

    except RuntimeError as e:
        if "Numpy is not available" in str(e):
            pytest.skip("NumPy compatibility issue with current environment")
        else:
            raise


@pytest.mark.slow
def test_multiple_proposed_multiple_controls(sample_time_series):
    """Test multiple proposed configs with multiple control configs."""
    proposed_configs = [
        create_default_proposed_config(
            approach="local_pretraining", m=8, optimizer=SvdOptimizer()
        ),
        create_default_proposed_config(
            approach="no_pretraining", m=10, optimizer=OlsOptimizer()
        ),
    ]

    control_configs = [
        create_default_control_config(m=5, epochs=10),
        create_default_control_config(m=7, epochs=15),
    ]

    try:
        proposed_results, control_results = run_comparison_experiments(
            series=sample_time_series,
            proposed_config=proposed_configs,
            ctrl_config=control_configs,
            train_ratio=0.7,
            run_parallel=False,
            show_progress=False,
        )

        # Check result types
        assert isinstance(proposed_results, list)
        assert isinstance(control_results, list)
        assert len(proposed_results) == 2
        assert len(control_results) == 2
        assert all(isinstance(result, ExperimentResult) for result in proposed_results)
        assert all(isinstance(result, ExperimentResult) for result in control_results)

    except RuntimeError as e:
        if "Numpy is not available" in str(e):
            pytest.skip("NumPy compatibility issue with current environment")
        else:
            raise


@pytest.mark.slow
def test_single_proposed_single_control_backward_compatibility(sample_time_series):
    """Test that single configs still work (backward compatibility)."""
    proposed_config = create_default_proposed_config(m=8)
    control_config = create_default_control_config(m=5, epochs=10)

    try:
        proposed_result, control_result = run_comparison_experiments(
            series=sample_time_series,
            proposed_config=proposed_config,
            ctrl_config=control_config,
            train_ratio=0.7,
            run_parallel=False,
            show_progress=False,
        )

        # Check result types (should be single results, not lists)
        assert isinstance(proposed_result, ExperimentResult)
        assert isinstance(control_result, ExperimentResult)

    except RuntimeError as e:
        if "Numpy is not available" in str(e):
            pytest.skip("NumPy compatibility issue with current environment")
        else:
            raise


def test_config_validation():
    """Test that invalid configurations raise appropriate errors."""
    sample_series = np.random.randn(50)

    # Test invalid proposed config type
    with pytest.raises(TypeError, match="proposed_config must be ProposedMethodConfig"):
        run_comparison_experiments(
            series=sample_series,
            proposed_config="invalid",  # type: ignore
            ctrl_config=create_default_control_config(),
            show_progress=False,
        )

    # Test invalid control config type
    with pytest.raises(TypeError, match="ctrl_config must be ControlConfig"):
        run_comparison_experiments(
            series=sample_series,
            proposed_config=create_default_proposed_config(),
            ctrl_config="invalid",  # type: ignore
            show_progress=False,
        )


def test_experiment_result_structure():
    """Test that ExperimentResult has the expected structure."""
    # This is a structural test that doesn't require running experiments

    # Just test the configuration setup without running
    proposed_config = create_default_proposed_config()
    control_configs = [
        create_default_control_config(m=5),
        create_default_control_config(m=7),
    ]

    # Verify config types are correct
    assert isinstance(proposed_config, ProposedMethodConfig)
    assert isinstance(control_configs, list)
    assert all(isinstance(cfg, ControlConfig) for cfg in control_configs)
