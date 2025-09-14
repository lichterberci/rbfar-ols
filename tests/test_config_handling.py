"""
Test module for experiment_runner configuration handling.
"""

import pytest
from experiment_runner import (
    create_default_proposed_config,
    create_default_control_config,
    ProposedMethodConfig,
    ControlConfig,
)
from ols_optimizer import OlsOptimizer
from svd_based_optimizer import SvdOptimizer


def test_single_config_creation():
    """Test that single configurations are created correctly."""
    single_proposed = create_default_proposed_config()
    single_control = create_default_control_config()

    assert isinstance(single_proposed, ProposedMethodConfig)
    assert isinstance(single_control, ControlConfig)


def test_multiple_config_creation():
    """Test that multiple configurations can be created correctly."""
    multiple_proposed = [
        create_default_proposed_config(approach="local_pretraining"),
        create_default_proposed_config(approach="no_pretraining"),
    ]

    multiple_control = [
        create_default_control_config(m=10, lr=1e-2),
        create_default_control_config(m=15, lr=5e-3),
        create_default_control_config(m=20, lr=1e-3),
    ]

    assert isinstance(multiple_proposed, list)
    assert len(multiple_proposed) == 2
    assert all(isinstance(cfg, ProposedMethodConfig) for cfg in multiple_proposed)

    assert isinstance(multiple_control, list)
    assert len(multiple_control) == 3
    assert all(isinstance(cfg, ControlConfig) for cfg in multiple_control)


def test_type_checking_logic():
    """Test that the type checking logic works correctly."""
    single_proposed = create_default_proposed_config()
    single_control = create_default_control_config()

    multiple_proposed = [
        create_default_proposed_config(approach="local_pretraining"),
        create_default_proposed_config(approach="no_pretraining"),
    ]

    multiple_control = [
        create_default_control_config(m=10, lr=1e-2),
        create_default_control_config(m=15, lr=5e-3),
    ]

    # Test type checking logic
    assert isinstance(single_proposed, ProposedMethodConfig)
    assert isinstance(multiple_proposed, list)
    assert isinstance(single_control, ControlConfig)
    assert isinstance(multiple_control, list)


@pytest.mark.parametrize(
    "proposed_config,control_config,expected_proposed,expected_control",
    [
        ("single", "single", 1, 1),
        ("single", "multiple", 1, 3),
        ("multiple", "single", 2, 1),
        ("multiple", "multiple", 2, 3),
    ],
)
def test_experiment_count_combinations(
    proposed_config, control_config, expected_proposed, expected_control
):
    """Test that all valid combinations calculate experiment counts correctly."""
    # Create configs based on test parameters
    if proposed_config == "single":
        prop_cfg = create_default_proposed_config()
    else:  # multiple
        prop_cfg = [
            create_default_proposed_config(approach="local_pretraining"),
            create_default_proposed_config(approach="no_pretraining"),
        ]

    if control_config == "single":
        ctrl_cfg = create_default_control_config()
    else:  # multiple
        ctrl_cfg = [
            create_default_control_config(m=10, lr=1e-2),
            create_default_control_config(m=15, lr=5e-3),
            create_default_control_config(m=20, lr=1e-3),
        ]

    # Calculate counts
    prop_count = 1 if isinstance(prop_cfg, ProposedMethodConfig) else len(prop_cfg)
    ctrl_count = 1 if isinstance(ctrl_cfg, ControlConfig) else len(ctrl_cfg)

    assert prop_count == expected_proposed
    assert ctrl_count == expected_control
    assert prop_count + ctrl_count == expected_proposed + expected_control


def test_config_parameter_customization():
    """Test that configuration parameters can be customized correctly."""
    # Test proposed config customization
    custom_proposed = create_default_proposed_config(
        approach="local_pretraining", m=150, post_tune=False
    )

    assert custom_proposed.approach == "local_pretraining"
    assert custom_proposed.m == 150
    assert custom_proposed.post_tune is False

    # Test control config customization
    custom_control = create_default_control_config(m=25, epochs=500, lr=1e-3)

    assert custom_control.m == 25
    assert custom_control.epochs == 500
    assert custom_control.lr == 1e-3
