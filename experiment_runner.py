"""
Experiment runner for RBF-AR methods comparison.

This module provides functionality to run configurable experiments comparing
SVD/OLS-based methods with a control method (Adam-optimized RBF) on given time series data.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import torch
import warnings
from scipy import signal
from scipy.spatial import KDTree
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import os
from dimension_estimation import (
    CaoEstimator,
    OptimalDimensionSelectionMethod,
    estimate_dimension,
)
from design_matrix_constructor import (
    RadialBasisFunction,
    estimate_sigma_median,
    construct_design_matrix_with_local_pretraining,
    construct_design_matrix_with_no_pretraining,
)
from ols_optimizer import OlsOptimizer
from svd_based_optimizer import SvdOptimizer
from optimizer import Optimizer
from lag_estimation import (
    estimate_tau,
    AmiEstimator,
    OptimalLagSelectionMethod,
)


@dataclass
class ExperimentResult:
    """Common return type containing only predicted time-series results."""

    train_predictions: np.ndarray  # Predicted values for training set
    train_targets: np.ndarray  # True target values for training set
    test_predictions: np.ndarray  # Predicted values for test set
    test_targets: np.ndarray  # True target values for test set
    method_name: str  # Name of the method used
    metadata: Dict[str, Any] = (
        None  # Optional metadata (execution time, parameters used, etc.)
    )


@dataclass
class ExperimentConfig:
    """Base configuration for experiments, containing common parameters."""

    # RBF parameters
    n: Optional[int] = None  # AR order (features per row), None for auto-estimation
    embedding_tau: Optional[int] = None  # Time delay (tau), None for auto-estimation
    sigma: Optional[float] = None  # None -> heuristic estimation
    rbf: RadialBasisFunction = RadialBasisFunction.GAUSSIAN

    # Embedding dimension estimation
    max_embedding_dim: int = 20  # for Cao's method


@dataclass
class ProposedMethodConfig(ExperimentConfig):
    """Configuration for RBF-AR experiments using SVD/OLS optimizers."""

    # Method selection
    approach: str = "no_pretraining"  # "no_pretraining" | "local_pretraining"
    optimizer: Optimizer = None  # instance implementing optimize(P, d)

    # RBF parameters
    m: int = 100  # number of candidate centres

    # Method-specific parameters
    ridge: float = 1e-4  # only for local_pretraining
    rho: float = 0.2  # only for local_pretraining
    post_tune: bool = True  # whether to do post-tuning of weights

    # Post-tuning parameters
    tuning_lr: float = 3e-3
    tuning_patience: int = 15
    tuning_max_epochs: int = 200
    tuning_val_split: float = 0.1  # validation split ratio for post-tuning

    def __post_init__(self):
        if self.optimizer is None:
            self.optimizer = SvdOptimizer()


@dataclass
class ControlConfig(ExperimentConfig):
    """Configuration for control experiments using Adam-optimized RBF method."""

    # Control method parameters
    m: int = 14  # number of centers for control method
    epochs: int = 1000
    lr: float = 5e-2
    weight_decay: float = 0.0
    train_sigma: bool = True  # optimize sigma as well
    sigma_global: bool = True  # if False, learn per-centre sigma
    patience: int = 10  # early stopping patience
    val_split: float = 0.1  # validation split ratio


def estimate_embedding_dimension_cao(y: np.ndarray, tau, max_m: int = 20) -> int:
    """Estimate embedding dimension using Cao's method."""

    estimator = CaoEstimator(
        delay=tau,
        max_dim=max_m,
        optimum_selection_method=OptimalDimensionSelectionMethod.E1_E2_COMBINED,
        plot=False,
        verbose=False,
    )
    dim = estimate_dimension(y, estimator)
    return dim


def make_lagged_matrix(
    y: np.ndarray, n: int, tau: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create lagged feature matrix and target vector."""
    l = len(y)

    num_rows = l - n * tau
    if num_rows <= 0:
        raise ValueError("Not enough data for embedding with given n and tau.")

    # Construct the lagged matrix by iterating over dimensions (columns)
    X = np.column_stack([y[j * tau : j * tau + num_rows] for j in range(n)])

    # Target vector is the next value after the embedding
    d = y[n * tau : n * tau + num_rows]

    return X, d


def estimate_tau_for_series(y: np.ndarray) -> int:
    return estimate_tau(
        y,
        AmiEstimator(
            max_lag=100,
            optimum_selection_method=OptimalLagSelectionMethod.FIRST_LOC_MIN,
        ),
    )


def run_proposed_experiment(
    series: np.ndarray,
    config: ProposedMethodConfig,
    train_ratio: float = 0.7,
    device: str = "cpu",
) -> ExperimentResult:
    """
    Run proposed method experiment with SVD/OLS-based methods.

    Args:
        series: Time series data (noisy version)
        config: Proposed method experiment configuration
        train_ratio: Ratio of data to use for training
        device: Device to run computations on

    Returns:
        ExperimentResult containing predictions and metadata
    """
    series = series.astype(np.float32)

    # Estimate tau from autocorrelation
    if config.embedding_tau is not None:
        tau = config.embedding_tau
    else:
        tau = estimate_tau_for_series(series)

    # Estimate embedding dimension if not provided
    n = config.n
    if n is None:
        n = estimate_embedding_dimension_cao(
            series, tau=tau, max_m=config.max_embedding_dim
        )

    # Create lagged matrix
    X, d = make_lagged_matrix(series, n, tau)

    # Train/test split
    split_idx = int(train_ratio * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    d_train, d_test = d[:split_idx], d[split_idx:]

    # Convert to torch tensors
    X_train = torch.from_numpy(X_train).to(device)
    X_test = torch.from_numpy(X_test).to(device)
    d_train = torch.from_numpy(d_train).to(device)
    d_test = torch.from_numpy(d_test).to(device)

    # Estimate sigma if not provided
    sigma_val = config.sigma
    if sigma_val is None:
        sigma_val = estimate_sigma_median(X_train.numpy())

    # Construct design matrix
    if config.approach == "local_pretraining":
        train_candidates = torch.randperm(X_train.shape[0], dtype=torch.long)[
            : min(config.m, X_train.shape[0])
        ]
        centres = X_train[train_candidates]

        # Construct P_train with local pretraining
        P_train, nu_hat_train = construct_design_matrix_with_local_pretraining(
            X_train.numpy(),
            d_train.numpy(),
            centres=centres.numpy(),
            sigma=sigma_val,
            radial_basis_function=config.rbf,
            ridge=config.ridge,
            rho=config.rho,
            return_weights=True,
        )

        # Construct P_test using the same centres and learned weights
        P_test = construct_design_matrix_with_local_pretraining(
            X_test,
            d_test,  # Not used when weights are provided
            centres=centres,
            weights=nu_hat_train,
            sigma=sigma_val,
            radial_basis_function=config.rbf,
            return_weights=False,
        )

    else:  # no_pretraining
        # Use stacked approach as in notebook
        train_candidates = torch.randperm(X_train.shape[0], dtype=torch.long)[
            : min(config.m, X_train.shape[0])
        ]
        if train_candidates.shape[0] == 0:
            raise ValueError("Not enough training samples to select centres.")
        centers = X_train[train_candidates]
        lt = X_test.shape[0]  # length of test set
        P_stack = construct_design_matrix_with_no_pretraining(
            torch.cat([X_test, X_train], dim=0),
            torch.cat([d_test, d_train], dim=0),
            centres=centers,
            sigma=sigma_val,
            radial_basis_function=config.rbf,
        )
        P_train, P_test = P_stack[lt:], P_stack[:lt]

    # Optimize weights
    sel_center_idxs, model_weights = config.optimizer.optimize(P_train, d_train)

    # Post-tuning if enabled
    weights = model_weights.clone()

    if config.post_tune and len(sel_center_idxs) > 0:
        weights = weights.clone().detach().requires_grad_(True)

        # Post-tune the selected weights using Adam with early stopping
        optim = torch.optim.Adam([weights], lr=config.tuning_lr)

        best_loss = float("inf")
        patience_counter = 0

        for _ in range(config.tuning_max_epochs):
            pred = P_train[:, sel_center_idxs] @ weights
            loss = torch.mean((pred - d_train) ** 2)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.tuning_patience:
                break

    # Generate predictions
    with torch.no_grad():
        train_pred = (P_train[:, sel_center_idxs] @ weights).detach().cpu().numpy()
        test_pred = (P_test[:, sel_center_idxs] @ weights).detach().cpu().numpy()

    metadata = {
        "tau": tau,
        "n": n,
        "sigma": sigma_val,
        "selected_centers": len(sel_center_idxs),
        "total_centers": config.m,
        "approach": config.approach,
        "optimizer_type": type(config.optimizer).__name__,
        "post_tuned": config.post_tune,
        "optimizer_params": vars(config.optimizer),
    }

    return ExperimentResult(
        train_predictions=train_pred,
        test_predictions=test_pred,
        train_targets=d_train.cpu().numpy(),
        test_targets=d_test.cpu().numpy(),
        method_name=f"PROPOSED-{config.approach}-{type(config.optimizer).__name__}",
        metadata=metadata,
    )


def run_control_experiment(
    series: np.ndarray,
    config: ControlConfig,
    train_ratio: float = 0.7,
    device: str = "cpu",
) -> ExperimentResult:
    """
    Run control experiment with Adam-optimized RBF method.

    Args:
        series: Time series data (noisy version)
        config: Control experiment configuration
        train_ratio: Ratio of data to use for training
        device: Device to run computations on

    Returns:
        ExperimentResult containing predictions and metadata
    """
    series = series.astype(np.float32)

    # Estimate tau from autocorrelation
    if config.embedding_tau is not None:
        tau = config.embedding_tau
    else:
        tau = estimate_tau_for_series(series)

    # Estimate embedding dimension if not provided
    n = config.n
    if n is None:
        n = estimate_embedding_dimension_cao(
            series, tau=tau, max_m=config.max_embedding_dim
        )

    # Create lagged matrix
    X, d = make_lagged_matrix(series, n, tau)

    # Train/test split
    split_idx = int(train_ratio * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    d_train, d_test = d[:split_idx], d[split_idx:]

    # Convert to torch tensors
    X_train = torch.from_numpy(X_train).to(device)
    X_test = torch.from_numpy(X_test).to(device)
    d_train = torch.from_numpy(d_train).to(device)
    d_test = torch.from_numpy(d_test).to(device)

    # Estimate sigma if not provided
    sigma_val = config.sigma
    if sigma_val is None:
        sigma_val = estimate_sigma_median(X_train.numpy())

    # Validation split (10% of training data)
    val_size = int(config.val_split * X_train.shape[0])
    train_size = X_train.shape[0] - val_size
    X_train_split = X_train[:train_size]
    d_train_split = d_train[:train_size]
    X_val_split = X_train[train_size:]
    d_val_split = d_train[train_size:]

    # Initialize control model parameters
    torch.manual_seed(0)
    m_ctrl = min(config.m, X_train_split.shape[0])

    C_param = (
        X_train_split[torch.randperm(X_train_split.shape[0])[:m_ctrl]]
        .clone()
        .detach()
        .requires_grad_(True)
    )  # (m, n) - centers

    Theta_param = (0.01 * torch.randn(m_ctrl, X_train_split.shape[1])).requires_grad_(
        True
    )  # (m, n) - projection weights

    # Initialize sigma parameter
    if config.sigma_global:
        log_sigma_param = torch.tensor(
            np.log(float(sigma_val) + 1e-6),
            dtype=X_train_split.dtype,
            requires_grad=config.train_sigma,
        )
    else:
        init_sigma = torch.full((m_ctrl,), float(sigma_val), dtype=X_train_split.dtype)
        log_sigma_param = torch.log(init_sigma + 1e-6)
        log_sigma_param.requires_grad_(config.train_sigma)

    # Setup optimizer
    params = [C_param, Theta_param]
    if config.train_sigma:
        params.append(log_sigma_param)

    optim = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)

    def _phi_from_params(X_in, C, log_sigma):
        """Compute RBF features from parameters."""
        x_sq = (X_in * X_in).sum(dim=1, keepdim=True)  # (l, 1)
        c_sq = (C * C).sum(dim=1)  # (m,)
        dist_sq = x_sq + c_sq.unsqueeze(0) - 2.0 * (X_in @ C.T)  # (l, m)
        dist_sq = torch.clamp(dist_sq, min=0.0)

        if config.sigma_global:
            sigma = torch.nn.functional.softplus(log_sigma) + 1e-6
        else:
            sigma = (
                torch.nn.functional.softplus(log_sigma).unsqueeze(0) + 1e-6
            )  # (1, m)

        if config.rbf == RadialBasisFunction.GAUSSIAN:
            if config.sigma_global:
                denom = 2.0 * (sigma * sigma)
            else:
                denom = 2.0 * (sigma * sigma)  # (1, m)
            phi = torch.exp(-dist_sq / denom)
        elif config.rbf == RadialBasisFunction.LAPLACIAN:
            dist = torch.sqrt(dist_sq + 1e-12)
            phi = torch.exp(-dist / (sigma + 1e-12))
        else:
            raise ValueError("Unknown RBF in config")

        return phi

    # Early stopping setup
    best_val_loss = float("inf")
    best_epoch = 0
    best_params = [p.clone() for p in params]

    # Training loop
    for epoch in range(1, config.epochs + 1):
        # Training step
        phi_tr = _phi_from_params(X_train_split, C_param, log_sigma_param)  # (l_tr, m)
        proj_tr = X_train_split @ Theta_param.T  # (l_tr, m)
        y_hat_tr_ctrl = (phi_tr * proj_tr).sum(dim=1)  # (l_tr,)
        loss = torch.mean((y_hat_tr_ctrl - d_train_split) ** 2)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Validation step
        with torch.no_grad():
            phi_val = _phi_from_params(X_val_split, C_param, log_sigma_param)
            proj_val = X_val_split @ Theta_param.T
            y_hat_val_ctrl = (phi_val * proj_val).sum(dim=1)
            val_loss = torch.mean((y_hat_val_ctrl - d_val_split) ** 2)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch
            best_params = [p.clone() for p in params]
        elif epoch - best_epoch >= config.patience:
            break

    # Restore best parameters
    for i, p in enumerate(params):
        p.data = best_params[i].data

    # Generate predictions on full train and test sets
    with torch.no_grad():
        # Training predictions
        phi_tr = _phi_from_params(X_train, C_param, log_sigma_param)
        proj_tr = X_train @ Theta_param.T
        train_pred = (phi_tr * proj_tr).sum(dim=1).cpu().numpy()

        # Test predictions
        phi_te = _phi_from_params(X_test, C_param, log_sigma_param)
        proj_te = X_test @ Theta_param.T
        test_pred = (phi_te * proj_te).sum(dim=1).cpu().numpy()

    metadata = {
        "tau": tau,
        "n": n,
        "sigma": sigma_val,
        "m_ctrl": m_ctrl,
        "epochs_trained": min(epoch, config.epochs),
        "best_epoch": best_epoch,
        "final_val_loss": float(best_val_loss),
        "train_sigma": config.train_sigma,
        "sigma_global": config.sigma_global,
    }

    return ExperimentResult(
        train_predictions=train_pred,
        test_predictions=test_pred,
        train_targets=d_train.cpu().numpy(),
        test_targets=d_test.cpu().numpy(),
        method_name="Control-Adam-RBF",
        metadata=metadata,
    )


def run_experiment(
    series: np.ndarray,
    config: Union[ProposedMethodConfig, ControlConfig],
    method_type: str = "proposed",
    train_ratio: float = 0.7,
    device: str = "cpu",
) -> ExperimentResult:
    """
    Run a single experiment with the specified method.

    Args:
        series: Time series data (noisy version)
        config: Experiment configuration (ProposedMethodConfig or ControlConfig)
        method_type: "proposed" for SVD/OLS methods, "control" for Adam-optimized method
        train_ratio: Ratio of data to use for training
        device: Device to run computations on

    Returns:
        ExperimentResult containing predictions and metadata
    """
    if method_type == "proposed":
        if not isinstance(config, ProposedMethodConfig):
            raise TypeError(
                "For 'proposed' method_type, config must be ProposedMethodConfig"
            )
        return run_proposed_experiment(series, config, train_ratio, device)
    elif method_type == "control":
        if not isinstance(config, ControlConfig):
            raise TypeError("For 'control' method_type, config must be ControlConfig")
        return run_control_experiment(series, config, train_ratio, device)
    else:
        raise ValueError(
            f"Unknown method_type: {method_type}. Use 'proposed' or 'control'."
        )


# Example usage and convenience functions
def create_default_proposed_config(**kwargs) -> ProposedMethodConfig:
    """Create a default proposed method configuration with optional overrides."""
    defaults = {
        "approach": "local_pretraining",
        "optimizer": SvdOptimizer(epsilon=0.05, alpha=1),
        "post_tune": True,
        "m": 1600,
    }
    defaults.update(kwargs)
    return ProposedMethodConfig(**defaults)


def create_default_control_config(**kwargs) -> ControlConfig:
    """Create a default control configuration with optional overrides."""
    defaults = {"m": 14, "epochs": 1000, "lr": 5e-2}
    defaults.update(kwargs)
    return ControlConfig(**defaults)


def run_comparison_experiments(
    series: np.ndarray,
    proposed_config: Union[ProposedMethodConfig, List[ProposedMethodConfig]],
    ctrl_config: Optional[Union[ControlConfig, List[ControlConfig]]] = None,
    train_ratio: float = 0.7,
    device: str = "cpu",
    run_parallel: bool = False,
    show_progress: bool = True,
) -> Tuple[
    Union[ExperimentResult, List[ExperimentResult]],
    Union[ExperimentResult, List[ExperimentResult]],
]:
    """
    Run proposed method(s) and control experiments for comparison.

    Args:
        series: Time series data
        proposed_config: Configuration(s) for proposed method(s)
        ctrl_config: Configuration(s) for control method(s) (uses default if None)
        train_ratio: Training data ratio
        device: Device to run on (for GPU parallelism, use "cuda" but see notes below)
        run_parallel: If True, run experiments in parallel; if False, run sequentially
        show_progress: If True, show progress bars during lag estimation

    Returns:
        Tuple of (proposed_result(s), control_result(s))
        - If single configs: (ExperimentResult, ExperimentResult)
        - If single proposed, multiple control: (ExperimentResult, List[ExperimentResult])
        - If multiple proposed, single control: (List[ExperimentResult], ExperimentResult)
        - If multiple configs: (List[ExperimentResult], List[ExperimentResult])

    Notes on Parallelism:
        - CPU device: True parallelism with ThreadPoolExecutor
        - GPU device: Limited parallelism due to CUDA context conflicts
        - For best GPU performance with multiple configs, consider:
          1. Using multiple GPU devices if available
          2. Running sequentially on single GPU
          3. Using smaller batch sizes per experiment
    """
    if ctrl_config is None:
        ctrl_config = create_default_control_config()

    # Calculate total number of experiments for progress bar
    num_proposed = len(proposed_config) if isinstance(proposed_config, list) else 1
    num_control = len(ctrl_config) if isinstance(ctrl_config, list) else 1
    total_experiments = num_proposed + num_control

    progress_bar = tqdm(
        range(total_experiments),
        disable=not show_progress,
        desc="Running Experiments",
        unit="experiment",
    )

    run_proposed_exp_and_update_progress = lambda cfg: (
        progress_bar.update(1),
        run_proposed_experiment(series, cfg, train_ratio, device),
    )[1]
    run_control_exp_and_update_progress = lambda cfg: (
        progress_bar.update(1),
        run_control_experiment(series, cfg, train_ratio, device),
    )[1]
    # Determine config types
    proposed_is_single = isinstance(proposed_config, ProposedMethodConfig)
    proposed_is_multiple = isinstance(proposed_config, list)
    control_is_single = isinstance(ctrl_config, ControlConfig)
    control_is_multiple = isinstance(ctrl_config, list)

    if not (proposed_is_single or proposed_is_multiple):
        raise TypeError(
            "proposed_config must be ProposedMethodConfig or List[ProposedMethodConfig]"
        )

    if not (control_is_single or control_is_multiple):
        raise TypeError("ctrl_config must be ControlConfig or List[ControlConfig]")

    # Convert single configs to lists for uniform processing
    proposed_configs = [proposed_config] if proposed_is_single else proposed_config
    control_configs = [ctrl_config] if control_is_single else ctrl_config

    # Run experiments
    if run_parallel and device == "cpu":
        # True parallelism only beneficial on CPU
        max_workers = len(proposed_configs) + len(control_configs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all proposed method experiments
            proposed_futures = []
            for config in proposed_configs:
                future = executor.submit(run_proposed_exp_and_update_progress, config)
                proposed_futures.append(future)

            # Submit all control experiments
            control_futures = []
            for config in control_configs:
                future = executor.submit(run_control_exp_and_update_progress, config)
                control_futures.append(future)

            # Collect results in order
            proposed_results = []
            for future in proposed_futures:
                proposed_results.append(future.result())

            control_results = []
            for future in control_futures:
                control_results.append(future.result())
    else:
        # Run sequentially (recommended for GPU and fallback)
        if run_parallel and device != "cpu":
            warnings.warn(
                "Parallel execution on GPU may not provide speedup due to CUDA context conflicts. "
                "Running sequentially instead for better GPU utilization.",
                UserWarning,
            )

        # Run proposed experiments sequentially
        proposed_results = []
        for config in proposed_configs:
            result = run_proposed_exp_and_update_progress(config)
            proposed_results.append(result)

        # Run control experiments sequentially
        control_results = []
        for config in control_configs:
            result = run_control_exp_and_update_progress(config)
            control_results.append(result)

    # Convert back to single results if original inputs were single configs
    final_proposed_results = (
        proposed_results[0] if proposed_is_single else proposed_results
    )
    final_control_results = control_results[0] if control_is_single else control_results

    return final_proposed_results, final_control_results
