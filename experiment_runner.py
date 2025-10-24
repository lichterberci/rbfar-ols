"""
Experiment runner for RBF-AR methods comparison.

This module provides functionality to run configurable experiments comparing
SVD/OLS-based methods with a control method (Adam-optimized RBF) on given time series data.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.spatial import KDTree
from tqdm import tqdm

from design_matrix_constructor import (
    RadialBasisFunction,
    construct_design_matrix_with_local_pretraining,
    construct_design_matrix_with_no_pretraining,
    estimate_sigma_median,
    select_centres_kmeans,
)
from dimension_estimation import (
    CaoEstimator,
    OptimalDimensionSelectionMethod,
    estimate_dimension,
)
from lag_estimation import AmiEstimator, OptimalLagSelectionMethod, estimate_tau
from ols_optimizer import OlsOptimizer
from optimizer import Optimizer
from svd_based_optimizer import SvdOptimizer

from em_vp import EMVPConfig, EMVPDiagnostics, EMVPModel, EMVPTrainer
from modified_em_vp import (
    ModifiedEMVPConfig,
    ModifiedEMVPDiagnostics,
    ModifiedEMVPModel,
    ModifiedEMVPTrainer,
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

    # Centre selection parameters
    use_kmeans_centres: bool = (
        False  # if True, use K-means for centre selection instead of random
    )
    kmeans_max_iters: int = 100  # maximum iterations for K-means clustering
    kmeans_tol: float = 1e-4  # tolerance for K-means convergence

    # Sigma estimation parameters
    use_local_sigma: bool = False  # if True, use local KNN-based sigma estimation
    local_sigma_k: int = 5  # number of nearest neighbors for local sigma estimation

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
    """Base configuration for control experiments."""


@dataclass
class ControlGDConfig(ControlConfig):
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
    ridge: float = 0.0  # L2 regularization (not used in Adam)


@dataclass
class ControlEMVPConfig(ControlConfig):
    """Configuration for control experiments using the EM-VP algorithm."""

    num_components: int = 5
    max_iters: int = 200
    tol_loglik: float = 1e-5
    tol_param: float = 1e-4
    min_variance: float = 1e-6
    ridge: float = 1e-5
    init_responsibility_temp: float = 1.0
    init_width_scale: float = 0.5
    loglik_window: int = 5
    responsibility_floor: float = 1e-8
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32
    centre_sampling_ratio: float = 1.0
    centre_restarts: int = 1


@dataclass
class ProposedModifiedEMVPConfig(ExperimentConfig):
    """Configuration for proposed experiments using the Modified EM-VP algorithm.

    This allows experimenting with novel modifications to the EM-VP algorithm
    as a proposed method alongside SVD/OLS-based approaches.
    """

    # Core EM-VP parameters (similar to ControlEMVPConfig but as a proposed method)
    num_components: int = 5
    max_iters: int = 200
    tol_loglik: float = 1e-5
    tol_param: float = 1e-4
    min_variance: float = 1e-6
    ridge: float = 1e-5
    init_responsibility_temp: float = 1.0
    init_width_scale: float = 0.5
    loglik_window: int = 5
    responsibility_floor: float = 1e-8
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32
    centre_sampling_ratio: float = 1.0
    centre_restarts: int = 1

    # Additional parameters for novel modifications
    # TSVD parameters for function parameter learning
    use_tsvd: bool = True  # Use TSVD instead of L2 regularization
    tsvd_epsilon: float = 1e-2  # Convergence threshold for TSVD
    tsvd_alpha: float = 1e-5  # Regularization parameter for TSVD
    tsvd_delta: float = 1e-6  # Sparsity threshold for TSVD
    tsvd_beta: Optional[float] = None  # Max singular value clipping


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
    config: Union[ProposedMethodConfig, ProposedModifiedEMVPConfig],
    train_ratio: float = 0.7,
    device: str = "cpu",
    seed: int = 0,
) -> ExperimentResult:
    """
    Run proposed method experiment with SVD/OLS-based methods or Modified EM-VP.

    Args:
        series: Time series data (noisy version)
        config: Proposed method experiment configuration (ProposedMethodConfig or ProposedModifiedEMVPConfig)
        train_ratio: Ratio of data to use for training
        device: Device to run computations on

    Returns:
        ExperimentResult containing predictions and metadata
    """
    series = series.astype(np.float32)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    # Handle ProposedModifiedEMVPConfig separately
    if isinstance(config, ProposedModifiedEMVPConfig):
        return _run_proposed_with_modified_emvp(
            config=config,
            X_train=X_train,
            X_test=X_test,
            d_train=d_train,
            d_test=d_test,
            tau=tau,
            n=n,
            base_device=device,
        )

    # Estimate sigma if not provided (for ProposedMethodConfig only)
    sigma_val = config.sigma
    if sigma_val is None:
        sigma_val = estimate_sigma_median(X_train.numpy())

    # Centre selection based on configuration
    if config.use_kmeans_centres:
        # Use K-means for centre selection
        centers = select_centres_kmeans(
            X_train,
            m=min(config.m, X_train.shape[0]),
            max_iters=config.kmeans_max_iters,
            tol=config.kmeans_tol,
        )
    else:
        # Use random centre selection (original method)
        train_candidates = torch.randperm(X_train.shape[0], dtype=torch.long)[
            : min(config.m, X_train.shape[0])
        ]
        centers = X_train[train_candidates]

    # Construct design matrix
    if config.approach in ["local_pretraining", "pretraining"]:
        # Construct P_train with local pretraining
        P_train, nu_hat_train = construct_design_matrix_with_local_pretraining(
            X_train,
            d_train,
            centres=centers,
            sigma=sigma_val,
            use_local_sigma=config.use_local_sigma,
            local_sigma_k=config.local_sigma_k,
            radial_basis_function=config.rbf,
            ridge=config.ridge,
            rho=config.rho,
            return_weights=True,
        )

        P_test = construct_design_matrix_with_local_pretraining(
            X_test,
            d_test,  # Not used when weights are provided
            centres=centers,
            weights=nu_hat_train,  # Use the pretraining weights
            sigma=sigma_val,
            use_local_sigma=config.use_local_sigma,
            local_sigma_k=config.local_sigma_k,
            radial_basis_function=config.rbf,
            rho=config.rho,
            return_weights=False,
        )

    else:  # no_pretraining
        if centers.shape[0] == 0:
            raise ValueError("Not enough training samples to select centres.")
        lt = X_test.shape[0]  # length of test set
        P_stack = construct_design_matrix_with_no_pretraining(
            torch.cat([X_test, X_train], dim=0),
            torch.cat([d_test, d_train], dim=0),
            centres=centers,
            sigma=sigma_val,
            radial_basis_function=config.rbf,
        )
        P_train, P_test = P_stack[lt:], P_stack[:lt]
        # Store the centers used for RBF construction

    # Optimize weights
    selected_indices, model_weights = config.optimizer.optimize(P_train, d_train)

    if config.post_tune:
        if config.approach in ["local_pretraining", "pretraining"]:

            # Map selected indices to centers and pretraining weights
            # Handle the case where selected_indices might include the constant term (last index)
            # Design matrix shape: (l, m+1) where m+1 includes m centers + 1 constant term
            # The centers array has shape (m, n), so valid center indices are 0 to m-1
            # The constant term corresponds to index m in the design matrix
            m = centers.shape[0]

            # Separate center indices from constant term index
            # Filter out the constant term index if it exists
            center_mask = selected_indices < m
            center_indices = selected_indices[center_mask]

            # Check if constant term is selected
            constant_term_selected = (selected_indices >= m).any()
            center_indices = selected_indices[center_mask]

            # Check if constant term is selected
            constant_term_selected = torch.any(selected_indices == m)

            # For pretraining approach: selected_indices refer to center indices
            adjustable_weights = model_weights.clone().detach().requires_grad_(True)

            # Select corresponding centers and pretraining weights
            if len(center_indices) > 0:
                selected_centres = centers[center_indices]
                nu_hat_train_for_selected_centres = nu_hat_train[center_indices]
            else:
                # Fallback: use all centers if no centers are selected
                selected_centres = centers
                nu_hat_train_for_selected_centres = nu_hat_train
                center_indices = torch.arange(centers.shape[0])
                # Update selected_indices to include all centers
                selected_indices = center_indices

            # Post-tune using Adam with early stopping
            optim = torch.optim.Adam(
                [adjustable_weights],
                lr=config.tuning_lr,
            )

            best_weights = adjustable_weights.clone()
            best_loss = float("inf")
            patience_counter = 0

            # Split training data for validation
            train_size = int((1 - config.tuning_val_split) * X_train.shape[0])
            X_train_for_tuning = X_train[:train_size]
            X_valid_for_tuning = X_train[train_size:]
            d_train_for_tuning = d_train[:train_size]
            d_valid_for_tuning = d_train[train_size:]

            # For pretraining, use only the selected centers for tuning
            P_train_for_tuning = construct_design_matrix_with_local_pretraining(
                X_train_for_tuning,
                d_train_for_tuning,  # Not used when weights are provided
                centres=selected_centres,
                weights=nu_hat_train_for_selected_centres,
                sigma=sigma_val,
                radial_basis_function=config.rbf,
                return_weights=False,
                rho=config.rho,
            )
            P_valid_for_tuning = construct_design_matrix_with_local_pretraining(
                X_valid_for_tuning,
                d_valid_for_tuning,  # Not used when weights are provided
                centres=selected_centres,
                weights=nu_hat_train_for_selected_centres,
                sigma=sigma_val,
                radial_basis_function=config.rbf,
                return_weights=False,
                rho=config.rho,
            )
            P_test_for_tuning = construct_design_matrix_with_local_pretraining(
                X_test,
                d_test,  # Not used when weights are provided
                centres=selected_centres,
                weights=nu_hat_train_for_selected_centres,
                sigma=sigma_val,
                radial_basis_function=config.rbf,
                return_weights=False,
                rho=config.rho,
            )

            # Create new indices for the tuning matrices
            # The tuning matrices have shape (l, len(selected_centres)+1)
            # We need to map the selected center indices and constant term appropriately
            num_selected_centers = len(center_indices)
            tuning_indices = torch.arange(
                num_selected_centers, device=selected_indices.device
            )
            if constant_term_selected:
                # Add the constant term index (which is the last column in the tuning matrix)
                constant_idx = torch.tensor(
                    [num_selected_centers], device=selected_indices.device
                )
                tuning_indices = torch.cat([tuning_indices, constant_idx])

            for _ in range(config.tuning_max_epochs):
                # Training step
                pred = (
                    P_train_for_tuning[:, tuning_indices] @ adjustable_weights
                ).squeeze()
                loss = torch.mean(
                    (pred - d_train_for_tuning) ** 2
                )  # TODO: regularization (L2?)

                optim.zero_grad()
                loss.backward()
                optim.step()

                # Validation step
                with torch.no_grad():
                    val_pred = (
                        P_valid_for_tuning[:, tuning_indices] @ adjustable_weights
                    ).squeeze()
                    val_loss = torch.mean((val_pred - d_valid_for_tuning) ** 2)

                # Early stopping based on validation loss
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    patience_counter = 0
                    best_weights = adjustable_weights.clone()
                else:
                    patience_counter += 1

                if patience_counter >= config.tuning_patience:
                    break

            # Restore best weights
            adjustable_weights.data = best_weights.data

            # Generate predictions using tuned weights on selected centers only
            with torch.no_grad():
                # Use the same mapping as used during training
                train_pred = (
                    (P_train_for_tuning[:, tuning_indices] @ adjustable_weights)
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                test_pred = (
                    (P_test_for_tuning[:, tuning_indices] @ adjustable_weights)
                    .squeeze()
                    .cpu()
                    .numpy()
                )
        else:  # no-pretraining approach
            # For no-pretraining: tune centers, weights, and sigma

            def _phi_from_params(X_in, C, log_sigma):
                """Compute RBF features from parameters."""
                x_sq = (X_in * X_in).sum(dim=1, keepdim=True)  # (l, 1)
                c_sq = (C * C).sum(dim=1)  # (m,)
                dist_sq = x_sq + c_sq.unsqueeze(0) - 2.0 * (X_in @ C.T)  # (l, m)
                dist_sq = torch.clamp(dist_sq, min=0.0)

                sigma = F.softplus(log_sigma) + 1e-6

                if config.rbf == RadialBasisFunction.GAUSSIAN:
                    denom = 2.0 * (sigma * sigma)
                    phi = torch.exp(-dist_sq / denom)
                elif config.rbf == RadialBasisFunction.LAPLACIAN:
                    dist = torch.sqrt(dist_sq + 1e-12)
                    phi = torch.exp(-dist / (sigma + 1e-12))
                else:
                    raise ValueError("Unknown RBF in config")

                return phi

            def _construct_design_matrix(X_in, centers_in, log_sigma):
                """Construct the full design matrix P (l, (n+1)*(m+1)) from X and centers."""
                phi = _phi_from_params(X_in, centers_in, log_sigma)  # (l, m)
                l, n = X_in.shape
                m = centers_in.shape[0]

                # Build design matrix with new structure: (n+1) functions × (m+1) parameters
                ones_col = torch.ones(
                    l, 1, device=X_in.device, dtype=X_in.dtype
                )  # Global terms

                # Build the design matrix function by function
                P_blocks = []

                # φ₀(X) - constant function: [1, Ψ₁(X), Ψ₂(X), ..., Ψₘ(X)]
                phi_0_block = torch.cat([ones_col, phi], dim=1)  # Shape: (l, m+1)
                P_blocks.append(phi_0_block)

                # φᵢ(X) - coefficient functions for i=1..n: [Xᵢ, Xᵢ*Ψ₁(X), Xᵢ*Ψ₂(X), ..., Xᵢ*Ψₘ(X)]
                for i in range(n):
                    X_i = X_in[:, i : i + 1]  # Shape: (l, 1)
                    phi_i_block = torch.cat([X_i, X_i * phi], dim=1)  # Shape: (l, m+1)
                    P_blocks.append(phi_i_block)

                # Concatenate all blocks horizontally
                P = torch.cat(P_blocks, dim=1)  # Shape: (l, (n+1)*(m+1))
                return P

            # Create adjustable parameters - only tune centers, weights, and sigma
            adjustable_centers = centers.clone().detach().requires_grad_(True)

            # For no-pretraining, we need to create a full weight vector ((n+1)*(m+1),)
            # The optimizer gave us weights only for selected indices
            m, n = adjustable_centers.shape[0], X_train.shape[1]
            total_params = (n + 1) * (m + 1)
            full_weights = torch.zeros(
                total_params, device=device, dtype=model_weights.dtype
            )
            full_weights[selected_indices] = model_weights
            adjustable_weights = full_weights.clone().detach().requires_grad_(True)

            adjustable_log_sigma = torch.tensor(
                float(np.log(sigma_val) + 1e-6),
                device=device,
                dtype=adjustable_centers.dtype,
                requires_grad=True,
            )

            # Post-tune using Adam with early stopping
            optim = torch.optim.Adam(
                [adjustable_centers, adjustable_weights, adjustable_log_sigma],
                lr=config.tuning_lr,
            )

            best_params = [
                p.clone()
                for p in [adjustable_centers, adjustable_weights, adjustable_log_sigma]
            ]
            best_loss = float("inf")
            patience_counter = 0

            # Split training data for validation
            train_size = int((1 - config.tuning_val_split) * X_train.shape[0])
            X_train_for_tuning = X_train[:train_size]
            X_valid_for_tuning = X_train[train_size:]
            d_train_for_tuning = d_train[:train_size]
            d_valid_for_tuning = d_train[train_size:]

            for _ in range(config.tuning_max_epochs):
                # Training step
                P_tr = _construct_design_matrix(
                    X_train_for_tuning, adjustable_centers, adjustable_log_sigma
                )  # (l_tr, m*n)
                pred = (P_tr @ adjustable_weights).squeeze()  # (l_tr,)
                loss = torch.mean((pred - d_train_for_tuning) ** 2)

                optim.zero_grad()
                loss.backward()
                optim.step()

                # Validation step
                with torch.no_grad():
                    P_val = _construct_design_matrix(
                        X_valid_for_tuning, adjustable_centers, adjustable_log_sigma
                    )  # (l_val, m*n)
                    val_pred = (P_val @ adjustable_weights).squeeze()  # (l_val,)
                    val_loss = torch.mean((val_pred - d_valid_for_tuning) ** 2)

                # Early stopping based on validation loss
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    patience_counter = 0
                    best_params = [
                        p.clone()
                        for p in [
                            adjustable_centers,
                            adjustable_weights,
                            adjustable_log_sigma,
                        ]
                    ]
                else:
                    patience_counter += 1

                if patience_counter >= config.tuning_patience:
                    break

            # Restore best parameters
            for i, p in enumerate(
                [adjustable_centers, adjustable_weights, adjustable_log_sigma]
            ):
                p.data = best_params[i].data

            # Generate predictions using tuned parameters
            with torch.no_grad():
                # Training predictions
                P_tr = _construct_design_matrix(
                    X_train, adjustable_centers, adjustable_log_sigma
                )  # (l_tr, m*n)
                train_pred = (P_tr @ adjustable_weights).squeeze().cpu().numpy()

                # Test predictions
                P_te = _construct_design_matrix(
                    X_test, adjustable_centers, adjustable_log_sigma
                )  # (l_te, m*n)
                test_pred = (P_te @ adjustable_weights).squeeze().cpu().numpy()
    else:  # no post-tuning
        # Handle the constant term index issue for non-post-tuning case
        if config.approach in ["local_pretraining", "pretraining"]:
            # The same logic applies: selected_indices may include constant term index
            m = centers.shape[0] if "centers" in locals() else P_train.shape[1] - 1
            safe_indices = selected_indices[selected_indices <= P_train.shape[1] - 1]
        else:
            safe_indices = selected_indices

        train_pred = (P_train[:, safe_indices] @ model_weights).squeeze().cpu().numpy()
        test_pred = (P_test[:, safe_indices] @ model_weights).squeeze().cpu().numpy()

    metadata = {
        "tau": tau,
        "n": n,
        "sigma": sigma_val,
        "selected_centers": np.count_nonzero(np.abs(model_weights) > 1e-5),
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
    seed: int = 0,
) -> ExperimentResult:
    """
    Run control experiment with either the Adam-optimized RBF method or the EM-VP algorithm.

    Args:
        series: Time series data (noisy version)
        config: Control experiment configuration
        train_ratio: Ratio of data to use for training
        device: Device to run computations on

    Returns:
        ExperimentResult containing predictions and metadata
    """
    series = series.astype(np.float32)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tau = (
        config.embedding_tau
        if config.embedding_tau is not None
        else estimate_tau_for_series(series)
    )
    n = (
        config.n
        if config.n is not None
        else estimate_embedding_dimension_cao(
            series, tau=tau, max_m=config.max_embedding_dim
        )
    )

    X, d = make_lagged_matrix(series, n, tau)

    split_idx = int(train_ratio * len(X))
    X_train_np, X_test_np = X[:split_idx], X[split_idx:]
    d_train_np, d_test_np = d[:split_idx], d[split_idx:]

    sigma_val = (
        float(config.sigma)
        if config.sigma is not None
        else float(estimate_sigma_median(X_train_np))
    )

    X_train = torch.from_numpy(X_train_np).to(device)
    X_test = torch.from_numpy(X_test_np).to(device)
    d_train = torch.from_numpy(d_train_np).to(device)
    d_test = torch.from_numpy(d_test_np).to(device)

    if isinstance(config, ControlGDConfig):
        return _run_control_with_gradient_descent(
            config=config,
            X_train=X_train,
            X_test=X_test,
            d_train=d_train,
            d_test=d_test,
            tau=tau,
            n=n,
            sigma_val=sigma_val,
        )

    if isinstance(config, ControlEMVPConfig):
        return _run_control_with_emvp(
            config=config,
            X_train=X_train,
            X_test=X_test,
            d_train=d_train,
            d_test=d_test,
            tau=tau,
            n=n,
            sigma_val=sigma_val,
            base_device=device,
        )

    raise TypeError(f"Unsupported control configuration type: {type(config).__name__}")


def _run_control_with_gradient_descent(
    *,
    config: ControlGDConfig,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    d_train: torch.Tensor,
    d_test: torch.Tensor,
    tau: int,
    n: int,
    sigma_val: float,
) -> ExperimentResult:
    """Execute the legacy gradient-descent-based control method."""

    val_size = int(config.val_split * X_train.shape[0])
    train_size = X_train.shape[0] - val_size
    X_train_split = X_train[:train_size]
    d_train_split = d_train[:train_size]
    X_val_split = X_train[train_size:]
    d_val_split = d_train[train_size:]

    if X_train_split.shape[0] == 0:
        raise ValueError(
            "Not enough training samples after validation split for control GD."
        )

    m_ctrl = min(config.m, X_train_split.shape[0])

    perm = torch.randperm(X_train_split.shape[0], device=X_train.device)
    selected = perm[:m_ctrl]
    C_param = (
        X_train_split[selected].clone().detach().requires_grad_(True)
    )  # (m, n) - centers

    Theta_param = (
        0.01 * torch.randn(m_ctrl, X_train_split.shape[1], device=X_train.device)
    ).requires_grad_(
        True
    )  # (m, n) - projection weights

    # Add constant terms parameter
    Beta_param = (0.01 * torch.randn(m_ctrl, device=X_train.device)).requires_grad_(
        True
    )  # (m,) - constant terms for each center

    if config.sigma_global:
        log_sigma_param = torch.tensor(
            np.log(float(sigma_val) + 1e-6),
            dtype=X_train_split.dtype,
            device=X_train.device,
            requires_grad=config.train_sigma,
        )
    else:
        init_sigma = torch.full(
            (m_ctrl,),
            float(sigma_val),
            dtype=X_train_split.dtype,
            device=X_train.device,
        )
        log_sigma_param = torch.log(init_sigma + 1e-6)
        log_sigma_param.requires_grad_(config.train_sigma)

    params: List[torch.Tensor] = [C_param, Theta_param, Beta_param]
    if config.train_sigma:
        params.append(log_sigma_param)

    optim = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)

    def _phi_from_params(
        X_in: torch.Tensor, C: torch.Tensor, log_sigma: torch.Tensor
    ) -> torch.Tensor:
        x_sq = (X_in * X_in).sum(dim=1, keepdim=True)  # (l, 1)
        c_sq = (C * C).sum(dim=1)  # (m,)
        dist_sq = x_sq + c_sq.unsqueeze(0) - 2.0 * (X_in @ C.T)
        dist_sq = torch.clamp(dist_sq, min=0.0)

        if config.sigma_global:
            sigma = F.softplus(log_sigma) + 1e-6
        else:
            sigma = F.softplus(log_sigma).unsqueeze(0) + 1e-6

        if config.rbf == RadialBasisFunction.GAUSSIAN:
            denom = 2.0 * (sigma * sigma)
            phi = torch.exp(-dist_sq / denom)
        elif config.rbf == RadialBasisFunction.LAPLACIAN:
            dist = torch.sqrt(dist_sq + 1e-12)
            phi = torch.exp(-dist / (sigma + 1e-12))
        else:
            raise ValueError("Unknown RBF in config")

        return phi

    best_val_loss = float("inf")
    best_epoch = 0
    best_params = [p.clone() for p in params]

    for epoch in range(1, config.epochs + 1):
        phi_tr = _phi_from_params(X_train_split, C_param, log_sigma_param)
        proj_tr = X_train_split @ Theta_param.T
        const_tr = Beta_param.unsqueeze(0)  # (1, m)
        y_hat_tr_ctrl = (phi_tr * (proj_tr + const_tr)).sum(dim=1)
        loss = torch.mean((y_hat_tr_ctrl - d_train_split) ** 2)

        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            phi_val = _phi_from_params(X_val_split, C_param, log_sigma_param)
            proj_val = X_val_split @ Theta_param.T
            const_val = Beta_param.unsqueeze(0)  # (1, m)
            y_hat_val_ctrl = (phi_val * (proj_val + const_val)).sum(dim=1)
            val_loss = (
                torch.mean((y_hat_val_ctrl - d_val_split) ** 2)
                + config.ridge * (Theta_param**2).sum()
                + config.ridge * (Beta_param**2).sum()
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch
            best_params = [p.clone() for p in params]
        elif epoch - best_epoch >= config.patience:
            break

    for i, param in enumerate(params):
        param.data = best_params[i].data

    with torch.no_grad():
        phi_tr = _phi_from_params(X_train, C_param, log_sigma_param)
        proj_tr = X_train @ Theta_param.T
        const_tr = Beta_param.unsqueeze(0)  # (1, m)
        train_pred = (phi_tr * (proj_tr + const_tr)).sum(dim=1).cpu().numpy()

        phi_te = _phi_from_params(X_test, C_param, log_sigma_param)
        proj_te = X_test @ Theta_param.T
        const_te = Beta_param.unsqueeze(0)  # (1, m)
        test_pred = (phi_te * (proj_te + const_te)).sum(dim=1).cpu().numpy()

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


def _run_proposed_with_modified_emvp(
    *,
    config: ProposedModifiedEMVPConfig,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    d_train: torch.Tensor,
    d_test: torch.Tensor,
    tau: int,
    n: int,
    base_device: str,
) -> ExperimentResult:
    """Execute the Modified EM-VP-based proposed method."""

    effective_device = config.device or base_device
    dtype = config.dtype

    if X_train.shape[0] == 0:
        raise ValueError(
            "Not enough training samples for Modified EM-VP proposed method."
        )

    X_train_device = X_train.to(device=effective_device, dtype=dtype)
    X_test_device = X_test.to(device=effective_device, dtype=dtype)
    d_train_device = d_train.to(device=effective_device, dtype=dtype)

    def _sample_initial_centres() -> torch.Tensor:
        pool_size = max(1, int(config.centre_sampling_ratio * X_train_device.shape[0]))
        pool_indices = torch.randperm(X_train_device.shape[0], device=effective_device)[
            :pool_size
        ]
        if pool_indices.numel() > 1:
            shuffle = torch.randperm(pool_indices.numel(), device=effective_device)
            pool_indices = pool_indices[shuffle]
        if pool_indices.numel() >= config.num_components:
            return X_train_device[pool_indices[: config.num_components]].clone()
        else:
            # Pad with repeated samples if needed
            repeats = (
                config.num_components + pool_indices.numel() - 1
            ) // pool_indices.numel()
            indices_repeated = pool_indices.repeat(repeats)[: config.num_components]
            return X_train_device[indices_repeated].clone()

    best_log_likelihood = float("-inf")
    best_model = None
    best_diagnostics = None

    for restart in range(config.centre_restarts):
        centres_init = _sample_initial_centres()

        # Estimate initial widths based on pairwise distances
        if centres_init.shape[0] > 1:
            pairwise_dists = torch.cdist(centres_init, centres_init, p=2)
            # Use median of non-zero distances
            upper_tri_dists = pairwise_dists[
                torch.triu_indices(
                    centres_init.shape[0], centres_init.shape[0], offset=1
                )
            ]
            median_dist = (
                upper_tri_dists.median()
                if upper_tri_dists.numel() > 0
                else torch.tensor(1.0)
            )
            width_init_val = median_dist * config.init_width_scale
        else:
            width_init_val = torch.tensor(1.0, device=effective_device, dtype=dtype)

        widths_init = torch.full(
            (config.num_components,),
            width_init_val,
            device=effective_device,
            dtype=dtype,
        )

        # Initialize noise variance
        noise_base = d_train_device.var(unbiased=False).item()
        noise_init = torch.full(
            (config.num_components,),
            float(noise_base + config.min_variance),
            device=effective_device,
            dtype=dtype,
        )

        # Create the modified trainer config
        trainer_config = ModifiedEMVPConfig(
            num_components=config.num_components,
            max_iters=config.max_iters,
            tol_loglik=config.tol_loglik,
            tol_param=config.tol_param,
            min_variance=config.min_variance,
            ridge=config.ridge,
            init_responsibility_temp=config.init_responsibility_temp,
            init_width_scale=config.init_width_scale,
            device=effective_device,
            dtype=dtype,
            loglik_window=config.loglik_window,
            responsibility_floor=config.responsibility_floor,
            # TSVD parameters
            use_tsvd=config.use_tsvd,
            tsvd_epsilon=config.tsvd_epsilon,
            tsvd_alpha=config.tsvd_alpha,
            tsvd_delta=config.tsvd_delta,
            tsvd_beta=config.tsvd_beta,
        )

        trainer = ModifiedEMVPTrainer(trainer_config)
        model = trainer.fit(
            X_train_device,
            d_train_device,
            centres_init=centres_init,
            widths_init=widths_init,
            noise_init=noise_init,
        )

        if trainer.diagnostics.log_likelihood:
            final_loglik = trainer.diagnostics.log_likelihood[-1]
        else:
            final_loglik = float("-inf")

        if final_loglik > best_log_likelihood:
            best_log_likelihood = final_loglik
            best_model = model
            best_diagnostics = trainer.diagnostics

    if best_model is None or best_diagnostics is None:
        raise RuntimeError("Modified EM-VP training did not produce a valid model")

    train_pred = best_model.predict(X_train_device).detach().cpu().numpy()
    test_pred = best_model.predict(X_test_device).detach().cpu().numpy()

    metadata = {
        "tau": tau,
        "n": n,
        "num_components": config.num_components,
        "final_log_likelihood": best_log_likelihood,
        "optimizer_type": "ModifiedEMVP",
        "num_iterations": len(best_diagnostics.log_likelihood),
        "approach": "modified-emvp",
    }

    return ExperimentResult(
        train_predictions=train_pred,
        train_targets=d_train.detach().cpu().numpy(),
        test_predictions=test_pred,
        test_targets=d_test.detach().cpu().numpy(),
        method_name=f"PROPOSED-ModifiedEMVP-{config.num_components}components",
        metadata=metadata,
    )


def _run_control_with_emvp(
    *,
    config: ControlEMVPConfig,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    d_train: torch.Tensor,
    d_test: torch.Tensor,
    tau: int,
    n: int,
    sigma_val: float,
    base_device: str,
) -> ExperimentResult:
    """Execute the EM-VP-based control method."""

    effective_device = config.device or base_device
    dtype = config.dtype

    if X_train.shape[0] == 0:
        raise ValueError("Not enough training samples for EM-VP control method.")

    X_train_device = X_train.to(device=effective_device, dtype=dtype)
    X_test_device = X_test.to(device=effective_device, dtype=dtype)
    d_train_device = d_train.to(device=effective_device, dtype=dtype)

    def _sample_initial_centres() -> torch.Tensor:
        pool_size = max(1, int(config.centre_sampling_ratio * X_train_device.shape[0]))
        pool_indices = torch.randperm(X_train_device.shape[0], device=effective_device)[
            :pool_size
        ]
        if pool_indices.numel() > 1:
            shuffle = torch.randperm(pool_indices.numel(), device=effective_device)
            pool_indices = pool_indices[shuffle]
        if pool_indices.numel() >= config.num_components:
            chosen = pool_indices[: config.num_components]
        else:
            repeats = config.num_components - pool_indices.numel()
            extra_idx = pool_indices[
                torch.randint(
                    0, pool_indices.numel(), (repeats,), device=effective_device
                )
            ]
            chosen = torch.cat([pool_indices, extra_idx], dim=0)
        return X_train_device[chosen].clone()

    noise_base = torch.var(d_train_device, unbiased=False).item()
    widths_base = max(sigma_val * config.init_width_scale, 1e-6)

    best_model: Optional[EMVPModel] = None
    best_diagnostics: Optional[EMVPDiagnostics] = None
    best_log_likelihood = float("-inf")

    for restart in range(max(1, config.centre_restarts)):
        centres_init = _sample_initial_centres()
        widths_init = torch.full(
            (config.num_components,),
            float(widths_base),
            device=effective_device,
            dtype=dtype,
        )
        noise_init = torch.full(
            (config.num_components,),
            float(noise_base + config.min_variance),
            device=effective_device,
            dtype=dtype,
        )

        trainer_config = EMVPConfig(
            num_components=config.num_components,
            max_iters=config.max_iters,
            tol_loglik=config.tol_loglik,
            tol_param=config.tol_param,
            min_variance=config.min_variance,
            ridge=config.ridge,
            init_responsibility_temp=config.init_responsibility_temp,
            init_width_scale=config.init_width_scale,
            device=effective_device,
            dtype=dtype,
            loglik_window=config.loglik_window,
            responsibility_floor=config.responsibility_floor,
        )

        trainer = EMVPTrainer(trainer_config)
        model = trainer.fit(
            X_train_device,
            d_train_device,
            centres_init=centres_init,
            widths_init=widths_init,
            noise_init=noise_init,
        )

        if trainer.diagnostics.log_likelihood:
            final_loglik = trainer.diagnostics.log_likelihood[-1]
        else:
            final_loglik = float("-inf")

        if final_loglik > best_log_likelihood:
            best_log_likelihood = final_loglik
            best_model = model
            best_diagnostics = trainer.diagnostics

    if best_model is None or best_diagnostics is None:
        raise RuntimeError("EM-VP training did not produce a valid model")

    train_pred = best_model.predict(X_train_device).detach().cpu().numpy()
    test_pred = best_model.predict(X_test_device).detach().cpu().numpy()

    metadata = {
        "tau": tau,
        "n": n,
        "sigma": sigma_val,
        "num_components": config.num_components,
        "max_iters": config.max_iters,
        "best_log_likelihood": float(best_log_likelihood),
        "loglik_history": list(best_diagnostics.log_likelihood),
        "centre_restarts": config.centre_restarts,
        "device": effective_device,
    }

    return ExperimentResult(
        train_predictions=train_pred,
        test_predictions=test_pred,
        train_targets=d_train.cpu().numpy(),
        test_targets=d_test.cpu().numpy(),
        method_name="Control-EMVP",
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
    method_type: "proposed" for SVD/OLS methods, "control" for baseline methods (GD or EM-VP)
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


def create_default_control_config(**kwargs) -> ControlGDConfig:
    """Create a default gradient-descent control configuration with optional overrides."""
    defaults = {"m": 14, "epochs": 1000, "lr": 5e-2}
    defaults.update(kwargs)
    return ControlGDConfig(**defaults)


def run_comparison_experiments(
    series: np.ndarray,
    proposed_config: Union[ProposedMethodConfig, List[ProposedMethodConfig]],
    ctrl_config: Optional[Union[ControlConfig, List[ControlConfig]]] = None,
    train_ratio: float = 0.7,
    device: str = "cpu",
    run_parallel: bool = False,
    show_progress: bool = True,
    seed: int = 0,
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
        run_proposed_experiment(series, cfg, train_ratio, device, seed=seed),
    )[1]
    run_control_exp_and_update_progress = lambda cfg: (
        progress_bar.update(1),
        run_control_experiment(series, cfg, train_ratio, device, seed=seed),
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
