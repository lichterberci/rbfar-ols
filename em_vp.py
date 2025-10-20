"""Expectation-Maximization with Variational Priors for RBF-AR models.

This module implements a practical EM-VP style algorithm tailored for the
repository's Radial Basis Function Autoregressive (RBF-AR) workflow.  The
implementation focuses on small- to medium-sized datasets and keeps the code
torch-first to stay compatible with the rest of the project.

The algorithm treats the model as a mixture of local autoregressive experts
with radial basis gating.  Each expert is characterised by a centre c_i, a
width sigma_i, an autoregressive weight vector w_i, a noise variance
sigma_i^2, and a mixing prior pi_i.  The algorithm iteratively refines the
latent responsibilities Gamma (E-step) and then updates the model parameters
using their expected sufficient statistics (M-step / variational updates).

Key highlights:
        * Fully vectorised E-step using log-sum-exp for numerical stability.
        * Weighted least-squares closed-form update for the AR weights per expert.
        * Width, centre, and noise updates derived from the radial gating view.
        * Diagnostics tracking (log-likelihood and parameter deltas) to support
          experiment monitoring.

While inspired by the EM-VP derivations in the accompanying publication, this
implementation intentionally emphasises clarity and ease-of-use for the
repository's small dataset experiments.  It is therefore a suitable baseline
for further research extensions (e.g., richer priors, annealing schedules).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log, pi
from typing import Iterable, Optional, Tuple

import torch

Tensor = torch.Tensor


@dataclass
class EMVPConfig:
    """Hyperparameters controlling the EM-VP optimisation process."""

    num_components: int = 5
    max_iters: int = 200
    tol_loglik: float = 1e-5
    tol_param: float = 1e-4
    min_variance: float = 1e-6
    ridge: float = 1e-5
    init_responsibility_temp: float = 1.0
    init_width_scale: float = 0.5
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    loglik_window: int = 5
    responsibility_floor: float = 1e-8


@dataclass
class EMVPDiagnostics:
    """Bookkeep convergence metrics collected during training."""

    log_likelihood: list[float] = field(default_factory=list)
    max_param_update: list[float] = field(default_factory=list)
    responsibilities_entropy: list[float] = field(default_factory=list)


@dataclass
class EMVPState:
    """Container describing the learned model parameters.

    The model uses (n+1) functions φᵢ for i=0..n, where:
    - φ₀: constant function
    - φᵢ: coefficient function for feature i
    Each function has parameters [νᵢ,₀, νᵢ,₁, ..., νᵢ,ₘ] of shape (m+1,)
    """

    centres: Tensor  # RBF centers (m, n)
    widths: Tensor  # RBF widths (m,)
    function_params: Tensor  # Function parameters (num_components, n+1, m+1)
    mixing_logits: Tensor
    noise_vars: Tensor
    responsibilities: Tensor
    config: EMVPConfig

    @property
    def mixing_probs(self) -> Tensor:
        return torch.softmax(self.mixing_logits, dim=0)


class EMVPModel:
    """Convenience wrapper exposing prediction APIs for the fitted model."""

    def __init__(self, state: EMVPState):
        self._state = state

    @property
    def state(self) -> EMVPState:
        return self._state

    def predict(self, X: Tensor) -> Tensor:
        """Predict using the fitted parameters on a new design matrix."""

        X = _ensure_tensor(X, self._state.config)
        centres = self._state.centres
        widths = self._state.widths
        function_params = self._state.function_params  # (num_components, n+1, m+1)
        mixing = self._state.mixing_probs

        N, n = X.shape
        m = centres.shape[0]
        num_components = function_params.shape[0]

        # Compute RBF activations
        features = _gaussian_activations(X, centres, widths)  # (N, m)

        # Compute local predictions for each component (vectorized)
        local_means = _compute_local_means(
            X, features, function_params
        )  # (N, num_components)

        # Compute gating weights and final prediction
        log_mixing = torch.log(mixing + 1e-12)  # (K,)
        log_features = torch.log(features + 1e-12)  # (N, m)

        # For each component k: log(π_k) + Σⱼ log(Ψⱼ(X))
        log_features_sum = log_features.sum(dim=1)  # (N,) - sum over RBF centers
        log_gating = log_mixing.unsqueeze(0) + log_features_sum.unsqueeze(1)  # (N, K)
        weights_resp = torch.softmax(log_gating, dim=1)  # (N, K)

        predictions = (weights_resp * local_means).sum(dim=1)
        return predictions


class EMVPTrainer:
    """Trainer class orchestrating the EM-VP procedure."""

    def __init__(self, config: Optional[EMVPConfig] = None):
        self.config = config or EMVPConfig()
        self.diagnostics = EMVPDiagnostics()
        self._fitted_state: Optional[EMVPState] = None

    def fit(
        self,
        X: Tensor,
        y: Tensor,
        *,
        centres_init: Optional[Tensor] = None,
        widths_init: Optional[Tensor] = None,
        mixing_logits_init: Optional[Tensor] = None,
        noise_init: Optional[Tensor] = None,
    ) -> EMVPModel:
        """Run the EM iterations until convergence, returning a model."""

        cfg = self.config
        X = _ensure_tensor(X, cfg)
        y = _ensure_tensor(y, cfg).reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D tensor")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must share the sample dimension")

        N = X.shape[0]
        device = X.device

        centres = _init_centres(X, centres_init, cfg)
        widths = _init_widths(X, centres, widths_init, cfg)
        mixing_logits = (
            mixing_logits_init.clone().to(device=device, dtype=X.dtype)
            if mixing_logits_init is not None
            else torch.zeros(centres.shape[0], device=device, dtype=X.dtype)
        )
        noise_vars = (
            noise_init.clone().to(device=device, dtype=X.dtype)
            if noise_init is not None
            else torch.full(
                (centres.shape[0],),
                (y.var(unbiased=False) + cfg.min_variance).item(),
                device=device,
                dtype=X.dtype,
            )
        )

        function_params = _init_function_params(X, y, centres.shape[0], cfg)
        responsibilities = _init_responsibilities(N, centres.shape[0], cfg, device)

        for iteration in range(cfg.max_iters):
            features = _gaussian_activations(X, centres, widths)
            local_means = _compute_local_means(X, features, function_params)
            log_prob = _component_log_prob(
                y, local_means, features, mixing_logits, noise_vars
            )
            responsibilities = _normalize_log_prob(log_prob, cfg)

            centres_new, widths_new = _update_centres_widths(X, responsibilities, cfg)
            function_params_new = _update_function_params(
                X, y, responsibilities, features, function_params, cfg
            )
            noise_vars_new = _update_noise_new(
                X, y, responsibilities, features, function_params_new, cfg
            )
            mixing_logits_new = _update_mixing(responsibilities)

            log_likelihood = _log_likelihood_from_log_prob(log_prob)
            max_param_delta = _max_parameter_delta(
                (
                    centres,
                    widths,
                    function_params,
                    mixing_logits,
                    noise_vars,
                ),
                (
                    centres_new,
                    widths_new,
                    function_params_new,
                    mixing_logits_new,
                    noise_vars_new,
                ),
            )
            entropy = _responsibility_entropy(responsibilities)

            self.diagnostics.log_likelihood.append(log_likelihood.item())
            self.diagnostics.max_param_update.append(max_param_delta.item())
            self.diagnostics.responsibilities_entropy.append(entropy.item())

            centres = centres_new
            widths = widths_new
            function_params = function_params_new
            noise_vars = noise_vars_new
            mixing_logits = mixing_logits_new

            if _should_stop(self.diagnostics, cfg, iteration):
                break

        final_state = EMVPState(
            centres=centres,
            widths=widths,
            function_params=function_params,
            mixing_logits=mixing_logits,
            noise_vars=noise_vars,
            responsibilities=responsibilities,
            config=cfg,
        )
        self._fitted_state = final_state
        return EMVPModel(final_state)

    @property
    def fitted_state(self) -> EMVPState:
        if self._fitted_state is None:
            raise RuntimeError("Model not yet fitted. Call fit() first.")
        return self._fitted_state


# ---------------------------------------------------------------------------
# Helper functions


def _ensure_tensor(data: Tensor | Iterable[float], cfg: EMVPConfig) -> Tensor:
    device = torch.device(cfg.device)
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=cfg.dtype)
    return torch.as_tensor(data, device=device, dtype=cfg.dtype)


def _init_centres(X: Tensor, centres_init: Optional[Tensor], cfg: EMVPConfig) -> Tensor:
    if centres_init is not None:
        if centres_init.ndim != 2:
            raise ValueError("centres_init must be 2D")
        return centres_init.to(device=X.device, dtype=X.dtype)

    # Sample without replacement when possible
    N = X.shape[0]
    m = min(max(cfg.num_components, 1), N)
    perm = torch.randperm(N, device=X.device)
    return X[perm[:m]].clone()


def _init_widths(
    X: Tensor,
    centres: Tensor,
    widths_init: Optional[Tensor],
    cfg: EMVPConfig,
) -> Tensor:
    if widths_init is not None:
        widths = widths_init.to(device=X.device, dtype=X.dtype)
        if widths.ndim == 0:
            widths = widths.expand(centres.shape[0]).clone()
        return widths

    if centres.shape[0] == 1:
        width_value = (X.std(unbiased=False) * cfg.init_width_scale).item()
        return torch.full((1,), float(width_value), device=X.device, dtype=X.dtype)

    pairwise_dists = torch.cdist(centres, centres, p=2)
    upper = pairwise_dists[
        torch.triu_indices(centres.shape[0], centres.shape[0], offset=1)
    ]
    median_dist = upper.median()
    width = median_dist * cfg.init_width_scale
    width = torch.clamp(width, min=cfg.min_variance**0.5)
    return torch.full((centres.shape[0],), width, device=X.device, dtype=X.dtype)


def _init_function_params(
    X: Tensor,
    y: Tensor,
    m: int,
    cfg: EMVPConfig,
) -> Tensor:
    """Initialize function parameters for (n+1) functions with (m+1) parameters each.

    Returns tensor of shape (num_components, n+1, m+1) where:
    - First dimension: mixture component
    - Second dimension: function index (0=constant, 1..n=feature coefficients)
    - Third dimension: parameter index (0=global, 1..m=RBF centers)
    """
    device = X.device
    dtype = X.dtype
    n = X.shape[1]
    num_components = cfg.num_components

    # Initialize with small random values
    function_params = 0.01 * torch.randn(
        num_components, n + 1, m + 1, device=device, dtype=dtype
    )

    # Initialize global terms from simple linear regression
    ridge = cfg.ridge
    ones_col = torch.ones(X.shape[0], 1, device=device, dtype=dtype)
    X_aug = torch.cat([ones_col, X], dim=1)  # (N, n+1)
    XtX = X_aug.T @ X_aug + ridge * torch.eye(n + 1, device=device, dtype=dtype)
    XtY = X_aug.T @ y.squeeze()
    base_params = torch.linalg.lstsq(XtX, XtY).solution  # (n+1,)

    # Set global terms (index 0) for all components and functions
    for k in range(num_components):
        function_params[k, :, 0] = base_params  # Global terms

    return function_params


def _compute_local_means(
    X: Tensor, features: Tensor, function_params: Tensor
) -> Tensor:
    """Compute local means for each component using the function parameters.

    Args:
        X: Input tensor (N, n)
        features: RBF activations (N, m)
        function_params: Function parameters (num_components, n+1, m+1)

    Returns:
        local_means: Tensor of shape (N, num_components)
    """
    N, n = X.shape
    m = features.shape[1]
    num_components = function_params.shape[0]

    # Vectorized computation: avoid loops over components and features
    # Create extended features: [1, Ψ₁, Ψ₂, ..., Ψₘ] for all samples
    features_ext = torch.cat(
        [torch.ones(N, 1, device=X.device, dtype=X.dtype), features], dim=1
    )  # (N, m+1)

    # Compute all φᵢ(X) values at once for all components
    # φᵢ(X) = νᵢ,₀ + Σⱼ νᵢ,ⱼ Ψⱼ(X) = features_ext @ function_params[k, i, :]

    # Reshape for batch matrix multiplication
    # function_params: (K, n+1, m+1) -> (K*(n+1), m+1)
    params_flat = function_params.view(-1, m + 1)  # (K*(n+1), m+1)

    # Compute all φᵢᵏ(X) values: (N, K*(n+1))
    phi_all = features_ext @ params_flat.T  # (N, K*(n+1))

    # Reshape back to (N, K, n+1)
    phi_reshaped = phi_all.view(N, num_components, n + 1)  # (N, K, n+1)

    # Extract φ₀ (constant terms) and φᵢ (coefficient terms)
    phi_0 = phi_reshaped[:, :, 0]  # (N, K)
    phi_coeffs = phi_reshaped[:, :, 1:]  # (N, K, n)

    # Compute coefficient contributions: Σᵢ φᵢ(X) * Xᵢ
    # X: (N, n) -> (N, 1, n), phi_coeffs: (N, K, n)
    X_expanded = X.unsqueeze(1)  # (N, 1, n)
    coeff_contributions = (X_expanded * phi_coeffs).sum(dim=2)  # (N, K)

    # Final local means: φ₀ + Σᵢ φᵢ(X) * Xᵢ
    local_means = phi_0 + coeff_contributions  # (N, K)

    return local_means


def _init_responsibilities(
    N: int,
    m: int,
    cfg: EMVPConfig,
    device: torch.device,
) -> Tensor:
    logits = torch.zeros((N, m), device=device)
    logits = logits + torch.randn_like(logits) * cfg.init_responsibility_temp
    return torch.softmax(logits, dim=1)


def _gaussian_activations(X: Tensor, centres: Tensor, widths: Tensor) -> Tensor:
    """Compute Gaussian RBF activations efficiently using cdist."""
    # Use torch.cdist for efficient pairwise distance computation
    dists = torch.cdist(X, centres, p=2)  # (N, m)
    widths_sq = torch.clamp(widths, min=1e-12) ** 2
    return torch.exp(-(dists**2) / (2.0 * widths_sq.unsqueeze(0)))


def _log_normal_pdf(
    x: Tensor,
    mean: Tensor,
    var: Tensor,
) -> Tensor:
    var = torch.clamp(var, min=1e-12)
    return -0.5 * (torch.log(var) + ((x - mean) ** 2) / var + log(2.0 * pi))


def _component_log_prob(
    y: Tensor,
    local_means: Tensor,
    features: Tensor,
    mixing_logits: Tensor,
    noise_vars: Tensor,
) -> Tensor:
    log_mixing = torch.log_softmax(mixing_logits, dim=0)
    log_gauss = torch.log(features + 1e-12)
    log_conditional = _log_normal_pdf(y, local_means, noise_vars.unsqueeze(0))
    return log_mixing.unsqueeze(0) + log_gauss + log_conditional


def _normalize_log_prob(log_prob: Tensor, cfg: EMVPConfig) -> Tensor:
    log_prob = log_prob - log_prob.max(dim=1, keepdim=True).values
    responsibilities = torch.softmax(log_prob, dim=1)
    responsibilities = responsibilities.clamp_min(cfg.responsibility_floor)
    responsibilities = responsibilities / responsibilities.sum(dim=1, keepdim=True)
    return responsibilities


def _update_centres_widths(
    X: Tensor,
    responsibilities: Tensor,
    cfg: EMVPConfig,
) -> Tuple[Tensor, Tensor]:
    Nk = responsibilities.sum(dim=0).clamp_min(cfg.responsibility_floor)
    centres_new = (responsibilities.T @ X) / Nk.unsqueeze(1)

    diff = X.unsqueeze(1) - centres_new.unsqueeze(0)
    sq_norm = (diff * diff).sum(dim=2)
    widths_sq = (responsibilities * sq_norm).sum(dim=0) / (Nk * X.shape[1])
    widths_sq = widths_sq.clamp_min(cfg.min_variance)
    widths_new = torch.sqrt(widths_sq)
    return centres_new, widths_new


def _update_function_params(
    X: Tensor,
    y: Tensor,
    responsibilities: Tensor,
    features: Tensor,
    current_function_params: Tensor,
    cfg: EMVPConfig,
) -> Tensor:
    """Update function parameters using weighted least squares for each component.

    For each component k, solve for the (n+1)*(m+1) parameters that minimize:
    Σᵢ γᵢₖ (yᵢ - ŷᵢₖ)² where ŷᵢₖ = φ₀(Xᵢ) + Σⱼ φⱼ(Xᵢ) * Xᵢⱼ
    """
    N, n = X.shape
    m = features.shape[1]
    num_components = responsibilities.shape[1]
    ridge = cfg.ridge

    # Pre-compute design matrix once (this was the main bottleneck)
    ones_col = torch.ones(N, 1, device=X.device, dtype=X.dtype)
    features_ext = torch.cat([ones_col, features], dim=1)  # (N, m+1)

    # Efficiently build the full design matrix using broadcasting
    # φ₀ terms: [1, Ψ₁, Ψ₂, ..., Ψₘ]
    phi_0_design = features_ext  # (N, m+1)

    # φᵢ terms for i=1..n: [Xᵢ, Xᵢ*Ψ₁, Xᵢ*Ψ₂, ..., Xᵢ*Ψₘ]
    # Use broadcasting: X.unsqueeze(2) * features_ext.unsqueeze(1)
    X_ext = X.unsqueeze(2)  # (N, n, 1)
    features_broadcast = features_ext.unsqueeze(1)  # (N, 1, m+1)
    phi_coeff_designs = X_ext * features_broadcast  # (N, n, m+1)

    # Reshape to (N, n*(m+1))
    phi_coeff_designs_flat = phi_coeff_designs.view(N, n * (m + 1))

    # Full design matrix: [φ₀_terms, φ₁_terms, ..., φₙ_terms]
    full_design = torch.cat(
        [phi_0_design, phi_coeff_designs_flat], dim=1
    )  # (N, (n+1)*(m+1))

    function_params_new = torch.zeros_like(current_function_params)

    # Batch solve for all components using vectorized operations
    sqrt_responsibilities = torch.sqrt(
        responsibilities.clamp_min(cfg.responsibility_floor)
    )  # (N, K)

    # Vectorized weighted design matrix computation
    # sqrt_responsibilities: (N, K) -> (N, K, 1)
    sqrt_resp_expanded = sqrt_responsibilities.unsqueeze(2)  # (N, K, 1)
    # full_design: (N, D) -> (N, 1, D)
    design_expanded = full_design.unsqueeze(1)  # (N, 1, D)
    # Weighted design for all components: (N, K, D)
    weighted_designs = sqrt_resp_expanded * design_expanded  # (N, K, D)

    # Weighted targets for all components: (N, K)
    y_squeezed = y.squeeze()  # (N,)
    weighted_targets = sqrt_responsibilities * y_squeezed.unsqueeze(1)  # (N, K)

    D = full_design.shape[1]  # (n+1)*(m+1)
    ridge_eye = ridge * torch.eye(D, device=X.device, dtype=X.dtype)

    for k in range(num_components):
        # Extract weighted design and targets for component k
        weighted_design_k = weighted_designs[:, k, :]  # (N, D)
        weighted_y_k = weighted_targets[:, k]  # (N,)

        # Normal equations: (D^T D + λI) θ = D^T y
        AtA = weighted_design_k.T @ weighted_design_k + ridge_eye  # (D, D)
        Atb = weighted_design_k.T @ weighted_y_k  # (D,)

        # Solve and reshape
        params_flat = torch.linalg.lstsq(AtA, Atb).solution  # (D,)
        function_params_new[k] = params_flat.reshape(n + 1, m + 1)

    return function_params_new


def _update_noise(
    X: Tensor,
    y: Tensor,
    responsibilities: Tensor,
    weights: Tensor,
    constants: Tensor,
    cfg: EMVPConfig,
) -> Tensor:
    residuals = y - (X @ weights.T + constants.unsqueeze(0))
    sq_resid = residuals.pow(2)
    Nk = responsibilities.sum(dim=0).clamp_min(cfg.responsibility_floor)
    noise = (responsibilities * sq_resid).sum(dim=0) / Nk
    noise = noise.clamp_min(cfg.min_variance)
    return noise


def _update_noise_new(
    X: Tensor,
    y: Tensor,
    responsibilities: Tensor,
    features: Tensor,
    function_params: Tensor,
    cfg: EMVPConfig,
) -> Tensor:
    """Update noise variances using the new function parameters structure."""
    local_means = _compute_local_means(
        X, features, function_params
    )  # (N, num_components)
    residuals = y.squeeze().unsqueeze(1) - local_means  # (N, num_components)
    sq_resid = residuals.pow(2)
    Nk = responsibilities.sum(dim=0).clamp_min(cfg.responsibility_floor)
    noise = (responsibilities * sq_resid).sum(dim=0) / Nk
    noise = noise.clamp_min(cfg.min_variance)
    return noise


def _update_mixing(responsibilities: Tensor) -> Tensor:
    probs = responsibilities.mean(dim=0)
    return torch.log(probs + 1e-12)


def _log_likelihood_from_log_prob(log_prob: Tensor) -> Tensor:
    max_per_row = log_prob.max(dim=1, keepdim=True).values
    stabilised = log_prob - max_per_row
    return (max_per_row.squeeze(1) + torch.logsumexp(stabilised, dim=1)).sum()


def _max_parameter_delta(before, after) -> Tensor:
    deltas = [torch.max(torch.abs(b - a)) for b, a in zip(before, after)]
    return torch.stack(deltas).max()


def _responsibility_entropy(responsibilities: Tensor) -> Tensor:
    log_resp = torch.log(responsibilities + 1e-12)
    entropy = -(responsibilities * log_resp).sum(dim=1).mean()
    return entropy


def _should_stop(diagnostics: EMVPDiagnostics, cfg: EMVPConfig, iteration: int) -> bool:
    if iteration < 1:
        return False

    if diagnostics.max_param_update[-1] < cfg.tol_param:
        return True

    window = min(cfg.loglik_window, len(diagnostics.log_likelihood))
    if window >= 2:
        diffs = torch.tensor(diagnostics.log_likelihood[-window:], dtype=torch.float64)
        if (diffs[-1] - diffs[0]).abs() < cfg.tol_loglik:
            return True
    return False
