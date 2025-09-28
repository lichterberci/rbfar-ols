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
    """Container describing the learned model parameters."""

    centres: Tensor
    widths: Tensor
    ar_weights: Tensor
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
        weights = self._state.ar_weights
        mixing = self._state.mixing_probs

        features = _gaussian_activations(X, centres, widths)
        local_means = X @ weights.T  # (N, M)
        log_gating = torch.log(mixing + 1e-12) + torch.log(features + 1e-12)
        weights_resp = torch.softmax(log_gating, dim=1)
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

        ar_weights = _init_ar_weights(X, y, centres.shape[0], cfg)
        responsibilities = _init_responsibilities(N, centres.shape[0], cfg, device)

        for iteration in range(cfg.max_iters):
            features = _gaussian_activations(X, centres, widths)
            local_means = X @ ar_weights.T
            log_prob = _component_log_prob(
                y, local_means, features, mixing_logits, noise_vars
            )
            responsibilities = _normalize_log_prob(log_prob, cfg)

            centres_new, widths_new = _update_centres_widths(X, responsibilities, cfg)
            ar_weights_new = _update_ar_weights(X, y, responsibilities, ar_weights, cfg)
            noise_vars_new = _update_noise(X, y, responsibilities, ar_weights_new, cfg)
            mixing_logits_new = _update_mixing(responsibilities)

            log_likelihood = _log_likelihood_from_log_prob(log_prob)
            max_param_delta = _max_parameter_delta(
                (
                    centres,
                    widths,
                    ar_weights,
                    mixing_logits,
                    noise_vars,
                ),
                (
                    centres_new,
                    widths_new,
                    ar_weights_new,
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
            ar_weights = ar_weights_new
            noise_vars = noise_vars_new
            mixing_logits = mixing_logits_new

            if _should_stop(self.diagnostics, cfg, iteration):
                break

        final_state = EMVPState(
            centres=centres,
            widths=widths,
            ar_weights=ar_weights,
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


def _init_ar_weights(
    X: Tensor,
    y: Tensor,
    m: int,
    cfg: EMVPConfig,
) -> Tensor:
    ridge = cfg.ridge
    XtX = X.T @ X + ridge * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    XtY = X.T @ y
    base_weights = torch.linalg.solve(XtX, XtY)
    return base_weights.T.repeat(m, 1)


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
    diff = X.unsqueeze(1) - centres.unsqueeze(0)
    sq_norm = (diff * diff).sum(dim=2)
    widths_sq = torch.clamp(widths, min=1e-12) ** 2
    return torch.exp(-sq_norm / (2.0 * widths_sq.unsqueeze(0)))


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


def _update_ar_weights(
    X: Tensor,
    y: Tensor,
    responsibilities: Tensor,
    current_weights: Tensor,
    cfg: EMVPConfig,
) -> Tensor:
    m = responsibilities.shape[1]
    P = X.shape[1]
    ridge = cfg.ridge
    weights_new = torch.zeros_like(current_weights)

    for k in range(m):
        gamma = responsibilities[:, k].unsqueeze(1)
        weighted_X = torch.sqrt(gamma) * X
        weighted_y = torch.sqrt(gamma) * y
        XtX = weighted_X.T @ weighted_X + ridge * torch.eye(
            P, device=X.device, dtype=X.dtype
        )
        XtY = weighted_X.T @ weighted_y
        weights_new[k] = torch.linalg.solve(XtX, XtY).squeeze(1)
    return weights_new


def _update_noise(
    X: Tensor,
    y: Tensor,
    responsibilities: Tensor,
    weights: Tensor,
    cfg: EMVPConfig,
) -> Tensor:
    residuals = y - X @ weights.T
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
