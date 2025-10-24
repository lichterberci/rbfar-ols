"""Modified EM-VP implementation for experimentation with novel approaches.

This module provides extensible versions of the EM-VP classes that inherit from
the base classes in em_vp.py, allowing for easy customization and experimentation
with novel modifications to the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from em_vp import EMVPConfig, EMVPDiagnostics, EMVPModel, EMVPState, EMVPTrainer

Tensor = torch.Tensor


@dataclass
class ModifiedEMVPConfig(EMVPConfig):
    """Configuration for the modified EM-VP algorithm.

    Extends the base EMVPConfig with additional parameters for novel modifications.
    """

    # TSVD parameters for function parameter learning
    use_tsvd: bool = True  # Use TSVD instead of L2 regularization
    tsvd_epsilon: float = 1e-2  # Convergence threshold for TSVD
    tsvd_alpha: float = 1e-5  # Regularization parameter for TSVD
    tsvd_delta: float = 1e-6  # Sparsity threshold for TSVD
    tsvd_beta: Optional[float] = None  # Max singular value clipping


class ModifiedEMVPState(EMVPState):
    """Extended state container for the modified EM-VP model.

    Can be used to store additional state variables for novel modifications.
    """

    pass


class ModifiedEMVPModel(EMVPModel):
    """Extended model class for the modified EM-VP algorithm.

    Inherits prediction capabilities from the base model and can be extended
    with custom prediction methods or post-processing.
    """

    def __init__(self, state: ModifiedEMVPState):
        super().__init__(state)

    # Add custom prediction methods here if needed
    # def predict_with_uncertainty(self, X: Tensor) -> Tuple[Tensor, Tensor]:
    #     """Predict with uncertainty estimates."""
    #     # Custom implementation here
    #     pass


class ModifiedEMVPTrainer(EMVPTrainer):
    """Extended trainer class for the modified EM-VP algorithm.

    Inherits all the base functionality and allows overriding of specific
    methods to implement novel modifications to the algorithm.
    """

    def __init__(self, config: Optional[ModifiedEMVPConfig] = None):
        super().__init__(config or ModifiedEMVPConfig())

    def _update_function_params(
        self,
        X: Tensor,
        y: Tensor,
        responsibilities: Tensor,
        features: Tensor,
        current_function_params: Tensor,
    ) -> Tensor:
        """Update function parameters using TSVD instead of L2 regularization."""

        if not (hasattr(self.cfg, "use_tsvd") and self.cfg.use_tsvd):
            # Fall back to base implementation if TSVD is disabled
            return super()._update_function_params(
                X, y, responsibilities, features, current_function_params
            )

        N, n = X.shape
        m = features.shape[1]
        num_components = responsibilities.shape[1]

        # Build the same design matrix as in the base method
        ones_col = torch.ones(N, 1, device=X.device, dtype=X.dtype)
        features_ext = torch.cat([ones_col, features], dim=1)  # (N, m+1)

        # φ₀ terms: [1, Ψ₁, Ψ₂, ..., Ψₘ]
        phi_0_design = features_ext  # (N, m+1)

        # φᵢ terms for i=1..n: [Xᵢ, Xᵢ*Ψ₁, Xᵢ*Ψ₂, ..., Xᵢ*Ψₘ]
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

        # Apply TSVD-based optimization for each component
        sqrt_responsibilities = torch.sqrt(
            responsibilities.clamp_min(self.cfg.responsibility_floor)
        )  # (N, K)

        y_squeezed = y.squeeze()  # (N,)

        for k in range(num_components):
            # Get responsibility weights for this component
            sqrt_resp_k = sqrt_responsibilities[:, k]  # (N,)

            # Weight the design matrix and targets
            weighted_design_k = sqrt_resp_k.unsqueeze(1) * full_design  # (N, D)
            weighted_y_k = sqrt_resp_k * y_squeezed  # (N,)

            # Apply TSVD optimization
            params_flat = self._tsvd_solve(weighted_design_k, weighted_y_k)
            function_params_new[k] = params_flat.reshape(n + 1, m + 1)

        return function_params_new

    def _tsvd_solve(self, P: Tensor, d: Tensor) -> Tensor:
        """Solve linear system using TSVD approach from SvdOptimizer."""

        # Get TSVD parameters from config
        epsilon = getattr(self.cfg, "tsvd_epsilon", 1e-2)
        alpha = getattr(self.cfg, "tsvd_alpha", 1e-5)
        delta = getattr(self.cfg, "tsvd_delta", 1e-6)
        beta = getattr(self.cfg, "tsvd_beta", None)

        # Perform reduced SVD: P = U @ diag(sigma) @ Vh
        U, sigma, Vh = torch.linalg.svd(P, full_matrices=False)

        # Clamp singular values if beta is specified
        if beta is not None:
            sigma = torch.clamp(sigma, max=beta)

        # Regularized inverse: sigma / (sigma^2 + alpha)
        regularized_sigma_inv = sigma / (sigma**2 + alpha)

        # Compute error threshold for component selection
        d_norm_squared = torch.dot(d, d)
        sigma_prime = sigma * regularized_sigma_inv  # in (0,1)

        # Select number of components by normalized residual threshold
        k_max = sigma.shape[0]
        m_selected = k_max
        err = d_norm_squared.clone()
        z_vals = []

        for k in range(k_max):
            z_k = torch.dot(U[:, k], d)
            z_vals.append(z_k)
            err = err + sigma_prime[k] * (sigma_prime[k] - 2.0) * (z_k**2)
            if (err / (d_norm_squared + 1e-12)) < epsilon:
                m_selected = k + 1
                break

        # Calculate the optimal weights using selected components
        if m_selected > 0:
            z_selected = torch.stack(z_vals[:m_selected])
            nu_hat = Vh[:m_selected, :].transpose(0, 1) @ (
                regularized_sigma_inv[:m_selected] * z_selected
            )
        else:
            nu_hat = torch.zeros(P.shape[1], dtype=P.dtype, device=P.device)

        # Apply sparsity threshold
        abs_nu = torch.abs(nu_hat)
        sparse_mask = abs_nu > delta

        # Create sparse solution
        sparse_nu = torch.zeros_like(nu_hat)
        if sparse_mask.any():
            sparse_nu[sparse_mask] = nu_hat[sparse_mask]
        else:
            # If no parameters pass threshold, keep the full solution
            sparse_nu = nu_hat

        return sparse_nu

    # Example of overriding a method to implement custom behavior
    # def _update_centres_widths(self, X: Tensor, responsibilities: Tensor) -> Tuple[Tensor, Tensor]:
    #     """Custom centre and width update with novel modifications."""
    #     # Call the base method first
    #     centres_new, widths_new = super()._update_centres_widths(X, responsibilities)
    #
    #     # Apply your custom modifications here
    #     # For example, adaptive width adjustment:
    #     # if hasattr(self.cfg, 'adaptive_widths') and self.cfg.adaptive_widths:
    #     #     widths_new = self._apply_adaptive_width_adjustment(widths_new, centres_new, X)
    #
    #     return centres_new, widths_new

    # def _apply_adaptive_width_adjustment(self, widths: Tensor, centres: Tensor, X: Tensor) -> Tensor:
    #     """Apply adaptive width adjustment based on local data density."""
    #     # Custom implementation here
    #     return widths

    # def _update_function_params(self, X: Tensor, y: Tensor, responsibilities: Tensor,
    #                            features: Tensor, current_function_params: Tensor) -> Tensor:
    #     """Custom function parameter update with regularization."""
    #     # Call base method
    #     params_new = super()._update_function_params(X, y, responsibilities, features, current_function_params)
    #
    #     # Apply custom regularization if configured
    #     # if hasattr(self.cfg, 'custom_regularization') and self.cfg.custom_regularization > 0:
    #     #     params_new = self._apply_custom_regularization(params_new)
    #
    #     return params_new

    # def _apply_custom_regularization(self, params: Tensor) -> Tensor:
    #     """Apply custom regularization to function parameters."""
    #     # Custom implementation here
    #     return params

    def fit(
        self,
        X: Tensor,
        y: Tensor,
        *,
        centres_init: Optional[Tensor] = None,
        widths_init: Optional[Tensor] = None,
        mixing_logits_init: Optional[Tensor] = None,
        noise_init: Optional[Tensor] = None,
    ) -> ModifiedEMVPModel:
        """Run the modified EM iterations until convergence."""

        # For now, just call the base implementation and wrap the result
        base_model = super().fit(
            X,
            y,
            centres_init=centres_init,
            widths_init=widths_init,
            mixing_logits_init=mixing_logits_init,
            noise_init=noise_init,
        )

        # Create modified state from base state
        modified_state = ModifiedEMVPState(
            centres=base_model.state.centres,
            widths=base_model.state.widths,
            function_params=base_model.state.function_params,
            mixing_logits=base_model.state.mixing_logits,
            noise_vars=base_model.state.noise_vars,
            responsibilities=base_model.state.responsibilities,
            config=self.config,
        )

        return ModifiedEMVPModel(modified_state)


# Convenience alias for backwards compatibility
ModifiedEMVPDiagnostics = EMVPDiagnostics
