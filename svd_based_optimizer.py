"""This module implements an SVD-based optimizer for calculating the optimal sparse linear combination."""

from typing import Tuple

import numpy as np
import torch

try:
    from optimizer import Optimizer
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    _root = Path(__file__).resolve().parent
    _opt_path = _root / "optimizer.py"
    _spec = importlib.util.spec_from_file_location("optimizer", str(_opt_path))
    _mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    assert _spec and _spec.loader
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    Optimizer = _mod.Optimizer


class SvdOptimizer(Optimizer):
    """
    This class performs SVD-based optimization for OLS regression.
    For further details, please refer to the documentation.
    """

    def __init__(
        self,
        epsilon: float = 1e-2,
        alpha: float = 1e-5,
        delta: float = 1e-6,
        beta: float = None,
    ):
        """Initialize the SvdOlsOptimizer.

        Args:
            epsilon (float, optional): The convergence threshold. Defaults to 1e-2.
            alpha (float, optional): The regularization parameter. Defaults to 1e-5.
            delta (float, optional): The sparsity parameter. Defaults to 1e-6.
            beta (float, optional): The maximum value of any singular value. Defaults to None.
        """

        self._epsilon = epsilon
        self._alpha = alpha
        self._delta = delta
        self._beta = beta

    def optimize(
        self,
        P: torch.Tensor | np.ndarray,
        d: torch.Tensor | np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[np.ndarray, np.ndarray]:
        """
        This is the main function of the optimizer. It performs an SVD-decomposition on P and then finds the optimal sparse linear combination and returns it in the form of a tensor or a numpy array.

        Args:
            P (torch.Tensor | np.ndarray): The design matrix of the input.
            d (torch.Tensor | np.ndarray): The target vector.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] | Tuple[np.ndarray, np.ndarray]: The indices of the selected centres and the corresponding weights.
        """
        # Validate inputs and unify types
        _, _ = self._validate_inputs(P, d)
        should_return_result_as_numpy = self._should_return_numpy(P, d)
        P, d, _, _ = self._to_torch(P, d)

        # Perform reduced SVD: P = U @ diag(sigma) @ Vh
        # Shapes: U (l, k), sigma (k,), Vh (k, m), where k = min(l, m)
        U, sigma, Vh = torch.linalg.svd(P, full_matrices=False)

        if self._beta is not None:
            sigma = torch.clamp(sigma, max=self._beta)

        # reg_sigma_inv = (Sigma^T @ Sigma + alpha * I)^-1 @ Sigma^T
        regularized_sigma_inv = sigma / (sigma**2 + self._alpha)

        # err_0 = ||d||_2^2 (normalized stopping uses ratio err/||d||^2)
        d_norm_squared = torch.dot(d, d)

        sigma_prime = (
            sigma * regularized_sigma_inv
        )  # = sigma^2 / (sigma^2 + alpha) in (0,1)

        # Select number of components by normalized residual threshold; compute z_k lazily
        k_max = sigma.shape[0]
        m_selected = k_max
        err = d_norm_squared.clone()
        z_vals: list[torch.Tensor] = []
        for k in range(k_max):
            z_k = torch.dot(U[:, k], d)
            z_vals.append(z_k)
            err = err + sigma_prime[k] * (sigma_prime[k] - 2.0) * (z_k**2)
            if (err / (d_norm_squared + 1e-12)) < self._epsilon:
                m_selected = k + 1
                break

        # Calculate the optimal weights (for all possible centroids)
        # Truncated regularized solution: x_hat = V @ diag(reg_sigma_inv) @ U^T d
        # Build z for selected components only
        if m_selected > 0:
            z_selected = torch.stack(z_vals[:m_selected])
        else:
            z_selected = torch.empty(0, dtype=P.dtype, device=P.device)

        nu_hat = Vh[:m_selected, :].transpose(0, 1) @ (
            regularized_sigma_inv[:m_selected] * z_selected
        )

        # Robust 1D indices with sparsity control
        # abs_w = torch.abs(nu_hat)
        # selected_indices = torch.nonzero(abs_w > self._delta, as_tuple=True)[0]

        selected_indices = torch.arange(P.shape[1])

        return (
            (selected_indices, nu_hat[selected_indices])
            if not should_return_result_as_numpy
            else (
                selected_indices.cpu().numpy(),
                nu_hat[selected_indices].cpu().numpy(),
            )
        )
