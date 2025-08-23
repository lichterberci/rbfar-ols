"""This module implements an SVD-based optimizer for calculating the optimal sparse linear combination."""

from typing import Tuple

import numpy as np
import torch


class SvdOptimizer:
    """
    This class performs SVD-based optimization for OLS regression.
    For further details, please refer to the documentation.
    """

    def __init__(self, epsilon: float = 1e-2, alpha: float = 1e-5, delta: float = 1e-6):
        """Initialize the SvdOlsOptimizer.

        Args:
            epsilon (float, optional): The convergence threshold. Defaults to 1e-2.
            alpha (float, optional): The regularization parameter. Defaults to 1e-5.
            delta (float, optional): The sparsity parameter. Defaults to 1e-6.
        """

        self._epsilon = epsilon
        self._alpha = alpha
        self._delta = delta

    def optimize(
        self,
        P: torch.tensor | np.ndarray[np.floating],
        d: torch.tensor | np.ndarray[np.floating],
    ) -> (
        Tuple[torch.tensor, torch.tensor]
        | Tuple[np.ndarray[np.integer], np.ndarray[np.floating]]
    ):
        """
        This is the main function of the optimizer. It performs an SVD-decomposition on P and then finds the optimal sparse linear combination and returns it in the form of a tensor or a numpy array.

        Args:
            P (torch.tensor | np.ndarray[np.floating]): The design matrix of the input.
            d (torch.tensor | np.ndarray[np.floating]): The target vector.

        Returns:
            Tuple[torch.tensor, torch.tensor] | Tuple[np.ndarray[np.integer], np.ndarray[np.floating]]: The indices of the selected centres and the corresponding weights.
        """
        if not (
            isinstance(P, (torch.Tensor, np.ndarray))
            and isinstance(d, (torch.Tensor, np.ndarray))
        ):
            raise TypeError("P and d must be either torch.Tensor or np.ndarray")

        should_return_result_as_numpy = isinstance(P, np.ndarray) and isinstance(
            d, np.ndarray
        )

        if P.ndim != 2 or d.ndim != 1:
            raise ValueError("Incompatible shapes: P must be 2D and d must be 1D.")

        l, m = P.shape

        if l != d.shape[0]:
            raise ValueError(
                "Incompatible shapes: P and d must have the same number of rows."
            )

        if isinstance(P, np.ndarray):
            P = torch.from_numpy(P)
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(d)

        # Perform SVD
        # NOTE: sigma is only the diagonal vector of the Sigma matrix
        Q, sigma, VT = torch.svd(P, some=True, compute_uv=True)

        V = VT.T

        # reg_sigma_inv = (Sigma^T @ Sigma + alpha * I)^-1 @ Sigma^T
        regularized_sigma_inv = sigma / (sigma**2 + self._alpha)

        # err_0 = ||d||_2^2
        # --> in the maths notes, it is indexed from k=1,
        # but in the implementation, we index k from 0
        d_norm_squared = d.T @ d

        err = d_norm_squared

        sigma_prime = sigma * regularized_sigma_inv

        m_selected = None

        for k in range(m):
            err_k = (
                err
                + sigma_prime[k]
                * (sigma_prime[k] - 2)
                * (Q[:, k].T @ d) ** 2
                / d_norm_squared
            )

            if err_k < self._epsilon:
                m_selected = k + 1
                break
        else:
            m_selected = m

        # Calculate the optimal weights (for all possible centroids)
        nu_hat = V[:m_selected, :] @ (
            regularized_sigma_inv[:m_selected] * (Q[:, :m_selected].T @ d)
        )

        selected_indices = torch.nonzero(torch.abs(nu_hat) > self._delta).squeeze()

        return (
            (selected_indices, nu_hat[selected_indices])
            if not should_return_result_as_numpy
            else (selected_indices.numpy(), nu_hat[selected_indices].numpy())
        )
