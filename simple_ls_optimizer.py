"""This module implements an SVD-based optimizer for calculating the optimal sparse linear combination."""

from typing import Tuple

import numpy as np
import torch

from optimizer import Optimizer


class SimpleLSOptimizer(Optimizer):
    """
    This class performs SVD-based optimization for OLS regression.
    For further details, please refer to the documentation.
    """

    def __init__(
        self,
        alpha: float = 1e-5,
        delta: float = 1e-6,
        regularitation_type: str = "l2",
    ):
        """Initialize the SvdOlsOptimizer.

        Args:
            alpha (float, optional): The regularization parameter. Defaults to 1e-5.
            delta (float, optional): The sparsity parameter. Defaults to 1e-6.
            regularitation_type (str, optional): The type of regularization to use. Currently only 'l2' is supported. Defaults to 'l2'.
        """

        self._alpha = alpha
        self._delta = delta
        if regularitation_type not in ["l0", "l1", "l2"]:
            raise ValueError("regularitation_type must be one of 'l0', 'l1', or 'l2'")
        self._regularitation_type = regularitation_type

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

        # !!! L0 ~ iteratively reweighed L1 (preferred this way) !!!

        # Augmented least squares for regularization
        sqrt_alpha = torch.sqrt(torch.tensor(self._alpha))
        P_aug = torch.cat([P, sqrt_alpha * torch.eye(P.shape[1])], dim=0)
        d_aug = torch.cat([d, torch.zeros(P.shape[1])], dim=0)

        nu_hat = torch.linalg.lstsq(P_aug, d_aug).solution

        # Apply sparsity threshold to select significant parameters
        abs_w = torch.abs(nu_hat)
        selected_indices = torch.nonzero(abs_w > self._delta, as_tuple=True)[0]

        # If no parameters pass the threshold, select all (fallback)
        if len(selected_indices) == 0:
            selected_indices = torch.arange(P.shape[1], device=P.device)

        return (
            (selected_indices, nu_hat[selected_indices])
            if not should_return_result_as_numpy
            else (
                selected_indices.cpu().numpy(),
                nu_hat[selected_indices].cpu().numpy(),
            )
        )
