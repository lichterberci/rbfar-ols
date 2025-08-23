from typing import Tuple
import numpy as np
import torch


class OlsOptimizer:
    """
    Ordinary Least Squares (OLS) forward-selection optimizer.
    This class implements a greedy orthogonal forward-selection procedure to build a
    sparse linear approximation of a target vector d using columns (centres) of a
    design matrix P.
    """

    def __init__(self, rho: float = 0.01):
        """Initialize the OlsOptimizer.

        Args:
            rho (float, optional): The convergence threshold. Defaults to 0.01.
        """
        self._rho = rho

    # Disable lint warnings about non-PEP8 naming for the following method:
    # pylint: disable=invalid-name
    # For flake8/pep8-naming plugin:
    # noqa: N802,N806
    def optimize(
        self,
        P: torch.tensor | np.ndarray[np.floating],
        d: torch.tensor | np.ndarray[np.floating],
    ) -> (
        Tuple[torch.tensor, torch.tensor]
        | Tuple[np.ndarray[np.integer], np.ndarray[np.floating]]
    ):
        """
        This is the main function of the optimizer. It performs an OLS optimization and returns the optimal sparse linear combination.

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

        device = P.device if isinstance(P, torch.Tensor) else torch.device("cpu")
        float_dtype = P.dtype if isinstance(P, torch.Tensor) else np.float32

        if isinstance(P, np.ndarray):
            P = torch.from_numpy(P)
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(d, device=device, dtype=float_dtype)

        # centralization of the target vector
        d_mean = d.mean()
        d_centered = d - d_mean

        # Initialization

        residuals = d_centered.clone()  # initial residuals are the centralized d
        selected_centre_indices = torch.tensor((), dtype=torch.int32, device=device)
        remaining_centre_indices = torch.arange(m, dtype=torch.int32, device=device)
        W = torch.tensor((), dtype=float_dtype, device=device)

        # NOTE: the alpha values are stored as a single vector,
        # from which the upper-diagonal A matrix can be constructed
        alphas = torch.tensor((), dtype=float_dtype, device=device)

        # iterative centre selection
        for k in range(m):
            P_rem = P[remaining_centre_indices, :]  # candidates: shape (r, l)
            if k == 0:
                # no previous w_j, so w_i = p_i
                alphas_cand = torch.empty((), dtype=float_dtype, device=device)
                W_candidate = P_rem[:]
            else:
                # alpha_{i,j} = (p_i^T w_j) / ||w_j||^2
                numer = P_rem @ W.T  # shape (r, k_prev)
                denom = W.pow(2).sum(dim=1)  # shape (k_prev,)
                alphas_cand = numer / denom.unsqueeze(0)  # broadcast to (r, k_prev)
                # w_i = p_i - sum_j alpha_{i,j} w_j  -> matrix form: P_rem - alphas @ W
                W_candidate = P_rem - alphas_cand @ W

            # g_i = (w_i^T d^{(k - 1)}) / ||w_i||^2
            g = (W_candidate @ residuals) / W_candidate.pow(2).sum(dim=1)
            # err_i = g_i^2 * ||w_i||^2 / ||d^{(k - 1)}||^2
            d_norm_squared = residuals.pow(2).sum()

            err = g.pow(2) * W_candidate.pow(2).sum(dim=1) / d_norm_squared

            # select the centre with the smallest error
            min_err_index = torch.argmin(err, dim=0)
            selected_centre_index = remaining_centre_indices[min_err_index]
            selected_centre_indices = torch.cat(
                (selected_centre_indices, selected_centre_index.unsqueeze(0))
            )

            w = W_candidate[min_err_index].unsqueeze(0)
            W = torch.cat((W, w), dim=0)

            # update the residuals
            residuals -= w * g[min_err_index]

            # remove the selected centre from the remaining centres
            remaining_centre_indices = torch.cat(
                (
                    remaining_centre_indices[:min_err_index],
                    remaining_centre_indices[min_err_index + 1 :],
                )
            )

            # update the A matrix
            alphas = torch.cat((alphas, alphas_cand[min_err_index].unsqueeze(0)), dim=0)

            # check for stopping criteria
            if residuals.pow(2).sum() / d_norm_squared < self._rho:
                break

        # the number of selected centres
        M_s = W.size(0)

        # Construct the full upper-triangular matrix A from the alphas vector
        # alphas contains the upper-diagonal elements row-wise: (0,1), (0,2), ..., (0,M_s-1), (1,2), ..., (M_s-2, M_s-1)
        A = torch.zeros((M_s, M_s), dtype=float_dtype, device=device)

        # TODO: test that this fills in the A matrix correctly
        # Get upper-triangular indices (excluding diagonal)
        triu_i, triu_j = torch.triu_indices(M_s, M_s, offset=1)
        A[triu_i, triu_j] = alphas
        A.fill_diagonal_(1.0)  # Set main diagonal to ones

        H = W.T @ W  # Gram matrix of W

        H_inv = torch.linalg.pinv(H)

        g_hat = (H_inv @ W.T) @ d_centered

        theta_hat = torch.linalg.solve_triangular(
            A, g_hat.view(-1, 1), upper=True, unitriangular=True, left=True
        ).view(-1)

        if should_return_result_as_numpy:
            selected_centre_indices = selected_centre_indices.numpy()
            theta_hat = theta_hat.numpy()

        return selected_centre_indices, theta_hat
