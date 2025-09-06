from typing import Tuple
import numpy as np
import torch

try:
    from optimizer import Optimizer
except ModuleNotFoundError:  # when loaded via importlib by path in tests
    import importlib.util
    from pathlib import Path

    _root = Path(__file__).resolve().parent
    _opt_path = _root / "optimizer.py"
    _spec = importlib.util.spec_from_file_location("optimizer", str(_opt_path))
    _mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    assert _spec and _spec.loader
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    Optimizer = _mod.Optimizer


class OlsOptimizer(Optimizer):
    """
    Ordinary Least Squares (OLS) forward-selection optimizer.

    Greedy orthogonal forward-selection to build a sparse linear approximation of
    a target vector d using columns (centres) of a design matrix P.

    Optimizations:
    - Avoid per-iteration tensor concatenations; keep lists and stack once at the end.
    - Compute candidate scores via cross-products (P^T r and P^T w_j) instead of
      materializing per-iteration candidate W matrices.
    - Keep the upper-triangular A in packed form during iteration; build dense once.
    """

    def __init__(self, rho: float = 0.01, epsilon: float = 1e-8):
        """Initialize the OlsOptimizer.

        Args:
            rho: Normalized residual energy threshold to stop (smaller means more features).
            epsilon: Small value to stabilize divisions and inner products.
        """
        self._rho = float(rho)
        self._epsilon = float(epsilon)

    # Disable lint warnings about non-PEP8 naming for the following method:
    # pylint: disable=invalid-name
    # For flake8/pep8-naming plugin:
    # noqa: N802,N806
    def optimize(
        self,
        P: torch.Tensor | np.ndarray,
        d: torch.Tensor | np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[np.ndarray, np.ndarray]:
        """
        Run greedy Orthogonal Least Squares (OLS) forward selection to build a sparse
        linear model of the target: d ≈ P[:, S] @ theta.

        Parameters
        - P (torch.Tensor | numpy.ndarray):
            Design/activation matrix with shape (l, m), where l is the number of samples
            and m is the number of candidate centres/features. Must be 2D and real-valued.
        - d (torch.Tensor | numpy.ndarray):
            Target vector with shape (l,). Must be 1D and real-valued.

        Behavior
        - If P and d are NumPy arrays, they are converted to torch.Tensors internally.
        - The algorithm orthogonalizes selected columns via modified Gram–Schmidt,
            maintains an orthogonal basis W, and selects the next feature that maximizes
            (w_i^T r)^2 / (||w_i||^2 · ||r||^2), where r is the current residual.
        - Iteration stops when the normalized residual energy ||r||^2 / ||d||^2 drops
            below rho (set in the constructor), or when all candidates are exhausted.

        Returns
        - (selected_indices, theta_hat):
            - selected_indices: 1D int indices (length k) of the selected columns of P.
            - theta_hat: 1D floats (length k) solving the triangular system derived from
                the packed A matrix (upper unit-triangular) and g = H^{-1} W^T d.
            The return types mirror the input types: if inputs were torch.Tensors, both
            outputs are torch.Tensors; if inputs were NumPy arrays, both outputs are
            numpy.ndarrays on CPU.

        Raises
        - TypeError: if P or d are not torch.Tensor or numpy.ndarray.
        - ValueError: if P is not 2D, d is not 1D, or their first dimensions differ.

        Notes
        - Typical complexity is ~O(l · m · k) time and O(l · k + m · k) extra memory,
            where k is the number of selected features (k ≤ m).
        - This method runs under torch.no_grad() and does not track gradients.
        """
        # Validate inputs and unify types
        l, m = self._validate_inputs(P, d)
        should_return_result_as_numpy = self._should_return_numpy(P, d)
        P_t, d_t, device, float_dtype = self._to_torch(P, d)

        # Centralize target (same behaviour as before)
        d_centered = d_t - d_t.mean()

        eps = self._epsilon
        with torch.no_grad():
            # Residual and its norms
            residuals = d_centered.clone()  # (l,)
            r_norm_sqrd = torch.dot(residuals, residuals) + eps
            d0_norm_sqrd = r_norm_sqrd.clone()

            # Candidate bookkeeping
            remaining_idx = torch.arange(m, dtype=torch.long, device=device)
            selected_idx_list: list[int] = []

            # Precompute column norms ||p_i||^2 and pr = P^T r
            pnorm2_all = P_t.pow(2).sum(dim=0)  # (m,)
            p_norm_sqrd_rem = pnorm2_all.clone()  # aligned with remaining_idx
            pr = (P_t.transpose(0, 1) @ residuals).clone()  # (m,)
            pr = pr[remaining_idx]

            # Orthogonal basis W as a list of columns, plus vectorized caches
            W_cols: list[torch.Tensor] = []  # each (l,)
            # Preallocate up to m; only the first k entries are valid during iteration
            w_norm_sqrd_vec = torch.empty(m, dtype=float_dtype, device=device)
            # C_mat holds P_rem^T w_j for all selected j stacked row-wise; columns align with remaining_idx
            C_mat = torch.empty((0, 0), dtype=float_dtype, device=device)  # (k, r)
            A_cols: list[torch.Tensor] = (
                []
            )  # column-wise off-diagonals; k-th has shape (k,)

            # Iterate selections
            for k in range(m):
                r = remaining_idx.shape[0]
                if r == 0:
                    break

                # For all remaining candidates, compute denom and numer (vectorized)
                if k == 0:
                    denom = p_norm_sqrd_rem.clamp_min(eps)
                else:
                    # denom = ||p_i||^2 - sum_j (C[j,i]^2 / ||w_j||^2)
                    denom = p_norm_sqrd_rem - (
                        C_mat.pow(2) / (w_norm_sqrd_vec[:k].unsqueeze(1) + eps)
                    ).sum(dim=0)
                    denom = denom.clamp_min(eps)
                numer = pr  # because residuals ⟂ span{w_0..w_{k-1}}

                # Selection criterion: maximize (w_i^T r)^2 / (||w_i||^2 * ||r||^2)
                err = numer.pow(2) / (denom * r_norm_sqrd)
                max_pos = torch.argmax(err)
                sel_col = remaining_idx[max_pos].item()
                selected_idx_list.append(sel_col)

                # Build new orthogonal vector via modified Gram–Schmidt (vectorized)
                s = P_t[:, sel_col].clone()  # (l,)
                if k == 0:
                    w_new = s
                    alpha_new = None
                else:
                    # alpha_new = C_mat[:, pos] / ||w||^2
                    alpha_new = C_mat[:, max_pos] / (w_norm_sqrd_vec[:k] + eps)  # (k,)
                    # Stack selected W once this iter
                    W_block = torch.stack(W_cols, dim=1)  # (l, k)
                    w_new = s - W_block @ alpha_new  # (l,)

                # Stats for new vector and residual update
                w_norm_sqrd_new = torch.dot(w_new, w_new) + eps
                wr_new = torch.dot(w_new, residuals)
                g_new = wr_new / w_norm_sqrd_new

                residuals = residuals - w_new * g_new
                r_norm_sqrd = torch.dot(residuals, residuals) + eps

                # Update pr and compute C_new over remaining (including selected),
                # then drop selected
                P_rem = P_t[:, remaining_idx]
                C_new_full = P_rem.transpose(0, 1) @ w_new  # (r,)
                pr = pr - C_new_full * g_new

                # Drop selected position from remaining-aligned vectors
                mask = torch.ones(r, dtype=torch.bool, device=device)
                mask[max_pos] = False
                pr = pr[mask]
                p_norm_sqrd_rem = p_norm_sqrd_rem[mask]
                if k > 0:
                    C_mat = C_mat[:, mask]
                C_new = C_new_full[mask]
                remaining_idx = remaining_idx[mask]

                # Append to structures
                W_cols.append(w_new)
                # Store in the preallocated vector at position k
                w_norm_sqrd_vec[k] = w_norm_sqrd_new

                # Append new row to C_mat (rows correspond to selected w_j);
                # C_new aligns with updated remaining (r-1)
                C_new_row = C_new.unsqueeze(0)  # (1, r-1)
                if k == 0:
                    C_mat = C_new_row
                else:
                    C_mat = torch.cat((C_mat, C_new_row), dim=0)

                if alpha_new is not None:
                    A_cols.append(alpha_new)

                # Stopping by normalized residual energy
                if (r_norm_sqrd / d0_norm_sqrd) < self._rho:
                    break

            M_s = len(W_cols)
            if M_s == 0:
                if should_return_result_as_numpy:
                    np_float = (
                        np.float64 if float_dtype == torch.float64 else np.float32
                    )
                    return np.asarray([], dtype=np.int64), np.asarray(
                        [], dtype=np_float
                    )
                return (
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=float_dtype, device=device),
                )

            # Build dense A (upper, unit diag) from packed columns
            A = torch.zeros((M_s, M_s), dtype=float_dtype, device=device)
            A.fill_diagonal_(1.0)
            for k in range(1, M_s):
                A[:k, k] = A_cols[k - 1]

            # Compute g_hat = H^{-1} W^T d_centered with H diagonal of ||w_j||^2
            W_materialized = torch.stack(W_cols, dim=1)  # (l, M_s)
            H_diag = w_norm_sqrd_vec[:M_s]  # (M_s,)
            g_hat = (W_materialized.transpose(0, 1) @ d_centered) / H_diag  # (M_s,)

            # Solve A theta = g_hat for theta (A upper, unit diag) via back-substitution
            theta_hat = g_hat.clone()
            for col in range(M_s - 1, -1, -1):
                # A has ones on the diagonal: no division
                if col > 0:
                    theta_hat[:col] -= A[:col, col] * theta_hat[col]

            selected_centre_indices = torch.tensor(
                selected_idx_list, dtype=torch.long, device=device
            )

        if should_return_result_as_numpy:
            return selected_centre_indices.cpu().numpy(), theta_hat.cpu().numpy()
        return selected_centre_indices, theta_hat
