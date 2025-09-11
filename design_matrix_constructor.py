from typing import Optional
import torch
import numpy as np
from enum import StrEnum, auto


class RadialBasisFunction(StrEnum):
    """Enum for different types of radial basis functions."""

    GAUSSIAN = auto()
    LAPLACIAN = auto()


def estimate_sigma_median(
    X: torch.Tensor | np.ndarray,
    candidate_indices: Optional[torch.Tensor | np.ndarray] = None,
) -> torch.Tensor:
    """Estimate RBF bandwidth sigma via the median heuristic.

    Uses pairwise L2 distances between all rows in X (as query points) and
    selected centres taken from rows of X by candidate_indices.

    Args:
        X: Array/tensor of shape (l, n).
        candidate_indices: Optional 1D indices selecting centres among rows of X.

    Returns:
        sigma as a torch scalar on the same device/dtype policy as other fns.
    """
    if not isinstance(X, (torch.Tensor, np.ndarray)):
        raise TypeError("X must be a torch.Tensor or numpy.ndarray")

    # Unify to torch (CPU for numpy inputs), pick float32/64 consistently
    if isinstance(X, torch.Tensor):
        device = X.device
        dtype = X.dtype if X.dtype.is_floating_point else torch.get_default_dtype()
        X_t = X.to(device=device, dtype=dtype)
    else:
        device = torch.device("cpu")
        dtype = torch.float64 if X.dtype == np.float64 else torch.float32
        X_t = torch.from_numpy(X).to(device=device, dtype=dtype)

    l = X_t.shape[0]
    if candidate_indices is None:
        centres_idx_t = torch.arange(l, device=device, dtype=torch.long)
    else:
        if isinstance(candidate_indices, np.ndarray):
            centres_idx_t = torch.from_numpy(candidate_indices).to(device=device)
        else:
            centres_idx_t = candidate_indices.to(device=device)
        if centres_idx_t.ndim != 1:
            raise ValueError("candidate_indices must be a 1D array of indices")
        centres_idx_t = centres_idx_t.to(dtype=torch.long)
        if centres_idx_t.numel() == 0:
            raise ValueError("candidate_indices must be non-empty")
        if centres_idx_t.min() < 0 or centres_idx_t.max() >= l:
            raise IndexError("candidate_indices out of bounds for rows of X")

    centres = X_t[centres_idx_t]
    x_sq = (X_t * X_t).sum(dim=1, keepdim=True)
    centres_sq = (centres * centres).sum(dim=1)
    dist_sq = (x_sq + centres_sq.unsqueeze(0)) - 2.0 * (X_t @ centres.transpose(0, 1))
    dist_sq = torch.clamp(dist_sq, min=0.0)
    nonzero_dist_sq = dist_sq[dist_sq > 0]

    if nonzero_dist_sq.numel() == 0:
        return torch.tensor(1.0, device=device, dtype=dtype)
    sigma_t = torch.median(torch.sqrt(nonzero_dist_sq + 1e-12))
    if not torch.isfinite(sigma_t):
        sigma_t = torch.tensor(1.0, device=device, dtype=dtype)
    return torch.clamp(sigma_t, min=torch.finfo(dtype).eps)


def construct_design_matrix_with_no_pretraining(
    X: torch.Tensor | np.ndarray,
    d: torch.Tensor | np.ndarray,
    centres: Optional[torch.Tensor | np.ndarray] = None,
    radial_basis_function: RadialBasisFunction = RadialBasisFunction.GAUSSIAN,
    sigma: Optional[float | int | torch.Tensor | np.ndarray] = None,
):
    """Build the design matrix P for the "second approach" (no pre-training).

    Given samples X in R^{l x n} and a set of centres C, compute the activation
    matrix Phi (l x m) using an RBF, then construct P in R^{l x (m*n)} where each
    group of n columns corresponds to a centre and equals Phi[:, j] element-wise
    multiplied by each column of X.

    Specifically, using 1-based indexing in the documentation's notation:
        P_{i, (j-1)*n + k} = Phi_{i, j} * X_{i, k}

    Args:
        X: Input matrix of shape (l, n). Torch tensor or NumPy array.
        d: Target vector of shape (l,). Not used here; kept for interface parity.
        centres: Optional matrix of shape (m, n) containing centre points.
            If None, all l rows of X are used as centres (m = l).
        radial_basis_function: Choice of RBF (Gaussian or Laplacian).

    Returns:
        P with shape (l, m*n), same array/tensor type as X.

    Notes:
                - RBF bandwidth (sigma): if provided, it overrides the heuristic and must
                    be a positive scalar. If None, it's chosen via the median heuristic over
                    pairwise distances between X and the selected centres. A small epsilon
                    guards division by zero; if the median is 0, sigma falls back to 1.0.
        - Gaussian uses exp(-||x - c||^2 / (2*sigma^2)); Laplacian uses
          exp(-||x - c||_2 / sigma).
    """
    # Validate input ranks quickly (accept both torch and numpy)
    if not isinstance(X, (torch.Tensor, np.ndarray)):
        raise TypeError("X must be a torch.Tensor or numpy.ndarray")
    if not isinstance(d, (torch.Tensor, np.ndarray)):
        raise TypeError("d must be a torch.Tensor or numpy.ndarray")

    # Unify to torch for computation while preserving device/dtype when possible
    if isinstance(X, torch.Tensor):
        device = X.device
        dtype = X.dtype if X.dtype.is_floating_point else torch.get_default_dtype()
        X_t = X.to(device=device, dtype=dtype)
        d_t = (
            d.to(device=device, dtype=dtype)
            if isinstance(d, torch.Tensor)
            else torch.from_numpy(d).to(device=device, dtype=dtype)
        )
        return_numpy = False
    else:
        # NumPy path: compute on CPU with default dtype
        device = torch.device("cpu")
        # Prefer float32 for speed unless numpy is float64 already
        np_is_f64 = X.dtype == np.float64 or (
            isinstance(d, np.ndarray) and d.dtype == np.float64
        )
        dtype = torch.float64 if np_is_f64 else torch.float32
        X_t = torch.from_numpy(X).to(device=device, dtype=dtype)
        d_t = (
            torch.from_numpy(d).to(device=device, dtype=dtype)
            if isinstance(d, np.ndarray)
            else d.to(device=device, dtype=dtype)
        )
        return_numpy = True

    if X_t.ndim != 2:
        raise ValueError("X must be 2D (l, n)")
    if d_t.ndim != 1:
        raise ValueError("d must be 1D (l,)")
    l, n = X_t.shape
    if d_t.shape[0] != l:
        raise ValueError("X and d must have compatible first dimension (l)")

    # Determine centres
    if centres is None:
        centres = X_t  # Use all rows of X as centres
    else:
        if isinstance(centres, np.ndarray):
            centres = torch.from_numpy(centres).to(device=device, dtype=dtype)
        else:
            centres = centres.to(device=device, dtype=dtype)
        if centres.ndim != 2:
            raise ValueError("centres must be a 2D array (m, n)")
        if centres.shape[1] != n:
            raise ValueError("centres must have same number of columns as X")
        if centres.shape[0] == 0:
            raise ValueError("centres must be non-empty")

    m = centres.shape[0]

    # Compute pairwise distances between all samples and centres efficiently
    # D2 = ||x||^2 + ||c||^2 - 2 x c^T, clamped at 0 for numerical safety
    x_sq = (X_t * X_t).sum(dim=1, keepdim=True)  # (l, 1)
    centres_sq = (centres * centres).sum(dim=1)  # (m,)
    dist_sq = (x_sq + centres_sq.unsqueeze(0)) - 2.0 * (
        X_t @ centres.transpose(0, 1)
    )  # (l, m)
    dist_sq = torch.clamp(dist_sq, min=0.0)

    # Bandwidth sigma: use provided value if any, otherwise median heuristic
    sigma_t: torch.Tensor
    if sigma is None:
        # For sigma estimation, we'll compute distances between X and centres directly
        x_sq_sigma = (X_t * X_t).sum(dim=1, keepdim=True)
        centres_sq_sigma = (centres * centres).sum(dim=1)
        dist_sq_sigma = (x_sq_sigma + centres_sq_sigma.unsqueeze(0)) - 2.0 * (
            X_t @ centres.transpose(0, 1)
        )
        dist_sq_sigma = torch.clamp(dist_sq_sigma, min=0.0)
        nonzero_dist_sq = dist_sq_sigma[dist_sq_sigma > 0]

        if nonzero_dist_sq.numel() == 0:
            sigma_t = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            sigma_t = torch.median(torch.sqrt(nonzero_dist_sq + 1e-12))
            if not torch.isfinite(sigma_t):
                sigma_t = torch.tensor(1.0, device=device, dtype=dtype)
        sigma_t = torch.clamp(sigma_t, min=torch.finfo(dtype).eps)
    else:
        # Accept Python scalar, NumPy scalar/array (0-d), or torch scalar
        if isinstance(sigma, torch.Tensor):
            if sigma.numel() != 1:
                raise ValueError("sigma must be a positive scalar")
            sigma_t = sigma.to(device=device, dtype=dtype).reshape(())
        elif isinstance(sigma, np.ndarray):
            if np.asarray(sigma).size != 1:
                raise ValueError("sigma must be a positive scalar")
            sigma_t = torch.as_tensor(
                float(np.asarray(sigma).item()), device=device, dtype=dtype
            )
        else:
            # int | float
            try:
                sigma_t = torch.tensor(float(sigma), device=device, dtype=dtype)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(
                    "sigma must be convertible to a positive float"
                ) from exc
        if not torch.isfinite(sigma_t) or (sigma_t <= 0):
            raise ValueError("sigma must be a positive, finite scalar")

    match radial_basis_function:
        case RadialBasisFunction.GAUSSIAN:
            denom = 2.0 * (sigma_t * sigma_t)
            phi = torch.exp(-dist_sq / denom)
        case RadialBasisFunction.LAPLACIAN:
            # Compute L2 distances only when needed for Laplacian
            dist = torch.sqrt(dist_sq + 1e-12)
            phi = torch.exp(-dist / (sigma_t + torch.finfo(dtype).eps))
        case _:
            raise ValueError(
                f"Unsupported radial_basis_function: {radial_basis_function}"
            )

    # Construct P: shape (l, m*n) with column order k varying fastest inside each centre j
    # Broadcast multiply then reshape
    p_t = (phi.unsqueeze(-1) * X_t.unsqueeze(1)).reshape(l, m * n)

    if return_numpy:
        # Match NumPy dtype to input's float kind
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        return p_t.cpu().numpy().astype(np_dtype, copy=False)
    return p_t


def construct_design_matrix_with_local_pretraining(
    X: torch.Tensor | np.ndarray,
    d: torch.Tensor | np.ndarray,
    centres: torch.Tensor | np.ndarray,
    weights: Optional[torch.Tensor | np.ndarray] = None,
    radial_basis_function: RadialBasisFunction = RadialBasisFunction.GAUSSIAN,
    sigma: Optional[float | int | torch.Tensor | np.ndarray] = None,
    ridge: float = 1e-8,
    rho: float = 0.05,
    return_weights: bool = False,
):
    """Build the design matrix P for the "first approach" (locally pre-trained).

    For each candidate centre j, fit a local linear model w_j in R^n by a
    weighted least squares problem: minimize sum_i phi_{i,j} (d_i - X_i·w)^2.
    Then the j-th column of the design matrix is
        P[:, j] = phi[:, j] ⊙ (X @ w_j).

    This yields P in R^{l x m}, so optimizers can select centres with scalar
    weights.

    Args:
        X: Input matrix of shape (l, n).
        d: Target vector of shape (l,).
        centres: Centre matrix of shape (m, n). These centres are used directly.
        weights: Optional pre-computed weights of shape (m, n). If provided, skips local training
            and uses these weights to construct P directly (useful for test set construction).
        radial_basis_function: RBF type used to build phi.
        sigma: Optional bandwidth; if None, median-heuristic is used.
        ridge: Small non-negative Tikhonov regularization to stabilize WLS.
        rho: Non-negative threshold for local support; points with RBF activation
            below rho are ignored in the local fit. Default 0.05.
        return_weights: If True, also return the local model weights of shape (m, n).

    Returns:
        P with shape (l, m), same array/tensor type as X.
        If return_weights=True, returns (P, weights) tuple.
    """
    if not isinstance(X, (torch.Tensor, np.ndarray)):
        raise TypeError("X must be a torch.Tensor or numpy.ndarray")
    if not isinstance(d, (torch.Tensor, np.ndarray)):
        raise TypeError("d must be a torch.Tensor or numpy.ndarray")

    # Unify to torch for computation
    if isinstance(X, torch.Tensor):
        device = X.device
        dtype = X.dtype if X.dtype.is_floating_point else torch.get_default_dtype()
        X_t = X.to(device=device, dtype=dtype)
        d_t = (
            d.to(device=device, dtype=dtype)
            if isinstance(d, torch.Tensor)
            else torch.from_numpy(d).to(device=device, dtype=dtype)
        )
        return_numpy = False
    else:
        device = torch.device("cpu")
        np_is_f64 = X.dtype == np.float64 or (
            isinstance(d, np.ndarray) and d.dtype == np.float64
        )
        dtype = torch.float64 if np_is_f64 else torch.float32
        X_t = torch.from_numpy(X).to(device=device, dtype=dtype)
        d_t = (
            torch.from_numpy(d).to(device=device, dtype=dtype)
            if isinstance(d, np.ndarray)
            else d.to(device=device, dtype=dtype)
        )
        return_numpy = True

    if X_t.ndim != 2:
        raise ValueError("X must be 2D (l, n)")
    if d_t.ndim != 1:
        raise ValueError("d must be 1D (l,)")
    l, n = X_t.shape
    if d_t.shape[0] != l:
        raise ValueError("X and d must have compatible first dimension (l)")

    # Determine centres
    if isinstance(centres, np.ndarray):
        centres_t = torch.from_numpy(centres).to(device=device, dtype=dtype)
    else:
        centres_t = centres.to(device=device, dtype=dtype)
    if centres_t.ndim != 2:
        raise ValueError("centres must be 2D (m, n)")
    if centres_t.shape[1] != n:
        raise ValueError("centres must have same number of features as X")
    m = centres_t.shape[0]

    # Distances
    x_sq = (X_t * X_t).sum(dim=1, keepdim=True)
    centres_sq = (centres_t * centres_t).sum(dim=1)
    dist_sq = (x_sq + centres_sq.unsqueeze(0)) - 2.0 * (X_t @ centres_t.transpose(0, 1))
    dist_sq = torch.clamp(dist_sq, min=0.0)

    # Sigma
    if sigma is None:
        # Use centres for sigma estimation
        sigma_t = estimate_sigma_median(
            X_t, None
        )  # Use all points for sigma estimation
    else:
        if isinstance(sigma, torch.Tensor):
            if sigma.numel() != 1:
                raise ValueError("sigma must be a positive scalar")
            sigma_t = sigma.to(device=device, dtype=dtype).reshape(())
        elif isinstance(sigma, np.ndarray):
            if np.asarray(sigma).size != 1:
                raise ValueError("sigma must be a positive scalar")
            sigma_t = torch.as_tensor(
                float(np.asarray(sigma).item()), device=device, dtype=dtype
            )
        else:
            sigma_t = torch.tensor(float(sigma), device=device, dtype=dtype)
        if not torch.isfinite(sigma_t) or (sigma_t <= 0):
            raise ValueError("sigma must be a positive, finite scalar")

    # Phi
    match radial_basis_function:
        case RadialBasisFunction.GAUSSIAN:
            denom = 2.0 * (sigma_t * sigma_t)
            phi = torch.exp(-dist_sq / denom)
        case RadialBasisFunction.LAPLACIAN:
            dist = torch.sqrt(dist_sq + 1e-12)
            phi = torch.exp(-dist / (sigma_t + torch.finfo(dtype).eps))
        case _:
            raise ValueError(
                f"Unsupported radial_basis_function: {radial_basis_function}"
            )

    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    # If weights are provided, use them directly to construct P (useful for test set)
    if weights is not None:
        if isinstance(weights, np.ndarray):
            weights_t = torch.from_numpy(weights).to(device=device, dtype=dtype)
        else:
            weights_t = weights.to(device=device, dtype=dtype)
        if weights_t.shape != (m, n):
            raise ValueError(f"weights must have shape (m, n) = ({m}, {n})")
        P = phi * (X_t @ weights_t.T)  # (l, m)
    else:
        # Compute weights through local training
        eye = torch.eye(n, device=device, dtype=dtype)
        # Vectorized computation to replace the loop
        Xi_mask = torch.where(
            phi > rho, phi, torch.tensor(0.0, device=device, dtype=dtype)
        )  # (l, m)
        X_masked_out = Xi_mask.unsqueeze(-1) * X_t.unsqueeze(1)  # (l, m, n)
        a = torch.einsum("lmi, lj -> mij", X_masked_out, X_t) + ridge * eye.unsqueeze(
            0
        )  # (m, n, n)
        b = torch.einsum("lmi, l -> mi", X_masked_out, d_t)  # (m, n)
        nu_hat = torch.linalg.solve(a, b)  # (m, n)
        P = phi * (X_t @ nu_hat.T)  # (l, m)

    if return_numpy:
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        P_numpy = P.cpu().numpy().astype(np_dtype, copy=False)
        if return_weights:
            if weights is not None:
                # Return the provided weights
                weights_numpy = weights_t.cpu().numpy().astype(np_dtype, copy=False)
            else:
                # Return the computed weights
                weights_numpy = nu_hat.cpu().numpy().astype(np_dtype, copy=False)
            return P_numpy, weights_numpy
        return P_numpy
    else:
        if return_weights:
            if weights is not None:
                # Return the provided weights
                return P, weights_t
            else:
                # Return the computed weights
                return P, nu_hat
        return P
