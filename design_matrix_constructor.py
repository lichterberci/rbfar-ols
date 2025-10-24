from typing import Optional, Tuple
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


def select_centres_kmeans(
    X: torch.Tensor | np.ndarray,
    m: int,
    max_iters: int = 100,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Select centres using K-means clustering.

    Args:
        X: Input matrix of shape (l, n). Torch tensor or NumPy array.
        m: Number of centres to select.
        max_iters: Maximum number of K-means iterations.
        tol: Convergence tolerance for K-means.

    Returns:
        Centres tensor of shape (m, n), same type/device as X.
    """
    if not isinstance(X, (torch.Tensor, np.ndarray)):
        raise TypeError("X must be a torch.Tensor or numpy.ndarray")

    # Unify to torch for computation
    if isinstance(X, torch.Tensor):
        device = X.device
        dtype = X.dtype if X.dtype.is_floating_point else torch.get_default_dtype()
        X_t = X.to(device=device, dtype=dtype)
        return_numpy = False
    else:
        device = torch.device("cpu")
        dtype = torch.float64 if X.dtype == np.float64 else torch.float32
        X_t = torch.from_numpy(X).to(device=device, dtype=dtype)
        return_numpy = True

    if X_t.ndim != 2:
        raise ValueError("X must be 2D (l, n)")
    l, n = X_t.shape

    if m <= 0:
        raise ValueError("m must be positive")
    if m > l:
        raise ValueError(f"Cannot select {m} centres from {l} points")

    # Initialize centres randomly
    init_indices = torch.randperm(l, device=device)[:m]
    centres = X_t[init_indices].clone()

    for _ in range(max_iters):
        # Assign each point to closest centre
        dist_sq = torch.cdist(X_t, centres, p=2) ** 2  # (l, m)
        assignments = torch.argmin(dist_sq, dim=1)  # (l,)

        # Update centres
        new_centres = torch.zeros_like(centres)
        for k in range(m):
            mask = assignments == k
            if mask.sum() > 0:
                new_centres[k] = X_t[mask].mean(dim=0)
            else:
                # If no points assigned, keep the old centre
                new_centres[k] = centres[k]

        # Check convergence
        centre_shift = torch.norm(new_centres - centres, dim=1).max()
        centres = new_centres.clone()

        if centre_shift < tol:
            break

    if return_numpy:
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        return centres.cpu().numpy().astype(np_dtype, copy=False)
    return centres


def estimate_local_sigma_knn(
    centres: torch.Tensor | np.ndarray,
    k: int = 5,
) -> torch.Tensor:
    """Estimate local sigma for each centre using KNN metric.

    For each centre C_j, computes:
    σ_j = (1/√2) * (1/k) * Σ_{i=1}^k ||C_j - C_{NN_i}||_2

    where C_{NN_i} are the k nearest neighbors of centre C_j.

    Args:
        centres: Centre matrix of shape (m, n). Torch tensor or NumPy array.
        k: Number of nearest neighbors to use for estimation.

    Returns:
        Local sigma values of shape (m,), same type/device as centres.
    """
    if not isinstance(centres, (torch.Tensor, np.ndarray)):
        raise TypeError("centres must be a torch.Tensor or numpy.ndarray")

    # Unify to torch for computation
    if isinstance(centres, torch.Tensor):
        device = centres.device
        dtype = (
            centres.dtype
            if centres.dtype.is_floating_point
            else torch.get_default_dtype()
        )
        centres_t = centres.to(device=device, dtype=dtype)
        return_numpy = False
    else:
        device = torch.device("cpu")
        dtype = torch.float64 if centres.dtype == np.float64 else torch.float32
        centres_t = torch.from_numpy(centres).to(device=device, dtype=dtype)
        return_numpy = True

    if centres_t.ndim != 2:
        raise ValueError("centres must be 2D (m, n)")
    m, n = centres_t.shape

    if k <= 0:
        raise ValueError("k must be positive")
    if k >= m:
        # If we don't have enough centres, use all available ones
        k = max(1, m - 1)

    # Handle the case of a single centre
    if m == 1:
        # Return a small but positive sigma for the single centre
        sigma_val = torch.tensor(1, device=device, dtype=dtype).unsqueeze(0)
        if return_numpy:
            np_dtype = np.float64 if dtype == torch.float64 else np.float32
            return sigma_val.cpu().numpy().astype(np_dtype, copy=False)
        return sigma_val

    # Compute pairwise distances between all centres
    dist = torch.cdist(centres_t, centres_t, p=2)  # (m, m)

    # For each centre, find k nearest neighbors (excluding itself)
    # Sort distances and take the k+1 smallest (excluding the 0 distance to itself)
    sorted_dists, _ = torch.sort(dist, dim=1)  # (m, m)
    knn_dists = sorted_dists[:, 1 : k + 1]  # (m, k) - exclude self-distance

    # Compute local sigma for each centre
    local_sigmas = (
        1.0 / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    ) * torch.mean(
        knn_dists, dim=1
    )  # (m,)

    # Ensure we don't get zero or very small sigmas
    local_sigmas = torch.clamp(local_sigmas, min=torch.finfo(dtype).eps * 100)

    if return_numpy:
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        return local_sigmas.cpu().numpy().astype(np_dtype, copy=False)
    return local_sigmas


def construct_design_matrix_with_no_pretraining(
    X: torch.Tensor | np.ndarray,
    d: torch.Tensor | np.ndarray,
    centres: Optional[torch.Tensor | np.ndarray] = None,
    radial_basis_function: RadialBasisFunction = RadialBasisFunction.GAUSSIAN,
    sigma: Optional[float | int | torch.Tensor | np.ndarray] = None,
):
    """Build the design matrix P for the "second approach" (no pre-training).

    Given samples X in R^{l x n} and a set of centres C, construct P in R^{l x (n+1)*(m+1)} where:
    - We have (n+1) functions: φ₀(X) (constant), φ₁(X), ..., φₙ(X) (coefficients for each feature)
    - Each function φᵢ(X) = νᵢ,₀ + Σⱼ₌₁ᵐ νᵢ,ⱼ Ψⱼ(X) has (m+1) parameters
    - Total parameters: (n+1) functions × (m+1) parameters = (n+1)*(m+1)

    Column layout: [φ₀_global, φ₀_center1, ..., φ₀_centerₘ, φ₁_global, φ₁_center1, ...]
    Specifically: P[:, i*(m+1) + 0] = 1 (global terms)
                  P[:, i*(m+1) + j+1] = Ψⱼ(X) for i=0 (constant function)
                  P[:, i*(m+1) + j+1] = Ψⱼ(X) * X[:, i-1] for i=1..n (feature functions)

    Args:
        X: Input matrix of shape (l, n). Torch tensor or NumPy array.
        d: Target vector of shape (l,). Not used here; kept for interface parity.
        centres: Optional matrix of shape (m, n) containing centre points.
            If None, all l rows of X are used as centres (m = l).
        radial_basis_function: Choice of RBF (Gaussian or Laplacian).

    Returns:
        P with shape (l, (n+1)*(m+1)), same array/tensor type as X.

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

    # Construct P: shape (l, (n+1)*(m+1))
    # For each function φᵢ (i=0..n), we have (m+1) columns: [global, center1, ..., centerₘ]
    ones_col = torch.ones(l, 1, device=X_t.device, dtype=X_t.dtype)  # Global terms

    # Build the design matrix function by function
    P_blocks = []

    # φ₀(X) - constant function: [1, Ψ₁(X), Ψ₂(X), ..., Ψₘ(X)]
    phi_0_block = torch.cat([ones_col, phi], dim=1)  # Shape: (l, m+1)
    P_blocks.append(phi_0_block)

    # φᵢ(X) - coefficient functions for i=1..n: [Xᵢ, Xᵢ*Ψ₁(X), Xᵢ*Ψ₂(X), ..., Xᵢ*Ψₘ(X)]
    for i in range(n):
        X_i = X_t[:, i : i + 1]  # Shape: (l, 1)
        phi_i_block = torch.cat([X_i, X_i * phi], dim=1)  # Shape: (l, m+1)
        P_blocks.append(phi_i_block)

    # Concatenate all blocks horizontally
    p_t = torch.cat(P_blocks, dim=1)  # Shape: (l, (n+1)*(m+1))

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
    use_local_sigma: bool = False,
    local_sigma_k: int = 5,
    ridge: float = 1e-8,
    rho: float = 0.05,
    return_weights: bool = False,
):
    """Build the design matrix P for the "first approach" (locally pre-trained).

    For each candidate centre j, fit a local linear model w_j in R^n by a
    weighted least squares problem: minimize sum_i phi_{i,j} (d_i - X_i·w_j - c_j)^2.
    The design matrix includes both the local linear predictions and constant terms:
        P[:, j] = phi[:, j] ⊙ (X @ w_j) for j=0..m-1 (local linear models)
        P[:, m] = sum_j phi[:, j] (global constant term)

    This yields P in R^{l x m+1}, where the last column represents the constant term.

    Args:
        X: Input matrix of shape (l, n).
        d: Target vector of shape (l,).
        centres: Centre matrix of shape (m, n). These centres are used directly.
        weights: Optional pre-computed weights of shape (m, n+1). If provided, skips local training
            and uses these weights to construct P directly. The weights should include both
            linear coefficients (first n entries) and constant terms (last entry) for each centre.
        radial_basis_function: RBF type used to build phi.
        sigma: Optional bandwidth; if None, median-heuristic is used. If use_local_sigma=True,
            this parameter is ignored and local sigmas are computed instead.
        use_local_sigma: If True, compute local sigma values for each centre using KNN.
        local_sigma_k: Number of nearest neighbors for local sigma estimation.
        ridge: Small non-negative Tikhonov regularization to stabilize WLS.
        rho: Non-negative threshold for local support; points with RBF activation
            below rho are ignored in the local fit. Default 0.05.
        return_weights: If True, also return the local model weights of shape (m, n+1).

    Returns:
        P with shape (l, m+1), same array/tensor type as X.
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

    # Sigma estimation
    if use_local_sigma:
        # Use local KNN-based sigma estimation
        local_sigmas_t = estimate_local_sigma_knn(centres_t, k=local_sigma_k)

        # Phi computation with local sigmas
        match radial_basis_function:
            case RadialBasisFunction.GAUSSIAN:
                # For each centre j, use sigma_j for computing distances to that centre
                # dist_sq is (l, m), local_sigmas_t is (m,)
                denom = 2.0 * (local_sigmas_t * local_sigmas_t).unsqueeze(0)  # (1, m)
                phi = torch.exp(-dist_sq / denom)  # (l, m)
            case RadialBasisFunction.LAPLACIAN:
                dist = torch.sqrt(dist_sq + 1e-12)
                phi = torch.exp(
                    -dist / (local_sigmas_t.unsqueeze(0) + torch.finfo(dtype).eps)
                )
            case _:
                raise ValueError(
                    f"Unsupported radial_basis_function: {radial_basis_function}"
                )
    else:
        # Use global sigma estimation
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

        # Phi computation with global sigma
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
        if weights_t.shape != (m, n + 1):
            raise ValueError(f"weights must have shape (m, n+1) = ({m}, {n + 1})")
        # Split weights into linear coefficients and constant terms
        linear_weights = weights_t[:, :n]  # (m, n)
        constant_weights = weights_t[:, n]  # (m,)

        # Construct P: local linear models + global constant term
        P_local = phi * (X_t @ linear_weights.T)  # (l, m)
        P_constant = (phi * constant_weights.unsqueeze(0)).sum(
            dim=1, keepdim=True
        )  # (l, 1)
        P = torch.cat([P_local, P_constant], dim=1)  # (l, m+1)
    else:
        # Compute weights through local training with augmented design matrix
        # Augment X with ones column for constant terms: X_aug = [X, 1]
        ones_col = torch.ones(l, 1, device=device, dtype=dtype)
        X_aug = torch.cat([X_t, ones_col], dim=1)  # (l, n+1)

        eye_aug = torch.eye(n + 1, device=device, dtype=dtype)
        # Vectorized computation to replace the loop

        theta = torch.exp(
            torch.tensor(-1 / 4, device=device, dtype=dtype)
        )  # the value of Phi at the halfway point

        Xi_mask = torch.where(
            phi > theta, phi, torch.tensor(0.0, device=device, dtype=dtype)
        )  # (l, m)

        # Xi_mask = torch.where(
        #     phi > rho, phi, torch.tensor(0.0, device=device, dtype=dtype)
        # )  # (l, m)
        X_masked_out = Xi_mask.unsqueeze(-1) * X_aug.unsqueeze(1)  # (l, m, n+1)
        a = torch.einsum(
            "lmi, lmj -> mij", X_masked_out, X_masked_out
        ) + ridge * eye_aug.unsqueeze(
            0
        )  # (m, n+1, n+1)
        b = torch.einsum("lmi, l -> mi", X_masked_out, d_t)  # (m, n+1)
        nu_hat = torch.linalg.lstsq(a, b).solution  # (m, n+1)

        # Split the fitted weights
        linear_weights = nu_hat[:, :n]  # (m, n)
        constant_weights = nu_hat[:, n]  # (m,)

        # Construct P: local linear models + global constant term
        P_local = phi * (X_t @ linear_weights.T)  # (l, m)
        P_constant = (phi * constant_weights.unsqueeze(0)).sum(
            dim=1, keepdim=True
        )  # (l, 1)
        P = torch.cat([P_local, P_constant], dim=1)  # (l, m+1)

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
