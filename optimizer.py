from __future__ import annotations

"""Common optimizer base class and shared utilities."""

from abc import ABC, abstractmethod
from typing import Tuple, cast

import numpy as np
import torch

# Local type alias to avoid repeating the same literal in casts
ArrayLike = np.ndarray | torch.Tensor


class Optimizer(ABC):
    """
    Abstract base class for optimizers in this project.

    Provides small, shared utilities for type/shape validation and
    converting inputs between NumPy and Torch while keeping device/dtype
    choices explicit in subclasses.
    """

    @abstractmethod
    def optimize(
        self,
        P: torch.Tensor | np.ndarray,
        d: torch.Tensor | np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[np.ndarray, np.ndarray]:
        """
        Implement in subclasses to return (selected_indices, weights).

        This method is abstract and intentionally has no implementation here.
        """
        raise NotImplementedError

    # ---- Shared helpers -------------------------------------------------
    @staticmethod
    def _should_return_numpy(
        P: torch.Tensor | np.ndarray, d: torch.Tensor | np.ndarray
    ) -> bool:
        """Return True when both inputs are NumPy arrays, to mirror return types."""
        return isinstance(P, np.ndarray) and isinstance(d, np.ndarray)

    @staticmethod
    def _validate_inputs(
        P: torch.Tensor | np.ndarray, d: torch.Tensor | np.ndarray
    ) -> tuple[int, int]:
        """Validate types and shapes, returning (l, m) for P with shape (l, m).

        Raises TypeError/ValueError on mismatch.
        """
        if not (
            isinstance(P, (torch.Tensor, np.ndarray))
            and isinstance(d, (torch.Tensor, np.ndarray))
        ):
            raise TypeError("P and d must be either torch.Tensor or np.ndarray")

        if cast(ArrayLike, P).ndim != 2:
            raise ValueError("P must be 2D")
        if cast(ArrayLike, d).ndim != 1:
            raise ValueError("d must be 1D")

        l, m = cast(ArrayLike, P).shape  # type: ignore[assignment]
        if cast(ArrayLike, d).shape[0] != l:
            raise ValueError("Incompatible shapes: P and d must have same rows.")

        return int(l), int(m)

    @staticmethod
    def _to_torch(
        P: torch.Tensor | np.ndarray,
        d: torch.Tensor | np.ndarray,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.device, torch.dtype]:
        """
        Convert inputs to torch tensors, optionally enforcing device/dtype.

        - If device/dtype are not provided, they will be inferred from torch inputs
          when available; otherwise defaults to CPU and the Torch default dtype.
        - Torch inputs will be moved/cast only when device/dtype are specified.
        """
        # Infer defaults
        if isinstance(P, torch.Tensor):
            device = device or P.device
            dtype = dtype or P.dtype
        elif isinstance(d, torch.Tensor):
            device = device or d.device
            dtype = dtype or d.dtype
        else:
            device = device or torch.device("cpu")
            dtype = dtype or torch.get_default_dtype()  # typically float32/float64

        # Convert or cast
        if isinstance(P, np.ndarray):
            P_t = torch.from_numpy(P).to(device=device, dtype=dtype)
        else:
            P_t = P.to(device=device, dtype=dtype)

        if isinstance(d, np.ndarray):
            d_t = torch.from_numpy(d).to(device=device, dtype=dtype)
        else:
            d_t = d.to(device=device, dtype=dtype)

        return P_t, d_t, device, dtype
