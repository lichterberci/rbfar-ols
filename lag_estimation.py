import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum, auto
from tqdm import tqdm


class TauEstimator(ABC):

    @abstractmethod
    def estimate(self, x: np.ndarray[np.floating]):
        pass


class OptimalLagSelectionMethod(StrEnum):
    MIN = auto()
    FIRST_LOC_MIN = auto()


@dataclass
class AmiEstimator(TauEstimator):

    max_lag: int = 100
    min_lag: int = 1
    num_bins_in_ami: int = 32
    optimum_selection_method: OptimalLagSelectionMethod = (
        OptimalLagSelectionMethod.FIRST_LOC_MIN
    )
    verbose: bool = False

    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate mutual information I(X;Y) using histogram binning.

        Notes:
        - This helper expects 1D arrays x and y (same length).
        - For multivariate series, the caller should aggregate across dimensions
          by calling this on (feature_i at t, feature_j at t+lag) pairs.
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        if x.size == 0:
            return np.nan

        # Remove non-finite pairs
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return np.nan
        x = x[mask]
        y = y[mask]

        # Compute joint histogram and normalize to probabilities
        joint_hist, _, _ = np.histogram2d(x, y, bins=self.num_bins_in_ami)
        total = joint_hist.sum()
        if total == 0:
            return 0.0

        p_xy = joint_hist / total
        # Marginals
        p_x = p_xy.sum(axis=1, keepdims=True)
        p_y = p_xy.sum(axis=0, keepdims=True)

        # Expected independent joint
        expected = p_x @ p_y  # outer product

        # Only where probabilities are positive
        mask = p_xy > 0
        mi = np.sum(p_xy[mask] * np.log(p_xy[mask] / expected[mask]))
        return float(mi)

    def estimate(self, x: np.ndarray[np.floating]):
        """
        Estimate the average mutual information (AMI) for a range of lags.

        Supports:
        - 1D time series: x.shape == (n,)
        - 2D time series: x.shape == (n, m) — AMI is averaged over all (i, j)
          feature pairs between x_t[:, i] and x_{t+lag}[:, j]. This heuristic is
          robust for lag selection even if not a full multivariate MI.
        """
        if x is None:
            raise ValueError("Input array must not be None")

        x = np.asarray(x)

        if x.ndim == 1:
            n_samples = x.shape[0]
            n_features = 1
        elif x.ndim == 2:
            n_samples, n_features = x.shape
        else:
            raise ValueError("Input array must be 1D or 2D: (n,) or (n, m)")

        if self.min_lag < 1 or self.max_lag < self.min_lag:
            raise ValueError("Invalid lag bounds: ensure 1 <= min_lag <= max_lag")

        lags = range(self.min_lag, min(self.max_lag, n_samples - 1) + 1)

        def mi_for_lag(lag: int) -> float:
            if x.ndim == 1:
                return self._mutual_information(x[:-lag], x[lag:])
            else:
                # Average MI across all feature pairs (i at t, j at t+lag)
                A = x[:-lag, :]
                B = x[lag:, :]
                s = 0.0
                cnt = 0
                for i in range(n_features):
                    xi = A[:, i]
                    for j in range(n_features):
                        yj = B[:, j]
                        val = self._mutual_information(xi, yj)
                        if np.isfinite(val):
                            s += val
                            cnt += 1
                return s / cnt if cnt > 0 else np.nan

        match self.optimum_selection_method:
            case OptimalLagSelectionMethod.MIN:
                mi_values = [
                    mi_for_lag(lag)
                    for lag in tqdm(
                        lags, desc="Estimating lag using AMI", disable=not self.verbose
                    )
                ]
                tau = self.min_lag + np.argmin(mi_values)
                return tau
            case OptimalLagSelectionMethod.FIRST_LOC_MIN:
                prev_mi = None
                lags_list = list(lags)
                for idx, lag in tqdm(
                    enumerate(lags),
                    desc="Estimating lag using AMI",
                    disable=not self.verbose,
                ):
                    mi = mi_for_lag(lag)
                    if idx > 0 and idx < len(lags_list) - 1:
                        # Check for local minimum
                        if prev_mi is not None and prev_mi < mi:
                            return lag - 1
                    prev_mi = mi
                # If no local minimum found, return the last evaluated lag
                return lags_list[-1]


def estimate_tau(x: np.ndarray[np.floating], estimator: TauEstimator):
    return estimator.estimate(x)
