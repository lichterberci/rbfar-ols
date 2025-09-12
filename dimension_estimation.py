import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum, auto


class DimensionEstimator(ABC):

    @abstractmethod
    def estimate(self, x: np.ndarray[np.floating]):
        pass


class OptimalDimensionSelectionMethod(StrEnum):
    MIN = auto()
    FIRST_LOC_MIN = auto()
    E1_E2_COMBINED = auto()  # Use both E1 and E2 for robust estimation


@dataclass
class CaoEstimator(DimensionEstimator):
    delay: int = 1
    max_dim: int = 10
    verbose: bool = False
    plot: bool = False
    optimum_selection_method: OptimalDimensionSelectionMethod = (
        OptimalDimensionSelectionMethod.MIN
    )
    epsilon: float = 1e-10

    def _cao(self, x: np.ndarray[np.floating], d: int):
        """Compute Cao's E1 and E2 statistics for dimension d.

        E1(d) = mean(|x_d+1(i) - x_d+1(nn(i))| / |x_d(i) - x_d(nn(i))|)
        E2(d) = mean(|x_d+1(i) - x_d+1(nn*(i))| / |x_d(i) - x_d(nn*(i))|)

        where nn(i) is the nearest neighbor of x_d(i) in d-dimensional space
        and nn*(i) is the nearest neighbor considering all points.

        Supports 1D (n,) and 2D (n, m) inputs. For 2D inputs, constructs
        multivariate delay embeddings by concatenating all m features at each
        lag: embedding_d shape (n_vectors_d, d*m), embedding_{d+1} shape
        (n_vectors_d1, (d+1)*m).

        Returns:
            tuple: (E1, E2) statistics
        """
        if x.ndim == 1:
            n = x.shape[0]
            m = 1
        elif x.ndim == 2:
            n, m = x.shape
        else:
            return np.nan, np.nan

        n_vectors_d = n - (d - 1) * self.delay
        n_vectors_d1 = n - d * self.delay

        if n_vectors_d1 < 2 or n_vectors_d <= 0:
            return np.nan, np.nan

        if m == 1:
            # 1D original path for speed/compatibility
            embedding = np.empty((n_vectors_d, d))
            for i in range(d):
                embedding[:, i] = x[i * self.delay : i * self.delay + n_vectors_d]

            next_dim_embedding = np.empty((n_vectors_d1, d + 1))
            for i in range(d + 1):
                next_dim_embedding[:, i] = x[
                    i * self.delay : i * self.delay + n_vectors_d1
                ]
        else:
            # Multivariate: concatenate features for each lag
            embedding = np.empty((n_vectors_d, d * m))
            for i in range(d):
                block = x[i * self.delay : i * self.delay + n_vectors_d, :]
                embedding[:, i * m : (i + 1) * m] = block

            next_dim_embedding = np.empty((n_vectors_d1, (d + 1) * m))
            for i in range(d + 1):
                block = x[i * self.delay : i * self.delay + n_vectors_d1, :]
                next_dim_embedding[:, i * m : (i + 1) * m] = block

        # Match the first n_vectors_d1 vectors from X_d to X_{d+1}
        embedding_trimmed = embedding[:n_vectors_d1]

        # Use KDTree for nearest neighbor search (excluding self-match)
        tree = cKDTree(embedding_trimmed)
        dist_d, nn_indices = tree.query(embedding_trimmed, k=2)

        # Use the second neighbor (first is self)
        nn_indices_1 = nn_indices[:, 1]
        distances_d = dist_d[:, 1]

        # Distances in (d+1)-dimensional space for the same point pairs
        diffs = next_dim_embedding - next_dim_embedding[nn_indices_1]
        distances_d1 = np.linalg.norm(diffs, axis=1)

        # Avoid division by zero
        valid_mask = distances_d > self.epsilon
        if not np.any(valid_mask):
            return np.nan, np.nan

        # Calculate E1 (standard Cao statistic)
        ratios_e1 = distances_d1[valid_mask] / distances_d[valid_mask]
        E1 = float(np.mean(ratios_e1))

        # Calculate E2 (variance-based statistic for distinguishing deterministic vs random data)
        # E2 uses a different nearest neighbor selection strategy
        # For E2, we look at all possible neighbors and compute the variance
        if n_vectors_d1 > 1:
            # For E2, we need to consider all possible nearest neighbors
            tree_full = cKDTree(next_dim_embedding)
            # Find nearest neighbor in (d+1)-dimensional space
            dist_d1_full, nn_indices_d1 = tree_full.query(next_dim_embedding, k=2)
            nn_indices_d1_1 = nn_indices_d1[:, 1]

            # Corresponding distances in d-dimensional space
            diffs_d_for_e2 = embedding_trimmed - embedding_trimmed[nn_indices_d1_1]
            distances_d_for_e2 = np.linalg.norm(diffs_d_for_e2, axis=1)

            valid_mask_e2 = distances_d_for_e2 > self.epsilon
            if np.any(valid_mask_e2):
                ratios_e2 = (
                    dist_d1_full[valid_mask_e2, 1] / distances_d_for_e2[valid_mask_e2]
                )
                E2 = float(np.mean(ratios_e2))
            else:
                E2 = np.nan
        else:
            E2 = np.nan

        return E1, E2

    def estimate(self, x: np.ndarray[np.floating]):

        if x is None:
            raise ValueError("Input array must not be None")

        x = np.asarray(x)

        if x.ndim not in (1, 2):
            raise ValueError("Input array must be 1D or 2D: (n,) or (n, m)")

        match self.optimum_selection_method:
            case OptimalDimensionSelectionMethod.MIN:

                cao_stats = [
                    self._cao(x, d)
                    for d in tqdm(
                        range(1, self.max_dim + 1),
                        desc="Estimating dimension via Cao's method",
                        disable=not self.verbose,
                    )
                ]

                # Extract E1 (first element) for backward compatibility
                cao2 = np.array(
                    [stat[0] if isinstance(stat, tuple) else stat for stat in cao_stats]
                )
                cao2 = np.where(np.isnan(cao2), np.inf, cao2)

                min_index = np.argmin(cao2)

                if self.plot:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(8, 5))
                    plt.plot(range(1, self.max_dim + 1), cao2, marker="o")
                    plt.title("Cao's E1 Statistic vs Embedding Dimension")
                    plt.xlabel("Embedding Dimension (d)")
                    plt.ylabel("Cao's E1 Statistic")
                    plt.xticks(range(1, self.max_dim + 1))
                    plt.grid(True)
                    plt.axvline(
                        x=min_index + 1,
                        color="r",
                        linestyle="--",
                        label="Selected Dimension",
                    )
                    plt.legend()
                    plt.show()

                return min_index + 1

            case OptimalDimensionSelectionMethod.FIRST_LOC_MIN:

                if self.plot:
                    history = []

                    def record_history(d, val):
                        history.append((d, val))

                    def plot_history():
                        import matplotlib.pyplot as plt_local

                        if history:
                            dims, vals = zip(*history)
                            plt_local.figure(figsize=(8, 5))
                            plt_local.plot(dims, vals, marker="o")
                            plt_local.title("Cao's E1 Statistic vs Embedding Dimension")
                            plt_local.xlabel("Embedding Dimension (d)")
                            plt_local.ylabel("Cao's E1 Statistic")
                            plt_local.xticks(range(1, self.max_dim + 1))
                            plt_local.grid(True)
                            plt_local.show()

                prev_cao2 = None

                for d in tqdm(
                    range(1, self.max_dim + 1),
                    desc="Estimating dimension via Cao's method",
                    disable=not self.verbose,
                ):
                    result = self._cao(x, d)
                    val = result[0] if isinstance(result, tuple) else result

                    if self.plot:
                        record_history(d, val)

                    if np.isnan(val):
                        continue

                    if prev_cao2 is not None and val > prev_cao2:
                        if self.plot:
                            plot_history()
                        return d - 1

                    prev_cao2 = val

                if self.plot:
                    plot_history()

                return self.max_dim

            case OptimalDimensionSelectionMethod.E1_E2_COMBINED:

                cao_stats = [
                    self._cao(x, d)
                    for d in tqdm(
                        range(1, self.max_dim + 1),
                        desc="Estimating dimension via Cao's method (E1+E2)",
                        disable=not self.verbose,
                    )
                ]

                e1_values = np.array(
                    [stat[0] if isinstance(stat, tuple) else stat for stat in cao_stats]
                )
                e2_values = np.array(
                    [
                        stat[1] if isinstance(stat, tuple) else np.nan
                        for stat in cao_stats
                    ]
                )

                # Handle NaN values
                e1_values = np.where(np.isnan(e1_values), np.inf, e1_values)
                e2_values = np.where(np.isnan(e2_values), np.inf, e2_values)

                # Find optimal dimension using combined criteria
                optimal_dim = self._find_optimal_dimension_e1_e2(e1_values, e2_values)

                if self.plot:
                    self._plot_e1_e2(e1_values, e2_values, optimal_dim)

                return optimal_dim

            case _:
                raise ValueError("Unknown optimum selection method")

    def _find_optimal_dimension_e1_e2(
        self, e1_values: np.ndarray, e2_values: np.ndarray
    ) -> int:
        """
        Find optimal embedding dimension using both E1 and E2 statistics.

        The strategy is:
        1. E1 should reach a plateau (stop changing significantly) at the optimal dimension
        2. E2 should be significantly different from 1 for deterministic data
        3. Look for the dimension where E1 stabilizes and E2 indicates deterministic structure
        """

        # Calculate the relative change in E1 (should be small at optimal dimension)
        e1_changes = np.abs(np.diff(e1_values)) / (e1_values[:-1] + self.epsilon)

        # For deterministic data, E2 should be noticeably different from 1
        # For random data, E2 should be close to 1
        e2_deviation_from_1 = np.abs(e2_values - 1.0)

        # Find dimensions where E1 change is small (plateau)
        change_threshold = 0.1  # 10% change threshold
        stable_e1_mask = e1_changes < change_threshold

        # Find dimensions where E2 indicates deterministic structure
        # (significantly different from 1)
        e2_threshold = 0.1  # E2 should differ from 1 by at least 0.1
        deterministic_mask = e2_deviation_from_1 > e2_threshold

        # Look for the first dimension where both criteria are met
        for i in range(min(len(stable_e1_mask), len(deterministic_mask))):
            if stable_e1_mask[i] and deterministic_mask[i]:
                return (
                    i + 2
                )  # +2 because we start from d=1 and diff reduces length by 1

        # Fallback: if no clear optimal dimension found, use minimum E1
        return int(np.argmin(e1_values)) + 1

    def _plot_e1_e2(
        self, e1_values: np.ndarray, e2_values: np.ndarray, optimal_dim: int
    ):
        """Plot both E1 and E2 statistics."""
        import matplotlib.pyplot as plt

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        dims = range(1, len(e1_values) + 1)

        # Plot E1
        ax1.plot(dims, e1_values, marker="o", label="E1", color="blue")
        ax1.set_title("Cao's E1 Statistic vs Embedding Dimension")
        ax1.set_xlabel("Embedding Dimension (d)")
        ax1.set_ylabel("E1 Statistic")
        ax1.grid(True)
        ax1.axvline(
            x=optimal_dim, color="r", linestyle="--", label="Selected Dimension"
        )
        ax1.legend()

        # Plot E2
        ax2.plot(dims, e2_values, marker="s", label="E2", color="green")
        ax2.axhline(
            y=1.0, color="gray", linestyle=":", alpha=0.7, label="E2=1 (random)"
        )
        ax2.set_title("Cao's E2 Statistic vs Embedding Dimension")
        ax2.set_xlabel("Embedding Dimension (d)")
        ax2.set_ylabel("E2 Statistic")
        ax2.grid(True)
        ax2.axvline(
            x=optimal_dim, color="r", linestyle="--", label="Selected Dimension"
        )
        ax2.legend()

        plt.tight_layout()
        plt.show()


def estimate_dimension(x: np.ndarray[np.floating], estimator: DimensionEstimator):
    """
    Estimate the dimension of the input data using the specified estimator.

    Parameters:
        x (np.ndarray): Input data as a 1D numpy array.
        estimator (DimensionEstimator): An instance of a DimensionEstimator subclass.

    Returns:
        int: Estimated dimension.
    """
    return estimator.estimate(x)
