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
        """Compute Cao's E2 statistic for dimension d.

        Supports 1D (n,) and 2D (n, m) inputs. For 2D inputs, constructs
        multivariate delay embeddings by concatenating all m features at each
        lag: embedding_d shape (n_vectors_d, d*m), embedding_{d+1} shape
        (n_vectors_d1, (d+1)*m).
        """
        if x.ndim == 1:
            n = x.shape[0]
            m = 1
        elif x.ndim == 2:
            n, m = x.shape
        else:
            return np.nan

        n_vectors_d = n - (d - 1) * self.delay
        n_vectors_d1 = n - d * self.delay

        if n_vectors_d1 < 2 or n_vectors_d <= 0:
            return np.nan

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
        if np.any(valid_mask):
            ratios = distances_d1[valid_mask] / distances_d[valid_mask]
            return float(np.mean(ratios))
        return np.nan

    def estimate(self, x: np.ndarray[np.floating]):

        if x is None:
            raise ValueError("Input array must not be None")

        x = np.asarray(x)

        if x.ndim not in (1, 2):
            raise ValueError("Input array must be 1D or 2D: (n,) or (n, m)")

        match self.optimum_selection_method:
            case OptimalDimensionSelectionMethod.MIN:

                cao2 = [
                    self._cao(x, d)
                    for d in tqdm(
                        range(1, self.max_dim + 1),
                        desc="Estimating dimension via Cao's method",
                        disable=not self.verbose,
                    )
                ]

                cao2 = np.array(cao2)
                cao2 = np.where(np.isnan(cao2), np.inf, cao2)

                min_index = np.argmin(cao2)

                if self.plot:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(8, 5))
                    plt.plot(range(1, self.max_dim + 1), cao2, marker="o")
                    plt.title("Cao's E2 Statistic vs Embedding Dimension")
                    plt.xlabel("Embedding Dimension (d)")
                    plt.ylabel("Cao's E2 Statistic")
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
                        import matplotlib.pyplot as plt

                        if history:
                            dims, vals = zip(*history)
                            plt.figure(figsize=(8, 5))
                            plt.plot(dims, vals, marker="o")
                            plt.title("Cao's E2 Statistic vs Embedding Dimension")
                            plt.xlabel("Embedding Dimension (d)")
                            plt.ylabel("Cao's E2 Statistic")
                            plt.xticks(range(1, self.max_dim + 1))
                            plt.grid(True)
                            plt.show()

                prev_cao2 = None

                for d in tqdm(
                    range(1, self.max_dim + 1),
                    desc="Estimating dimension via Cao's method",
                    disable=not self.verbose,
                ):
                    val = self._cao(x, d)

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
            case _:
                raise ValueError("Unknown optimum selection method")


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
