from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


class DataLoading:
    def __init__(self, seed: int, min_features: float, max_features: float, clusters: int, n_features: int, n_elements: int) -> None:
        self.seed = seed
        self.min_features = min_features
        self.max_features = max_features
        self.clusters = clusters
        self.n_features = n_features
        self.n_elements = n_elements

    def GetPoints(self, rescaling: bool) -> tuple:
        X, y_truth = make_blobs(n_samples=self.n_elements, centers=self.clusters,
                                n_features=self.n_features, random_state=self.seed)
        if rescaling == True:
            scaler = MinMaxScaler((self.min_features, self.max_features))
            X_rescaled = scaler.fit_transform(X)
            return X_rescaled, y_truth
        else:
            return X, y_truth


def PLotData(data: np.array, marker_shape: str, marker_color: str, marker_edge: str):
    plt.scatter(data[:, 0], data[:, 1], c=marker_color, edgecolors=marker_edge, marker=marker_shape)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title('Data')
    return plt.show()
