import numpy as np
import pennylane as qml
from data_creation import X, seed, clusters


n_qubits = 3
shots = 2048
device = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(device=device, shots=shots)
def DistanceEstimation(datapoint: np.array, centroid: np.array) -> list:
    """
    Calculates the distance between each point in the dataset and the
    centroid of the cluster we guess. This distance is related to the
    fidelity between the quantum states representative of the centroid
    and the datapoints.

    :param datapoint: (np.array) The datapoints to embed in the circuit;
    :param centroid: (np.array) The centroid of the cluster to embed in the circuit;
    :return: measure: (list) List of distances between the datapoints and the centroids
    after quantum circuit measurement of the expectation value along PauliX,
    PaulyY and PauliZ operators respectively.
    """
    qml.Hadamard(wires=0)
    qml.RX(datapoint[0], wires=1)
    qml.RY(datapoint[1], wires=1)
    qml.RX(centroid[0], wires=2)
    qml.RY(centroid[1], wires=2)
    qml.CSWAP(wires=[0, 1, 2])
    qml.Hadamard(wires=0)
    measure = [qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliZ(wires=0))]
    return measure


class QuantumKmeans:
    def __init__(self, seed: int, k: int, data: np.array) -> None:
        """
        Class which implements the quantum version of the K-means algorithm for
        clustering unlabelled data. This is a hybrid implementation, where the
        distance between each point in the data and the centroid is calculated
        by using a quantum circuit, in particular, getting the fidelity between
        quantum states representing points and centroids.
        Rather, the cluster assignment is done classically.

        :param seed: (int) The random seed for reproducibility;
        :param k: (int) The number of clusters we guess;
        :param data: (np.array) The datapoints;
        :return None.
        """
        self.seed = seed
        self.k = k
        self.data = data

    @staticmethod
    def check_convergence(centroids_prev: float, centroids_current: float, threshold: float) -> np.array:
        """
        Check convergence based on the distance between centroids.

        :param centroids_prev: (float) Centroid at the previous iteration;
        :param centroids_current: (float) Centroid at the current iteration;
        :param threshold: (float) Threshold for convergence;
        :return: (np.array) True if the convergence is achieved, False otherwise.
        """
        distances = np.linalg.norm(centroids_prev - centroids_current, axis=1)
        return np.all(distances < threshold)

    @staticmethod
    def initialize_centroids(data: np.array, k: int) -> np.array:
        """
        Initialize the centroids by randomly selecting k centroids from the dataset.

        :param data: (np.array) The datapoints;
        :param k: (int) The number of clusters we guess
        :return: centroids (np.array) The centroids;
        """
        centroids_indices = np.random.choice(len(data), k, replace=False)
        centroids = data[centroids_indices]
        return centroids

    @staticmethod
    def calculate_distances(data: np.array, centroids: np.array) -> np.array:
        """
        Call the distance calculation function and use the swap test to get the distances.

        :param data: (np.array) The datapoints;
        :param centroids: (np.array) Centroids the user has guessed;
        :return: np.array(distances) (np.array) Distances' array.
        """
        distances = []
        for point in data:
            point_distances = [1 - DistanceEstimation(datapoint=point, centroid=centroid)[2] for centroid in centroids]
            distances.append(point_distances)
        return np.array(distances)

    @staticmethod
    def assign_clusters(distances: np.array) -> int:
        """
        Assign the current cluster label to the point which has the smaller distance between
        the centroid from which the distance is calculated.

        :param distances: (np.array) Distances between the points and centroids;
        :return: np.argmin(distances, axis=1): (int) The index of the smallest distance.
        """
        return np.argmin(distances, axis=1)

    @staticmethod
    def update(clusters: int, k: int, data: np.array) -> np.array:
        """
        Update the cluster's centroid by calculating the geometrical cluster's centre by
        taking the mean between all the points which belong to the cluster if there are
        points inside the cluster, otherwise the centroid is taken randomly.

        :param clusters: (int) The cluster's labels;
        :param k: (int) The number of clusters we guess;
        :param data: (np.array) The datapoints;
        :return: centroids_arr: (np.array) The new centroids updated.
        """
        centroids = []
        for i in range(k):
            cluster_point = data[clusters == i]
            if len(cluster_point) > 0:
                new_centroid = np.mean(cluster_point, axis=0)
            else:
                new_centroid = data[np.random.choice(len(data))]
            centroids.append(new_centroid)
        centroids_arr = np.array(centroids)
        return centroids_arr

    def kmeans_quantum(self, max_iter: int, threshold: float) -> tuple:
        """
        Perform k-means clustering with quantum circuit defined previously, and call
        all the method listed inside this class.

        :param max_iter: (int) The maximum number of iteration which the algorithm will run;
        :param threshold: (float) The threshold to stop the algorithm;
        :return: clusters, centroids: (tuple) The clusters and their centroids.
        """
        centroids = self.initialize_centroids(self.data, self.k)
        centroids_prev = centroids.copy()
        for it in range(max_iter):
            print(f"Iteration {it + 1}")
            distances = self.calculate_distances(self.data, centroids)
            clusters = self.assign_clusters(distances)
            centroids = self.update(clusters, self.k, self.data)
            if self.check_convergence(centroids_prev, centroids, threshold):
                break
            centroids_prev = centroids.copy()
        return clusters, centroids


q_kmeans = QuantumKmeans(seed=seed, k=clusters, data=X)
clusters, centroids = q_kmeans.kmeans_quantum(max_iter=20, threshold=0.0001)