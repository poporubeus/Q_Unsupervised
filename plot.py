import matplotlib.pyplot as plt
import numpy as np


def plot_classical_kmeans(data: np.array, centroids: np.array, kmeans_labels) -> plt.figure:
    fig = plt.figure(figsize=(7, 6))
    for i, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
        cluster_points = data[kmeans_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", c=color, alpha=0.5)
    plt.scatter(centroids[0, 0], centroids[0, 1], label="Centroid", c='tab:blue', s=200, marker='+', alpha=1, linewidths=2.5)
    plt.scatter(centroids[1, 0], centroids[1, 1], label="Centroid", c='tab:orange', s=200, marker='+', alpha=1, linewidths=2.5)
    plt.scatter(centroids[2, 0], centroids[2, 1], label="Centroid", c='tab:green', s=200, marker='+', alpha=1, linewidths=2.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title('Classical K-Means Clustering')
    plt.legend(loc="best")
    return fig


def plot_quantum_clusters(data: np.array, clusters: np.array, centroids: np.array) -> plt.figure():
    """
    Plot datapoints colored according to their assigned quantum clusters.
    :param data: (np.array) Data to be plotted;
    :param clusters: (np.array) Clusters;
    :param centroids: (np.array) Centroids;
    :return:
    """

    fig = plt.figure(figsize=(7, 6))
    for cluster_label in range(len(centroids)):
        cluster_points = data[clusters == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_label}", alpha=0.5)
    plt.scatter(centroids[0, 0], centroids[0, 1], color='tab:blue', marker='x', label='Centroid', s=200, linewidths=2.5, alpha=1)
    plt.scatter(centroids[1, 0], centroids[1, 1], color='tab:orange', marker='x', label='Centroid', s=200, linewidths=2.5, alpha=1)
    plt.scatter(centroids[2, 0], centroids[2, 1], color='tab:green', marker='x', label='Centroid', s=200, linewidths=2.5, alpha=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Quantum K-Means Clustering')
    plt.legend(loc="best")
    return fig


def plot_classical_vs_quantum_centroids(c_centroids, q_centroids) -> plt.figure():
    """
    Plot quantum and classical centroids.

    :param c_centroids: (np.array) Classical centroids;
    :param q_centroids: (np.array) Quantum centroids;
    :return: fig: (plt.figure).
    """
    fig = plt.figure()
    plt.scatter(q_centroids[:, 0], q_centroids[:, 1], label="quantum", c="red", marker="+", s=200, linewidths=3)
    plt.scatter(c_centroids[:, 0], c_centroids[:, 1], label="centroid", c="royalblue", marker="x", s=200, linewidths=3)
    plt.legend()
    return fig
