import numpy as np
import pennylane as qml
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

k = 3  # Number of clusters
device = qml.device("default.qubit", wires=3)
X, y = make_blobs(n_samples=100, centers=k, n_features=2, random_state=0)
num_params = 3
params = np.random.rand(num_params)

@qml.qnode(device=device, shots=1024)
def DistanceEstimationVariational(datapoint, centroid, params):
    qml.Hadamard(wires=0)
    qml.Rot(datapoint[0], datapoint[1], np.pi, wires=1)
    qml.Rot(centroid[0]*params[0], centroid[1]*params[1], np.pi*params[2], wires=2)
    qml.CSWAP(wires=[0, 1, 2])
    qml.Hadamard(wires=0)
    measure = qml.expval(qml.PauliZ(wires=0))
    return measure

def initialize_centroids(data, k):
    centroids_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[centroids_indices]
    return centroids

def loss(data, centroids, distances, params, theta=0.6):
    distance_loss = np.mean(distances)
    output = [DistanceEstimationVariational(datapoint=data[i], centroid=centroids[i], params=params) for i in range(len(data))]
    total_loss = distance_loss + theta * np.mean(output)
    return total_loss

def optimize(data, k, max_iter=300, threshold=0.0001):
    centroids = initialize_centroids(data, k)
    opt = qml.AdagradOptimizer(stepsize=0.1)
    costs = []
    for iteration in range(max_iter):
        distances = calculate_distances(data, centroids)
        params, cost = opt.step_and_cost(lambda params: loss(data, centroids, distances, params, theta=0.6), params)
        centroids = update(data, clusters, k)
        costs.append(cost)
        print(f"It. number {iteration} --- cost: ", cost)

    return clusters, centroids, params

def calculate_distances(data, centroids):
    distances = []
    for point in data:
        point_distances = [DistanceEstimationVariational(datapoint=point, centroid=centroid, params=params) for centroid in centroids]
        distances.append(point_distances)
    return np.array(distances)

def assign_clusters(distances):
    return np.argmin(distances, axis=1)

def update(data, clusters, k):
    centroids = []
    for i in range(k):
        cluster_point = data[clusters == i]
        if len(cluster_point) > 0:
            new_centroid = np.mean(cluster_point, axis=0)
        else:
            new_centroid = data[np.random.choice(len(data))]
        centroids.append(new_centroid)
    return np.array(centroids)


def check_convergence(centroids_prev, centroids_current, threshold):
    """Check convergence based on the distance between centroids."""
    distances = np.linalg.norm(centroids_prev - centroids_current, axis=1)
    return np.all(distances < threshold)
def kmeans_quantum(data, k, max_iter=300, threshold=0.0001):
    centroids = initialize_centroids(data, k)
    centroids_prev = centroids.copy()

    for _ in range(max_iter):
        distances = calculate_distances(data, centroids)
        clusters = assign_clusters(distances)
        centroids = update(data, clusters, k)
        if check_convergence(centroids_prev, centroids, threshold):
            break

        centroids_prev = centroids.copy()

    return clusters, centroids

def plot_clusters(data, clusters, centroids):
    for cluster_label in range(len(centroids)):
        cluster_points = data[clusters == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_label}")
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids', s=200)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Quantum K-Means Clustering')
    plt.legend()
    plt.show()

clusters, centroids = kmeans_quantum(X, k)
plot_clusters(X, clusters, centroids)
