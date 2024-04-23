import numpy as np
import pennylane as qml
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
import qutip



seed = np.random.seed(3)

k = 3  # Number of clusters
device = qml.device("default.qubit", wires=3)
device2 = qml.device("default.qubit", wires=5)
X, y = make_blobs(n_samples=100, centers=k, n_features=2, random_state=seed)


### Riscalare le features separatamente
scaler = MinMaxScaler((-np.pi/2, 3*np.pi/4))
X = scaler.fit_transform(X)


def post_process(measurement_result):
    result = float(measurement_result) + 1
    return result/2

@qml.qnode(device=device, shots=1024)
def DistanceEstimation(datapoint, centroid):
    '''qml.AmplitudeEmbedding(features=(datapoint), wires=1, normalize=True)
    qml.AmplitudeEmbedding(features=(centroid), wires=2, normalize=True)'''
    qml.Hadamard(wires=0)
    qml.RX(datapoint[0], wires=1)
    qml.RY(datapoint[1], wires=1)
    qml.RX(centroid[0], wires=2)
    qml.RY(centroid[1], wires=2)
    '''qml.U3(datapoint[0], datapoint[1], np.pi, wires=1)
    qml.U3(centroid[0], centroid[1], np.pi, wires=2)'''
    qml.CSWAP(wires=[0, 1, 2])
    qml.Hadamard(wires=0)
    measure = [qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliZ(wires=0))]
    return measure



'''
@qml.qnode(device=device2, shots=1024)
def DistanceEstimation5Qubits(datapoint, centroid):
    qml.Hadamard(wires=0)
    qml.RX(datapoint[0], wires=1)
    qml.RY(datapoint[1], wires=2)
    qml.RX(centroid[0], wires=3)
    qml.RY(centroid[1], wires=4)
    qml.CSWAP(wires=[0, 1, 3])
    qml.CSWAP(wires=[0, 2, 4])
    qml.Hadamard(wires=0)
    measure = [qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliZ(wires=0))]
    return measure'''
## Costruire la sfera di Bloch



def initialize_centroids(data, k):
    """
    Qua i centroidi vengono presi a caso dai dati, pertanto sono già riscalati.
    :param data: actual datapoints that need to be encoded;
    :param k: number of clusters / centroids the user guesses;
    :return: centroids.
    """
    centroids_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[centroids_indices]
    return centroids


def calculate_distances(data, centroids):
    """
    Calcola la distanza tra ogni punto e centroide, utilizzando il quantum circuit.
    :param data: actual datapoints;
    :param centroids: centroids the user has guessed;
    :return: distances' array.
    """
    distances = []
    for point in data:
        point_distances = [1-DistanceEstimation(datapoint=point, centroid=centroid)[2] for centroid in centroids]
        distances.append(point_distances)
    return np.array(distances)

def assign_clusters(distances):
    """
    Qui viene assegnato ad ogni punto il cluster corrispondente alla minima distanza dal centoride.
    L'idea è realizzare un Grover che ricerchi e sostituisca di fatto questa funzione, garantendo uno speed-up.
    :param distances:
    :return:np.argmin(distances, axis=1).
    """
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


def kmeans_quantum(data, k, max_iter=20, threshold=0.0001):
    """Perform k-means clustering with quantum subroutines."""
    centroids = initialize_centroids(data, k)
    centroids_prev = centroids.copy()

    for it in range(max_iter):
        print(f"Iteration {it+1}")
        # Step 1: Calculate distances between data points and centroids
        distances = calculate_distances(data, centroids)

        clusters = assign_clusters(distances)
        centroids = update(data, clusters, k)

        if check_convergence(centroids_prev, centroids, threshold):
            break

        centroids_prev = centroids.copy()

    return clusters, centroids


#print(f"Using method {method}")


# Example usage:
data = X  # Example data
clusters, centroids = kmeans_quantum(data, k)


clusters[np.where(clusters == 0)] = 3
clusters[np.where(clusters == 2)] = 0
clusters[np.where(clusters == 3)] = 2

print("Cluster assignments:", clusters)
print("Final centroids (quantum):\n", centroids)

def plot_clusters(data, clusters, centroids):
    """Plot data points colored according to their assigned clusters."""
    for cluster_label in range(len(centroids)):
        cluster_points = data[clusters == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_label}")
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids', s=200)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Quantum K-Means Clustering')
    plt.legend()
    plt.show()

plot_clusters(data, clusters, centroids)

kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto", tol=0.0001, max_iter=20)
kmeans.fit(X)
classical_cluster_centers = kmeans.cluster_centers_
print("Classical centroids: ", classical_cluster_centers)

def plot_classical_kmeans(data, centroids):
    for i, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
        cluster_points = data[kmeans.labels_ == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", c=color)
    plt.scatter(centroids[:, 0], centroids[:, 1], label="Centroids", c="black", s=200, marker='x')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title('Classical K-Means Clustering')
    plt.legend()
    plt.show()

plot_classical_kmeans(data, classical_cluster_centers)

DIFF_QC = np.subtract(classical_cluster_centers, centroids)
print("Difference between classical and quantum centroid-coordinates: ", DIFF_QC)

def classical_vs_quantum_centroids(c_centroids, q_centroids):
    plt.scatter(q_centroids[:, 0], q_centroids[:, 1], label="quantum", c="red", marker="x", s=200, linewidths=3)
    plt.scatter(c_centroids[:, 0], c_centroids[:, 1], label="centroid", c="royalblue", marker="x", s=200, linewidths=3)
    plt.legend()
    plt.show()

classical_vs_quantum_centroids(classical_cluster_centers, centroids)
ypredicted_classical = kmeans.predict(X)

print(y[:10])
print(ypredicted_classical[:10])

accuracy = adjusted_rand_score(y, ypredicted_classical)
print("Classical clusters: ", kmeans.labels_)
print("Accuracy classical model: ", accuracy)
accuracy_quantum = adjusted_rand_score(y, clusters)
print("Accuracy quantum model: ", accuracy_quantum)


@qml.qnode(device=device, shots=1024)
def DataQuantum(datapoint):
    '''qml.AmplitudeEmbedding(features=(datapoint), wires=1, normalize=True)
    qml.AmplitudeEmbedding(features=(centroid), wires=2, normalize=True)'''
    qml.RX(datapoint[0], wires=0)
    qml.RY(datapoint[1], wires=0)
    measure = [qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliZ(wires=0))]
    return measure

q_centroid = np.zeros(shape=[3, 3])
q_data = np.zeros(shape=[len(X), 3])
for c_index in range(len(q_centroid)):
    q_centroid[c_index, :] = DataQuantum(datapoint=centroids[c_index])

# Calculate quantum states for data points
for data_index in range(len(q_data)):
    q_data[data_index, :] = DataQuantum(datapoint=X[data_index])


def plot_default_data_onBloch(q_data, clusters, q_centroid):
    b = qutip.Bloch()
    '''b.point_color = ["r", "b", "g"]
    b.point_marker = ['o', 'o', 'o', 's', 's', 's']'''
    b.add_points([q_data[np.where(clusters == 0), 0][0], q_data[np.where(clusters == 0), 1][0], q_data[np.where(clusters == 0), 2][0]], alpha=0.4, colors="r")
    b.add_points([q_data[np.where(clusters == 1), 0][0], q_data[np.where(clusters == 1), 1][0], q_data[np.where(clusters == 1), 2][0]], alpha=0.4, colors="b")
    b.add_points([q_data[np.where(clusters == 2), 0][0], q_data[np.where(clusters == 2), 1][0], q_data[np.where(clusters == 2), 2][0]], alpha=0.4, colors="g")
    #b.add_points([q_data[np.where(y == 0), 0][0], q_data[np.where(y == 0), 1][0], q_data[np.where(y == 0), 2][0]], alpha=0.4)
    #b.add_points([q_data[np.where(y == 1), 0][0], q_data[np.where(y == 1), 1][0], q_data[np.where(y == 1), 2][0]], alpha=0.4)
    #b.add_points([q_data[np.where(y == 2), 0][0], q_data[np.where(y == 2), 1][0], q_data[np.where(y == 2), 2][0]], alpha=0.4)
    b.add_points([q_centroid[0, 0], q_centroid[0, 1], q_centroid[0, 2]], colors="r")
    b.add_points([q_centroid[1, 0], q_centroid[1, 1], q_centroid[1, 2]], colors="b")
    b.add_points([q_centroid[2, 0], q_centroid[2, 1], q_centroid[2, 2]], colors="g")
    b.show()
    return plt.show(block=True)
plot_default_data_onBloch(q_data, clusters, q_centroid)


