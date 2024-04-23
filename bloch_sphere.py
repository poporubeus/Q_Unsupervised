import numpy as np
import pennylane as qml
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import qutip


seed = np.random.seed(13)
shots = 100
k = 3  # Number of clusters
wires = 3
device = qml.device("default.qubit", wires=wires)
X, y = make_blobs(n_samples=100, centers=k, n_features=2, random_state=seed)



### Riscalare le features separatamente
scaler = MinMaxScaler((0, 3*np.pi/2))
X = scaler.fit_transform(X)


@qml.qnode(device=device, shots=shots)
def DistanceEstimation(datapoint, centroid):
    '''qml.AmplitudeEmbedding(features=(datapoint), wires=1, normalize=True)
    qml.AmplitudeEmbedding(features=(centroid), wires=2, normalize=True)'''
    qml.Hadamard(wires=0)
    qml.RX(datapoint[0], wires=1)
    qml.RY(datapoint[1], wires=1)
    qml.RX(centroid[0], wires=2)
    qml.RY(centroid[1], wires=2)
    qml.CSWAP(wires=[0, 1, 2])
    qml.Hadamard(wires=0)
    measure = [qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliZ(wires=0))]
    return measure

## Costruire la sfera di Bloch
def initialize_centroids(data, k):
    """
    k centroids randomly initialized by the user.
    data points are assumed to come from sklearn dataset.
    """
    centroids_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[centroids_indices]
    return centroids


@qml.qnode(device=device, shots=shots)
def DataQuantum(datapoint):
    '''qml.AmplitudeEmbedding(features=(datapoint), wires=1, normalize=True)
    qml.AmplitudeEmbedding(features=(centroid), wires=2, normalize=True)'''
    qml.RX(datapoint[0], wires=0)
    qml.RY(datapoint[1], wires=0)
    measure = [qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliZ(wires=0))]
    return measure

'''c = initialize_centroids(X, k)


q_centroid = np.zeros(shape=[3, 3])
q_data = np.zeros(shape=[len(X), 3])
for c_index in range(len(q_centroid)):
    q_centroid[c_index, :] = DataQuantum(datapoint=c[c_index])

# Calculate quantum states for data points
for data_index in range(len(q_data)):
    q_data[data_index, :] = DataQuantum(datapoint=X[data_index])'''

#### Se si implementa il Grover, ricordarsi che forse vogliono direttamente la fidelity
def calculate_distances(data, centroids):
    distances = []
    for point in data:
        point_distances = [1-DistanceEstimation(datapoint=point, centroid=centroid)[2] for centroid in centroids]
        distances.append(point_distances)
    return np.array(distances)


# THIS IS ONLY A TEST #################
def oracle():
    n_wires = 3
    wires = list(range(n_wires))
    qml.Hadamard(wires=wires[-1])
    qml.Toffoli(wires=wires)
    qml.Hadamard(wires=wires[-1])


@qml.qnode(device=device)
def GroverSubroutine(num_iter=2):
    for wire in wires:
        qml.Hadamard(wires=wire)

    for it in range(num_iter):
        oracle()
        qml.templates.GroverOperator(wires=wires)
    return qml.probs(wires)
###################################


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


def kmeans_quantum(data, k, max_iter=30, threshold=0.00001):
    """Perform k-means clustering with quantum subroutines."""
    centroids = initialize_centroids(data, k)
    centroids_prev = centroids.copy()

    for it in range(max_iter):
        print('Iteration {}'.format(it + 1))
        distances = calculate_distances(data, centroids)
        clusters = assign_clusters(distances)
        centroids = update(data, clusters, k)
        if check_convergence(centroids_prev, centroids, threshold):
            break
        centroids_prev = centroids.copy()
    return clusters, centroids


clusters, centroids = kmeans_quantum(X, k)

def plot_points_on_sphere(q_data, q_centroid):
    b = qutip.Bloch()
    b.add_points([q_data[:, 0], q_data[:, 1], q_data[:, 2]], alpha=0.4, colors="#CC6600")
    b.add_points([q_centroid[:, 0], q_centroid[:, 1], q_centroid[:, 2]], colors="blue")
    b.point_marker = ["o"] * len(q_data)
    b.show()
    return plt.show(block=True)



q_centroid = np.zeros(shape=[3, 3])
q_data = np.zeros(shape=[len(X), 3])
for c_index in range(len(q_centroid)):
    q_centroid[c_index, :] = DataQuantum(datapoint=centroids[c_index])

# Calculate quantum states for data points
for data_index in range(len(q_data)):
    q_data[data_index, :] = DataQuantum(datapoint=X[data_index])

b = qutip.Bloch()
b.point_color = ["r", "b", "g"]
b.point_marker = ['o', 'o', 'o', 's', 's', 's']
b.add_points([q_data[np.where(clusters == 0), 0][0], q_data[np.where(clusters == 0), 1][0], q_data[np.where(clusters == 0), 2][0]], alpha=0.4)
b.add_points([q_data[np.where(clusters == 1), 0][0], q_data[np.where(clusters == 1), 1][0], q_data[np.where(clusters == 1), 2][0]], alpha=0.4)
b.add_points([q_data[np.where(clusters == 2), 0][0], q_data[np.where(clusters == 2), 1][0], q_data[np.where(clusters == 2), 2][0]], alpha=0.4)
#b.add_points([q_data[np.where(y == 0), 0][0], q_data[np.where(y == 0), 1][0], q_data[np.where(y == 0), 2][0]], alpha=0.4)
#b.add_points([q_data[np.where(y == 1), 0][0], q_data[np.where(y == 1), 1][0], q_data[np.where(y == 1), 2][0]], alpha=0.4)
#b.add_points([q_data[np.where(y == 2), 0][0], q_data[np.where(y == 2), 1][0], q_data[np.where(y == 2), 2][0]], alpha=0.4)
b.add_points([q_centroid[0, 0], q_centroid[0, 1], q_centroid[0, 2]])
b.add_points([q_centroid[1, 0], q_centroid[1, 1], q_centroid[1, 2]])
b.add_points([q_centroid[2, 0], q_centroid[2, 1], q_centroid[2, 2]])
b.show()
plt.show(block=True)

plt.style.use('default')
plt.title("Original plot")
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

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

plot_clusters(X, clusters, centroids)

'''def plot_clustered_points_on_sphere(data, clusters, centroids):
    cluster_points_grouped = np.zeros([len(data), 3])
    for cluster_label in range(len(centroids)):
        cluster_points = data[clusters == cluster_label]
        cluster_points_grouped[cluster_label] = cluster_points.copy()
    b = qutip.Bloch()
    b.add_points([cluster_points_grouped[:, 0], cluster_points_grouped[:, 1], cluster_points_grouped[:, 2]])
    b.point_color(["b", "r", "g"])
    b.show()
    return plt.show(block=True)'''


'''def plot_clustered_points_on_sphere(data, clusters, centroids):
    k = len(centroids)
    num_features = data.shape[1]
    cluster_points_grouped = np.zeros([num_features, k])
    for cluster_label in range(k):
        cluster_points = data[clusters == cluster_label]
        cluster_points_grouped[:, cluster_label] = cluster_points.T 
    b = qutip.Bloch()
    b.add_points(cluster_points_grouped)
    b.point_color(["b", "r", "g"])  
    b.show()
    return plt.show(block=True)


plot_clustered_points_on_sphere(X, clusters, centroids)'''


print(clusters)


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
    b.add_points([q_data[np.where(clusters == 0), 0][0], q_data[np.where(clusters == 0), 1][0], q_data[np.where(clusters == 0), 2][0]], alpha=0.4)
    b.add_points([q_data[np.where(clusters == 1), 0][0], q_data[np.where(clusters == 1), 1][0], q_data[np.where(clusters == 1), 2][0]], alpha=0.4)
    b.add_points([q_data[np.where(clusters == 2), 0][0], q_data[np.where(clusters == 2), 1][0], q_data[np.where(clusters == 2), 2][0]], alpha=0.4)
    #b.add_points([q_data[np.where(y == 0), 0][0], q_data[np.where(y == 0), 1][0], q_data[np.where(y == 0), 2][0]], alpha=0.4)
    #b.add_points([q_data[np.where(y == 1), 0][0], q_data[np.where(y == 1), 1][0], q_data[np.where(y == 1), 2][0]], alpha=0.4)
    #b.add_points([q_data[np.where(y == 2), 0][0], q_data[np.where(y == 2), 1][0], q_data[np.where(y == 2), 2][0]], alpha=0.4)
    b.add_points([q_centroid[0, 0], q_centroid[0, 1], q_centroid[0, 2]])
    b.add_points([q_centroid[1, 0], q_centroid[1, 1], q_centroid[1, 2]])
    b.add_points([q_centroid[2, 0], q_centroid[2, 1], q_centroid[2, 2]])
    b.show()
    return plt.show(block=True)



