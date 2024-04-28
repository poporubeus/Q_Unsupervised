from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from qkmeans_class import QuantumKmeans
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, rand_score


clusters = 4
seed = 9
method = "Partial"

iris = load_iris()
max_scaling = np.pi/4
min_scaling = -np.pi
X = iris.data
y = iris.target

scaler = MinMaxScaler(feature_range=(0, np.pi/2))
X_scaled = scaler.fit_transform(X)
X = X_scaled

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.05)

q_kmeans = QuantumKmeans(seed=seed, k=clusters, data=X_train, kind=method)
#print("Clusters:\n", clusters, "\n", "Centroids:\n", centroids)
c_kmeans = KMeans(n_clusters=clusters, n_init="auto", random_state=seed, tol=0.0001, max_iter=40)

c_kmeans.fit(X_train)
c_clusters = c_kmeans.labels_
c_centroids = c_kmeans.cluster_centers_
q_clusters, q_centroids = q_kmeans.kmeans_quantum(max_iter=40, threshold=0.0001)


print("Quantum clusters:\n", q_clusters)
print("Classical clusters:\n", c_clusters)
'''q_clusters[np.where(q_clusters == 2)] = 4
q_clusters[np.where(q_clusters == 3)] = 2
q_clusters[np.where(q_clusters == 1)] = 3
q_clusters[np.where(q_clusters == 4)] = 1'''

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,7))
for classical_cluster_label in set(c_clusters):
    axs[0].scatter(X_train[c_clusters == classical_cluster_label, 0], X_train[c_clusters == classical_cluster_label, 1], s=50, label=f"Cluster {classical_cluster_label}", alpha=0.4)
axs[0].scatter(c_centroids[0, 0], c_centroids[0, 1], marker='x', s=200, c='tab:blue', linewidths=3)
axs[0].scatter(c_centroids[1, 0], c_centroids[1, 1], marker='x', s=200, c='tab:orange', linewidths=3)
axs[0].scatter(c_centroids[2, 0], c_centroids[2, 1], marker='x', s=200, c='tab:green', linewidths=3)
axs[0].scatter(c_centroids[3, 0], c_centroids[3, 1], marker='x', s=200, c='tab:red', linewidths=3)
axs[0].set_title("Classical Kmeans")
axs[0].legend()
for cluster_label in set(q_clusters):
    axs[1].scatter(X_train[q_clusters == cluster_label, 0], X_train[q_clusters == cluster_label, 1], s=50, label=f"Cluster {cluster_label}", alpha=0.4)
axs[1].scatter(q_centroids[0, 0], q_centroids[0, 1], marker='+', s=200, c='tab:blue', linewidths=3)
axs[1].scatter(q_centroids[1, 0], q_centroids[1, 1], marker='+', s=200, c='tab:orange', linewidths=3)
axs[1].scatter(q_centroids[2, 0], q_centroids[2, 1], marker='+', s=200, c='tab:green', linewidths=3)
axs[1].scatter(q_centroids[3, 0], q_centroids[3, 1], marker='+', s=200, c='tab:red', linewidths=3)
axs[1].set_title("Q-Kmeans")
axs[1].legend()
plt.show()

#print("New quantum clusters:\n", q_clusters)

quantum_rs, classical_rs = adjusted_rand_score(y_train, q_clusters), adjusted_rand_score(y_train, c_clusters)
print("Adjusted RS score: QUANTUM", quantum_rs)
print("Adjusted RS score: CLASSICAL", classical_rs)

print("RS score: QUANTUM", rand_score(y_train, q_clusters))
print("RS score: CLASSICAL", rand_score(y_train, c_clusters))

