from data_creation import X, y
from classical_kmeans import classical_cluster_centers, clusters_predicted, kmeans
from sklearn.metrics import adjusted_rand_score
from qkmeans_class import clusters, centroids
from plot import *


quantum_centroids = centroids
quantum_clusters = clusters

print("Cluster assignments:\n", quantum_clusters)
print("Final centroids (quantum):\n", quantum_centroids)
print("Classical centroids:\n", classical_cluster_centers)


DIFF_QC = np.subtract(classical_cluster_centers, quantum_centroids)
print("Difference between classical and quantum centroid-coordinates: ", DIFF_QC)


accuracy = adjusted_rand_score(y, clusters_predicted)
print("Classical clusters: ", kmeans.labels_)
print("Accuracy classical model: ", accuracy)
accuracy_quantum = adjusted_rand_score(y, clusters)
print("Accuracy quantum model: ", accuracy_quantum)


plot_quantum_clusters(data=X, clusters=quantum_clusters, centroids=quantum_centroids)
plot_classical_kmeans(data=X, centroids=classical_cluster_centers, kmeans_labels=kmeans.labels_)
plot_classical_vs_quantum_centroids(c_centroids=classical_cluster_centers, q_centroids=quantum_centroids)
plt.show()