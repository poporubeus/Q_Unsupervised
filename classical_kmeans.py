from data_creation import X, clusters, seed
from sklearn.cluster import KMeans


threshold = 0.0001


kmeans = KMeans(n_clusters=clusters, random_state=seed, n_init="auto", tol=threshold, max_iter=20)
kmeans.fit(X)

clusters_predicted = kmeans.predict(X)

classical_cluster_centers = kmeans.cluster_centers_
