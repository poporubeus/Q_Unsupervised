import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

plt.style.use('default')
seed = np.random.seed(13)

k = 3  # Number of clusters
X, y = make_blobs(n_samples=100, centers=k, n_features=2, random_state=seed)

scaler = MinMaxScaler((0, 3*np.pi/4))
X = scaler.fit_transform(X)

plt.title("Original plot")
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
