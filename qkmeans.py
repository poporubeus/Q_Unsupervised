import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pennylane as qml
import math


quantum_device = qml.device("default.qubit", wires=3, shots=1000)
@qml.qnode(quantum_device)
def qcircuit(phi):
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.U3(theta=phi[0], phi=np.pi, delta=np.pi, wires=1)
    qml.U3(theta=phi[1], phi=np.pi, delta=np.pi, wires=2)
    qml.CSWAP(wires=[0, 1, 2])
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))


def get_theta(d):
    x = d[0]
    y = d[1]
    theta = 2*math.acos((x+y)/2.0)
    return theta


def Find_distance(x, y):
    theta1 = get_theta(x)
    theta2 = get_theta(y)
    thetas = [theta1, theta2]
    results = qcircuit(thetas)
    return results


def get_data(n, k, std):
    data = make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=std, random_state=100)
    points = data[0]
    centers = data[1]
    return points, centers


def draw_plot(points,centers,label=True):
    if label==False:
        plt.scatter(points[:,0], points[:,1])
    else:
        plt.scatter(points[:,0], points[:,1], c=centers, cmap='viridis')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

def plot_centroids(centers):
    plt.scatter(centers[:,0], centers[:,1], marker='s', s=100)
    plt.xlim(0,1)
    plt.ylim(0,1)


def initialize_centers(points,k):
    return points[np.random.randint(points.shape[0],size=k),:]

def get_distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)*(p1-p2)))


def find_nearest_neighbour(points, centroids):
    n = len(points)
    k = centroids.shape[0]
    centers = np.zeros(n)
    for i in range(n):
        min_dis = 10000
        ind = 0
        for j in range(k):
            temp_dis = Find_distance(points[i, :], centroids[j, :])
            if temp_dis < min_dis:
                min_dis = temp_dis
                ind = j
        centers[i] = ind
    return centers


def find_centroids(points, centers):
    n = len(points)
    k = int(np.max(centers)) + 1
    centroids = np.zeros([k, 2])
    for i in range(k):
        # print(points[centers==i])
        centroids[i, :] = np.average(points[centers == i])
    return centroids


def preprocess(points):
    n = len(points)
    x = 30.0 * np.sqrt(2)
    for i in range(n):
        points[i, :] += 15
        points[i, :] /= x
    return points


n = 100  # number of data points
k = 4  # Number of centers
std = 2  # std of datapoints

points, o_centers = get_data(n, k, std)  # dataset

points = preprocess(points)  # Normalize dataset
plt.figure()
draw_plot(points, o_centers, label=False)




def plot_data_and_centroids(points, centers, centroids, label=True):
    plt.figure(figsize=(6, 5))
    if label==False:
        plt.scatter(points[:,0], points[:,1])
    else:
        plt.scatter(points[:,0], points[:,1], c=centers, cmap='viridis', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='s', s=100, cmap='viridis')
    plt.xlim(0, 1)
    plt.ylim(0, 1)



centroids = initialize_centers(points, k)  # Intialize centroids

for i in range(5):
    centers = find_nearest_neighbour(points, centroids)  # find nearest centers
    plot_data_and_centroids(points, centers, centroids, True)
    '''draw_plot(points, centers)
    plot_centroids(centroids)'''
    centroids = find_centroids(points, centers)
    plt.show()