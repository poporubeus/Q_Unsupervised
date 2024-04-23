import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
'''from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler


seed = np.random.seed(13)
shots = 100
k = 3  # Number of clusters
wires = 3
device = qml.device("default.qubit", wires=2)
X, y = make_blobs(n_samples=100, centers=k, n_features=2, random_state=seed)



### Riscalare le features separatamente
scaler = MinMaxScaler((0, 3*np.pi/2))
X = scaler.fit_transform(X)


def initialize_centroids(data, k):
    """
    k centroids randomly initialized by the user.
    data points are assumed to come from sklearn dataset.
    """
    centroids_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[centroids_indices]
    return centroids


c = initialize_centroids(X, k)

@qml.qnode(device=device, shots=shots)
def Qcircuit_encoding(x):
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    return qml.state()


fid_list = []
for x in range(len(X)):  # Iterate over the data points
    rho_data = Qcircuit_encoding(X[x])
    rho_k = Qcircuit_encoding(c[y[x]])  # Use the cluster index to choose the centroid
    fid_list.append(np.round(1-qml.math.fidelity(rho_data, rho_k),4))

fidelity = np.array(fid_list)
print("Fidelity for each data point and its centroid:", fidelity)
'''

#device = qml.device("default.qubit", wires=3)
'''

def oracle():
    qml.PauliX(wires=0)
    qml.Hadamard(wires=2)
    qml.Toffoli(wires=[0,1,2])
    qml.Hadamard(wires=2)
    qml.X(wires=0)


def Init():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)


def GroverSearchDiffusion():
    oracle()
    Init()

    qml.PauliX(wires=0)
    qml.PauliX(wires=1)
    qml.PauliX(wires=2)

    qml.Hadamard(wires=2)
    qml.Toffoli(wires=[0,1,2])
    qml.Hadamard(wires=2)

    qml.PauliX(wires=0)
    qml.PauliX(wires=1)
    qml.PauliX(wires=2)
    Init()


@qml.qnode(device=device, shots=10)
def circuit_running():
    Init()
    GroverSearchDiffusion()
    GroverSearchDiffusion()
    return qml.probs([0, 1, 2])

results = np.array(circuit_running())

y = np.real(results)
bit_strings = [f"{x:0{3}b}" for x in range(len(y))]

plt.bar(bit_strings, y, color = "#70CEFF")

plt.xticks(rotation="vertical")
plt.xlabel("State label")
plt.ylabel("Probability Amplitude")
plt.title("States probabilities amplitudes")
plt.show()'''

num_qbits = 5
iterations=1
device = qml.device("default.qubit", wires=num_qbits)
omega = np.array([0,1])
wires = list(range(num_qbits))

def oracle(wires, omega):
    qml.FlipSign(omega, wires=wires[2:4])


@qml.qnode(device=device, shots=10)
def circuit():
    qml.PauliX(wires=3)
    qml.Snapshot("Before querying the Oracle")
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    '''oracle(wires=wires, omega=omega)'''
    qml.Snapshot("After querying the Oracle")
    for it in range(iterations):
        '''for omg in omega:'''
        oracle(wires, omega)
        qml.templates.GroverOperator(wires[:2])

    return qml.probs(wires=[0,1,2])

print(circuit())