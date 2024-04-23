import matplotlib.pyplot as plt
import qutip
import pennylane as qml
from qkmeans_class import shots, device
import numpy as np
from data_creation import X
from comparison import quantum_centroids, quantum_clusters


@qml.qnode(device=device, shots=shots)
def SingleQubitCircuit(datapoint: np.array) -> list:
    """
    Create a circuit to plot the datapoints on te Bloch sphere.
    :param datapoint: (np.array) The datapoints;
    :return: measure: (list) The measured expectation values of the circuit along PauliX, PauliY, PauliZ operators.
    """
    qml.RX(datapoint[0], wires=0)
    qml.RY(datapoint[1], wires=0)
    measure = [qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliZ(wires=0))]
    return measure


q_centroid = np.zeros(shape=[3, 3])
q_data = np.zeros(shape=[len(X), 3])
for c_index in range(len(q_centroid)):
    q_centroid[c_index, :] = SingleQubitCircuit(datapoint=quantum_centroids[c_index])

for data_index in range(len(q_data)):
    q_data[data_index, :] = SingleQubitCircuit(datapoint=X[data_index])


def PlotonSphere(q_data: np.array, q_centroid: np.array):
    b = qutip.Bloch()
    b.point_color = ["r", "b", "g"]
    b.point_marker = ['o', 'o', 'o', 's', 's', 's']
    b.add_points([q_data[np.where(quantum_clusters == 0), 0][0], q_data[np.where(quantum_clusters == 0), 1][0], q_data[np.where(quantum_clusters == 0), 2][0]], alpha=0.4)
    b.add_points([q_data[np.where(quantum_clusters == 1), 0][0], q_data[np.where(quantum_clusters == 1), 1][0], q_data[np.where(quantum_clusters == 1), 2][0]], alpha=0.4)
    b.add_points([q_data[np.where(quantum_clusters == 2), 0][0], q_data[np.where(quantum_clusters == 2), 1][0], q_data[np.where(quantum_clusters == 2), 2][0]], alpha=0.4)
    #b.add_points([q_data[np.where(y == 0), 0][0], q_data[np.where(y == 0), 1][0], q_data[np.where(y == 0), 2][0]], alpha=0.4)
    #b.add_points([q_data[np.where(y == 1), 0][0], q_data[np.where(y == 1), 1][0], q_data[np.where(y == 1), 2][0]], alpha=0.4)
    #b.add_points([q_data[np.where(y == 2), 0][0], q_data[np.where(y == 2), 1][0], q_data[np.where(y == 2), 2][0]], alpha=0.4)
    b.add_points([q_centroid[0, 0], q_centroid[0, 1], q_centroid[0, 2]])
    b.add_points([q_centroid[1, 0], q_centroid[1, 1], q_centroid[1, 2]])
    b.add_points([q_centroid[2, 0], q_centroid[2, 1], q_centroid[2, 2]])
    b.show()
    return plt.show(block=True)


PlotonSphere(q_data=q_data, q_centroid=q_centroid)