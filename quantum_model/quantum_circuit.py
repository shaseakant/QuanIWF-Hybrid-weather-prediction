import pennylane as qml

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(x, weights):

    
    # 1. Feature Encoding
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)


    # 2. Variational Layer 1
    for i in range(n_qubits):
        qml.RX(weights[i], wires=i)


    # 3. Entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # 4. Variational Layer 2

    for i in range(n_qubits):
        qml.RZ(weights[i], wires=i)

    # 5. Measurement
    return qml.expval(qml.PauliZ(0))
