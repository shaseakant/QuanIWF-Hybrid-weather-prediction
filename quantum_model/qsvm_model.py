import pennylane as qml
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Quantum Device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum Feature Map
def quantum_feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

# Kernel Circuit
@qml.qnode(dev)
def kernel_circuit(x1, x2):
    quantum_feature_map(x1)
    qml.adjoint(quantum_feature_map)(x2)
    return qml.probs(wires=range(n_qubits))

# Quantum Kernel Matrix
def quantum_kernel(X1, X2):
    kernel = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            kernel[i, j] = kernel_circuit(x1, x2)[0]
    return kernel

# QSVM Training
def train_qsvm(X, y):

    # Scalability Control
    # Limit dataset size for QSVM to avoid quadratic explosion
    max_samples = 100
    if len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Compute Quantum Kernel
    K_train = quantum_kernel(X_train, X_train)
    K_test = quantum_kernel(X_test, X_train)

    # Train SVR with Precomputed Kernel
    model = SVR(kernel="precomputed")
    model.fit(K_train, y_train)

    # Prediction
    predictions = model.predict(K_test)

    # Evaluation
    mse = mean_squared_error(y_test, predictions)

    return mse
