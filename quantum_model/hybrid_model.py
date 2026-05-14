import numpy as np
from sklearn.model_selection import train_test_split
from quantum_model.quantum_circuit import quantum_circuit

def train_hybrid_model(X, y, epochs=15, lr=0.001):

    # Train-Test Split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize Target 
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    # Initialize Parameters
    q_weights = np.random.randn(4) * 0.1
    w_classical = np.random.randn() * 0.1
    b_classical = np.random.randn() * 0.1

    # Training Loop
    for _ in range(epochs):
        for i in range(len(X_train)):

            # Forward pass
            q_out = quantum_circuit(X_train[i], q_weights)
            y_pred = w_classical * q_out + b_classical

            # Error 
            error = y_pred - y_train_scaled[i]

            # Update classical layer
            w_classical -= lr * error * q_out
            b_classical -= lr * error

            # Stabilized quantum update
            q_weights -= lr * error * np.ones_like(q_weights)

    # Testing Phase
    predictions = []
    for i in range(len(X_test)):
        q_out = quantum_circuit(X_test[i], q_weights)
        y_pred = w_classical * q_out + b_classical
        predictions.append(y_pred)

    predictions = np.array(predictions)

    # Compute MSE on scaled values
    mse = np.mean((y_test_scaled - predictions) ** 2)

    return mse
