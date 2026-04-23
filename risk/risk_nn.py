"""
MediNav — Numpy-Only Neural Network Risk Model
risk/risk_nn.py

PROOF-OF-CONCEPT: A lightweight 1-hidden-layer MLP that learns to predict
risk values from spatial features (distance_to_wall, distance_to_human).

This NN is trained on rule-based risk labels — it learns to approximate
what the formula already computes. The purpose is to demonstrate that a
neural network architecture CAN learn spatial risk patterns from features,
which could generalise to learned features from real hospital sensor data
in future work. It is not a claimed improvement over rule-based risk.

Architecture:
    Input(2) -> Hidden(16, ReLU) -> Output(1, Sigmoid)

Uses ONLY numpy — no PyTorch, TensorFlow, or sklearn.
"""

import numpy as np
import os

# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def _relu(x):
    return np.maximum(0, x)

def _relu_deriv(x):
    return (x > 0).astype(float)

def _sigmoid(x):
    # Numerically stable sigmoid
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def _sigmoid_deriv(s):
    """Derivative given sigmoid output s (not raw input)."""
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Neural Network class
# ---------------------------------------------------------------------------

class RiskNN:
    """
    Simple 1-hidden-layer MLP for risk prediction.

    Parameters
    ----------
    input_dim  : int   — number of input features (default 2)
    hidden_dim : int   — hidden layer width (default 16)
    """

    def __init__(self, input_dim=2, hidden_dim=16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Xavier initialisation
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, 1))

    # ----- Forward pass -----

    def _forward(self, X):
        """Returns (z1, a1, z2, a2) for backprop."""
        z1 = X @ self.W1 + self.b1          # (N, hidden)
        a1 = _relu(z1)                      # (N, hidden)
        z2 = a1 @ self.W2 + self.b2         # (N, 1)
        a2 = _sigmoid(z2)                   # (N, 1)
        return z1, a1, z2, a2

    def predict(self, X):
        """
        Forward pass only — returns risk predictions in [0, 1].

        Parameters
        ----------
        X : np.ndarray of shape (N, input_dim)

        Returns
        -------
        np.ndarray of shape (N,) with values in [0, 1]
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        _, _, _, a2 = self._forward(X)
        return a2.flatten()

    # ----- Training -----

    def train_model(self, X, y, epochs=500, lr=0.01, verbose=True):
        """
        Train the network using mini-batch gradient descent with MSE loss.

        Parameters
        ----------
        X       : np.ndarray (N, input_dim) — feature matrix
        y       : np.ndarray (N,)           — target risk values in [0, 1]
        epochs  : int
        lr      : float — learning rate
        verbose : bool  — print loss every 100 epochs
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        N = X.shape[0]

        for epoch in range(epochs):
            # Forward
            z1, a1, z2, a2 = self._forward(X)

            # Loss: MSE
            loss = np.mean((a2 - y) ** 2)

            # Backward
            dL_da2 = 2.0 * (a2 - y) / N               # (N, 1)
            da2_dz2 = _sigmoid_deriv(a2)                # (N, 1)
            dz2 = dL_da2 * da2_dz2                     # (N, 1)

            dW2 = a1.T @ dz2                            # (hidden, 1)
            db2 = np.sum(dz2, axis=0, keepdims=True)    # (1, 1)

            da1 = dz2 @ self.W2.T                       # (N, hidden)
            dz1 = da1 * _relu_deriv(z1)                 # (N, hidden)

            dW1 = X.T @ dz1                             # (input, hidden)
            db1 = np.sum(dz1, axis=0, keepdims=True)    # (1, hidden)

            # Update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  Loss: {loss:.6f}")

        if verbose:
            print(f"  Training complete. Final loss: {loss:.6f}")

    # ----- Persistence -----

    def save_model(self, path="outputs/risk_nn_weights.npz"):
        """Save weights to a .npz file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        print(f"  Model saved -> {path}")

    def load_model(self, path="outputs/risk_nn_weights.npz"):
        """Load weights from a .npz file."""
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.input_dim = int(data["input_dim"])
        self.hidden_dim = int(data["hidden_dim"])
        print(f"  Model loaded <- {path}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  RiskNN Self-Test")
    print("=" * 55)

    np.random.seed(42)

    # Generate synthetic training data:
    #   feature 0 = distance_to_wall (0–15)
    #   feature 1 = distance_to_human (0–15)
    #   label     = risk (high when close to wall OR human)
    N = 2000
    dist_wall  = np.random.uniform(0, 15, N)
    dist_human = np.random.uniform(0, 15, N)

    # Rule-based risk label
    risk_wall  = np.exp(-dist_wall)
    risk_human = np.exp(-dist_human * 0.8)
    risk_label = np.clip(0.7 * risk_wall + 0.3 * risk_human, 0, 1)

    X = np.column_stack([dist_wall, dist_human])

    # Train
    model = RiskNN(input_dim=2, hidden_dim=16)
    model.train_model(X, risk_label, epochs=500, lr=0.01)

    # Evaluate
    preds = model.predict(X)
    mae = np.mean(np.abs(preds - risk_label))
    print(f"\n  Mean Absolute Error on training set: {mae:.4f}")

    # Save & reload test
    os.makedirs("outputs", exist_ok=True)
    model.save_model("outputs/risk_nn_weights.npz")

    model2 = RiskNN()
    model2.load_model("outputs/risk_nn_weights.npz")
    preds2 = model2.predict(X)
    assert np.allclose(preds, preds2), "Save/load mismatch!"
    print("  Save/load round-trip: PASSED")
    print("=" * 55)
