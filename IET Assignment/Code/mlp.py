"""
mlp.py
A Multilayer Perceptron implemented entirely from first principles using
only NumPy: forward propagation, back-propagation (one hidden layer,
generalises to more), mini-batch gradient descent, L2 regularisation.

Architecture: Input -> Dense(ReLU) -> Dense(Sigmoid, 1 output unit)

Activation justification:
- ReLU in the hidden layer avoids vanishing gradients and is cheap to
  differentiate, which matters because weight initialisation is itself
  being searched over by the Genetic Algorithm (many short training runs).
- Sigmoid on the output layer maps to a valid probability in (0, 1) for
  binary disease-risk classification, and pairs with binary cross-entropy
  to give a simple, numerically stable gradient (y_hat - y).
"""
import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


class MLP:
    def __init__(self, n_input, n_hidden, seed=0, weight_init=None):
        rng = np.random.default_rng(seed)
        if weight_init is not None:
            # Weights supplied externally (e.g. by the Genetic Algorithm)
            self.W1, self.b1, self.W2, self.b2 = weight_init
        else:
            # He initialisation, a sensible default for ReLU networks
            self.W1 = rng.normal(0, np.sqrt(2.0 / n_input), size=(n_input, n_hidden))
            self.b1 = np.zeros((1, n_hidden))
            self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), size=(n_hidden, 1))
            self.b2 = np.zeros((1, 1))
        self.n_input = n_input
        self.n_hidden = n_hidden

    def get_flat_weights(self):
        return np.concatenate([self.W1.ravel(), self.b1.ravel(),
                                self.W2.ravel(), self.b2.ravel()])

    def set_flat_weights(self, flat):
        ni, nh = self.n_input, self.n_hidden
        i = 0
        self.W1 = flat[i:i + ni * nh].reshape(ni, nh); i += ni * nh
        self.b1 = flat[i:i + nh].reshape(1, nh); i += nh
        self.W2 = flat[i:i + nh * 1].reshape(nh, 1); i += nh
        self.b2 = flat[i:i + 1].reshape(1, 1)

    @staticmethod
    def n_params(n_input, n_hidden):
        return n_input * n_hidden + n_hidden + n_hidden * 1 + 1

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = sigmoid(Z2)
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def backward(self, cache, y, l2=0.0):
        X, Z1, A1, Z2, A2 = cache
        m = X.shape[0]
        y = y.reshape(-1, 1)

        dZ2 = (A2 - y) / m                      # BCE + sigmoid gradient
        dW2 = A1.T @ dZ2 + l2 * self.W2 / m
        db2 = dZ2.sum(axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_grad(Z1)
        dW1 = X.T @ dZ1 + l2 * self.W1 / m
        db1 = dZ1.sum(axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def step(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    @staticmethod
    def bce_loss(y_hat, y, W1=None, W2=None, l2=0.0):
        y = y.reshape(-1, 1)
        eps = 1e-9
        loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        if l2 > 0 and W1 is not None:
            loss += (l2 / (2 * len(y))) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        return loss

    def fit(self, X, y, epochs=200, lr=0.05, batch_size=32, l2=1e-3,
            seed=0, verbose=False, track_history=False):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        history = []
        for epoch in range(epochs):
            perm = rng.permutation(n)
            X_s, y_s = X[perm], y[perm]
            for start in range(0, n, batch_size):
                xb = X_s[start:start + batch_size]
                yb = y_s[start:start + batch_size]
                y_hat, cache = self.forward(xb)
                grads = self.backward(cache, yb, l2=l2)
                self.step(grads, lr)
            if track_history or (verbose and epoch % 20 == 0):
                y_hat_full, _ = self.forward(X)
                loss = self.bce_loss(y_hat_full, y, self.W1, self.W2, l2)
                if track_history:
                    history.append(loss)
                if verbose and epoch % 20 == 0:
                    print(f"epoch {epoch:3d}  loss={loss:.4f}")
        return history

    def predict_proba(self, X):
        y_hat, _ = self.forward(X)
        return y_hat.ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
