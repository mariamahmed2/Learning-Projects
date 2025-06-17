import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# ---------- Gradient Descent (Vectorized) ---------- #

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# ---------- Cost Function ---------- #

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ---------- Normal Equation (Closed-form) ---------- #

def linear_regression_normal_equation(X, y):
    # Add bias term (intercept)
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    # θ = (XᵀX)^(-1) Xᵀy
    theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return theta_best


# ---------- Example Usage ---------- #

if __name__ == "__main__":
    # Generate synthetic regression dataset
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Fit using Gradient Descent
    model_gd = LinearRegressionGD(learning_rate=0.01, n_iters=1000)
    model_gd.fit(X_train, y_train)
    preds_gd = model_gd.predict(X_test)

    print("Gradient Descent MSE:", mean_squared_error(y_test, preds_gd))

    # Fit using Normal Equation
    theta_normal = linear_regression_normal_equation(X_train, y_train)
    X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    preds_ne = X_test_b @ theta_normal

    print("Normal Equation MSE:", mean_squared_error(y_test, preds_ne))

    # Visualization
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, preds_gd, color='red', label='Gradient Descent')
    plt.plot(X_test, preds_ne, color='green', linestyle='--', label='Normal Equation')
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()

