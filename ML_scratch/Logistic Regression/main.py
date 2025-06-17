import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 1000 == 0:
                cost = -1 / n_samples * np.sum(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


# ------------- Test 1: Breast Cancer Dataset ------------- #
if __name__ == '__main__':
    # Load binary classification dataset
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression
    model = LogisticRegression(lr=0.0001, n_iters=10000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy (Breast Cancer Dataset):", accuracy_score(y_test, predictions))

    # Optional visualization (for 2D data)
    if X.shape[1] == 2:
        plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='coolwarm', alpha=0.6)
        plt.title("Logistic Regression Predictions")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

