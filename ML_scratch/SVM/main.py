import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iter=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(y_[idx], x_i))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LinearSVM()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = np.mean(y_pred == y_test)
    print(f"Accuracy (linear SVM): {acc:.2f}")

