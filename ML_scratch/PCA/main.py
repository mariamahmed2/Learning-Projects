import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Step 1: Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Compute covariance matrix
        cov = np.cov(X_centered.T)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors.T)  # shape: (features, features)

        # Step 4: Sort eigenvectors by descending eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[idxs[:self.n_components]]

        # Optional: print explained variance
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        print("Explained Variance Ratio:", explained_variance_ratio[:self.n_components])

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)


if __name__ == '__main__':
    data = datasets.load_iris()
    X = data.data
    y = data.target

    pca = PCA(n_components=2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print('Original shape:', X.shape)
    print('Transformed shape:', X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.figure(figsize=(8,6))
    plt.scatter(x1, x2, c=y, edgecolor='k', cmap='viridis', alpha=0.8)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection of Iris Dataset")
    plt.colorbar()
    plt.grid(True)
    plt.show()

