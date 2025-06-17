import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(42)

def euclidean_distance(x1, x2):
    """Compute the Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Step 1: Initialize centroids randomly
        random_indices = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_indices]

        # Step 2: Iteratively optimize centroids
        for _ in range(self.max_iters):
            # Assign samples to nearest centroids
            self.clusters = self._assign_clusters()

            if self.plot_steps:
                self._plot_clusters()

            # Recalculate centroids
            previous_centroids = self.centroids
            self.centroids = self._compute_centroids()

            if self.plot_steps:
                self._plot_clusters()

            # Stop if centroids do not change
            if self._has_converged(previous_centroids, self.centroids):
                break

        return self._generate_labels()

    def _assign_clusters(self):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._find_nearest_centroid(sample)
            clusters[centroid_idx].append(idx)
        return clusters

    def _find_nearest_centroid(self, sample):
        distances = [euclidean_distance(sample, centroid) for centroid in self.centroids]
        return np.argmin(distances)

    def _compute_centroids(self):
        centroids = np.zeros((self.K, self.n_features))
        for idx, cluster in enumerate(self.clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean
        return centroids

    def _has_converged(self, old_centroids, new_centroids):
        total_movement = sum(
            euclidean_distance(old, new) for old, new in zip(old_centroids, new_centroids)
        )
        return total_movement == 0

    def _generate_labels(self):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _plot_clusters(self):
        plt.figure(figsize=(12, 8))
        for cluster in self.clusters:
            points = self.X[cluster].T
            plt.scatter(*points)

        for centroid in self.centroids:
            plt.scatter(*centroid, marker='x', color='black', linewidth=2)

        plt.show()

# Run example
if __name__ == '__main__':
    X, y = datasets.make_blobs(n_samples=500, centers=4, n_features=2, random_state=42)

    print(f'Data shape: {X.shape}')
    num_clusters = len(np.unique(y))
    print(f'Number of clusters: {num_clusters}')

    model = KMeans(K=num_clusters, max_iters=150, plot_steps=True)
    predictions = model.predict(X)

