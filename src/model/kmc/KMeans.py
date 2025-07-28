import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.k = n_clusters
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        np.random.seed(0)
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]
        for _ in range(self.max_iters):
            labels = self.predict(X)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def kmeans_display(self, X):
        if X.shape[1] != 2:
            raise ValueError("Chỉ hiển thị được dữ liệu 2 chiều.")
        labels = self.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=200)
        plt.title('KMeans Clustering')
        plt.show()
