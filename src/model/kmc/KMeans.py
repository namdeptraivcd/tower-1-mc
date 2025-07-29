import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        """
        Gần giống sklearn.cluster.KMeans:
        - n_clusters: số cụm cần tìm
        - max_iter: số lần lặp tối đa
        - tol: ngưỡng sai số để dừng
        - random_state: cố định seed ngẫu nhiên để tái hiện kết quả
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Các thuộc tính sau sẽ được khởi tạo sau khi gọi fit()
        self.cluster_centers_ = None  # centroid của các cụm
        self.labels_ = None           # nhãn cụm của từng điểm
        self.inertia_ = None          # tổng lỗi bình phương

    def fit(self, X):
        """
        Huấn luyện mô hình KMeans trên tập dữ liệu X.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Khởi tạo ngẫu nhiên centroid
        random_idxs = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_idxs]

        for _ in range(self.max_iter):
            # Gán nhãn cụm cho mỗi điểm
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)

            # Tính centroid mới
            new_centers = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else self.cluster_centers_[i]
                for i in range(self.n_clusters)
            ])

            # Kiểm tra hội tụ
            diff = np.linalg.norm(new_centers - self.cluster_centers_)
            if diff < self.tol:
                break

            self.cluster_centers_ = new_centers

        # Gán nhãn cuối cùng
        self.labels_ = np.argmin(self._compute_distances(X), axis=1)

        # Tính inertia_
        self.inertia_ = np.sum([
            np.linalg.norm(X[i] - self.cluster_centers_[self.labels_[i]])**2
            for i in range(n_samples)
        ])

        return self

    def predict(self, X):
        """
        Dự đoán nhãn cụm cho dữ liệu mới.
        """
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """
        Huấn luyện mô hình rồi trả về nhãn cụm cho X.
        """
        self.fit(X)
        return self.labels_

    def score(self, X):
        """
        Trả về -inertia_ giống sklearn để có thể dùng trong GridSearchCV, cross_val_score,...
        """
        labels = self.predict(X)
        return -np.sum([
            np.linalg.norm(X[i] - self.cluster_centers_[labels[i]])**2
            for i in range(len(X))
        ])

    def _compute_distances(self, X):
        """
        Tính khoảng cách Euclidean từ mỗi điểm đến từng centroid.
        """
        return np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
