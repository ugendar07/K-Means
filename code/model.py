import numpy as np

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Initialize cluster centers (need to be careful with the initialization,
        # otherwise you might see that none of the pixels are assigned to some
        # of the clusters, which will result in a division by zero error)
        self.cluster_centers = X[np.random.choice(X.shape[0], self.num_clusters, replace=False)]
        
        for _ in range(max_iter):
            # Assign each sample to the closest prototype
            distances = np.sqrt(np.sum((X[:, np.newaxis] - self.cluster_centers) ** 2, axis=2))
            labels = np.argmin(distances, axis=1)
            
            # Update prototypes
            new_cluster_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.num_clusters)])
            
            # Check if the difference between the old and new prototypes is less than epsilon
            if np.sqrt(np.sum((new_cluster_centers - self.cluster_centers) ** 2)) < self.epsilon:
                break
            
            self.cluster_centers = new_cluster_centers

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        distances = np.sqrt(np.sum((X[:, np.newaxis] - self.cluster_centers) ** 2, axis=2))
        labels = np.argmin(distances, axis=1)
        return labels
    
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        distances = np.sqrt(np.sum((X[:, np.newaxis] - self.cluster_centers) ** 2, axis=2))
        labels = np.argmin(distances, axis=1)
        return self.cluster_centers[labels]
