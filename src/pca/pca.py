import numpy as np


class Pca:

    def __init__(self, data: np.ndarray):
        self.data = data
        self.num_records, self.num_variables = data.shape
        self.correlation_matrix = self._calculate_correlation_matrix()
        self.eigenvalues, self.eigenvectors = self._calculate_eigen()
        self.principal_components = self._calculate_principal_components()

    def _calculate_correlation_matrix(self) -> np.ndarray:
        return np.cov(self.data, rowvar=False)

    def _calculate_eigen(self) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues, eigenvectors = np.linalg.eig(self.correlation_matrix)
        sort_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sort_indices]
        sorted_eigenvectors = eigenvectors[:, sort_indices]
        return sorted_eigenvalues, sorted_eigenvectors

    def _calculate_principal_components(self) -> np.ndarray:
        return self.data @ self.eigenvectors

    def get_correlation_matrix(self) -> np.ndarray:
        return self.correlation_matrix

    def get_eigenvalues(self) -> np.ndarray:
        return self.eigenvalues

    def get_eigenvectors(self) -> np.ndarray:
        return self.eigenvectors

    def get_principal_components(self) -> np.ndarray:
        return self.principal_components