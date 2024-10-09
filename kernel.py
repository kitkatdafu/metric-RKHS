from typing import List

import numpy as cp
from tqdm import tqdm


class RKHSKernel:

    kernel_type: str

    def __init__(self, kernel_type: str, sigma, eigen_cutoff, eigen_threshold):
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.eigen_cutoff = eigen_cutoff
        self.eigen_threshold = eigen_threshold

    def k(self, x: cp.ndarray, y: cp.ndarray):
        if self.kernel_type == "linear":
            return cp.dot(x, y)
        elif self.kernel_type == "gaussian":
            return cp.exp(-cp.linalg.norm(x - y) ** 2 / (2 * self.sigma**2))
        else:
            raise NotImplementedError

    def gram_matrix(self, X: cp.ndarray, Y: cp.ndarray = None):
        print("Computing the gram matrix")
        if Y is None:
            Y = X
        if self.kernel_type == "linear":
            return cp.dot(X, Y.T)
        elif self.kernel_type == "gaussian":
            X_norm = cp.sum(X**2, axis=1)[:, cp.newaxis]  # shape: (n, 1)
            X_test_norm = cp.sum(Y**2, axis=1)[cp.newaxis, :]  # shape: (1, m)

            dist_matrix = X_norm + X_test_norm - 2 * cp.dot(X, Y.T)  # shape: (n, m)

            return cp.exp(-dist_matrix / (2 * self.sigma**2))
        else:
            raise NotImplementedError

    def center_gram_matrix(self, K: cp.ndarray):
        print("Centering the gram matrix")
        one = cp.ones_like(K) / K.shape[0]
        return K - one @ K - K @ one + one @ K @ one

    def form_matrix_a(self, eig_val: cp.ndarray, eig_vec: cp.ndarray):
        print("Forming the matrix A")
        if self.eigen_threshold is not None:
            num_eigens = cp.sum(eig_val > self.eigen_threshold)
        else:
            num_eigens = self.eigen_cutoff
        A = eig_vec[:, -num_eigens:]
        return A

    def kpca(self, X: cp.ndarray, X_val_s: List[cp.ndarray]):

        # create the kernel matrix
        K = self.gram_matrix(X)

        # center the gram matrix
        K_bar = self.center_gram_matrix(K)

        # compute the eigenvalues and eigenvectors
        print("Computing the eigenvectors")
        eig_val, eig_vec = cp.linalg.eigh(K_bar)
        A = self.form_matrix_a(eig_val, eig_vec)

        # project the data
        print("Projecting the data")
        phi = K @ A

        print("Projecting the validation data")
        phi_val_s = []
        for X_val in X_val_s:
            K_matrix = self.gram_matrix(X_val, X)
            phi_val_s.append(K_matrix @ A)

        return phi, phi_val_s


def main():
    X = cp.random.randn(100, 18)
    X_val_s = [cp.random.randn(100, 18) for _ in range(10)]
    kernel = RKHSKernel("linear", sigma=1, eigen_cutoff=10, eigen_threshold=None)
    phi, phi_val_s = kernel.kpca(X, X_val_s)
    print(phi.shape, phi_val_s[0].shape)


if __name__ == "__main__":
    main()
