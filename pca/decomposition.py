import numpy as np
from copy import deepcopy


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance_ = None

    def fit(self, X):
        mean = np.mean(X, axis=0)
        X = X - mean

        cov = np.cov(X.T)

        self.eigenvalues, eignvectors = np.linalg.eigh(cov)

        self.eigenvectors = eignvectors.T

        if type(self.n_components) is float:
            sum_ = np.sum(self.eigenvalues[np.argsort(self.eigenvalues)[::-1]])
            nor = self.eigenvalues / sum_
            cumulative = 0

            for i, value in enumerate(nor[np.argsort(nor)[::-1]]):
                cumulative += value
                if cumulative > self.n_components:
                    break

            self.explained_variance_ = np.sort(self.eigenvalues)[::-1][:i]
            self.components = self.eigenvectors[np.argsort(self.eigenvalues)[::-1][:i]]

        else:
            self.explained_variance_ = np.sort(self.eigenvalues)[::-1][
                : self.n_components
            ]
            self.components = self.eigenvectors[
                np.argsort(self.eigenvalues)[::-1][: self.n_components]
            ]

    def transform(self, X):
        mean = np.mean(X, axis=0)
        X = X - mean

        return np.dot(X, self.components.T)

    def reconstruct(self, X):
        mean = np.mean(X, axis=0)
        X = X - mean
        return (
            np.dot(np.dot(X, self.components.T), np.linalg.pinv(self.components.T))
            + mean
        )


class FisherLDA:
    def __init__(self):
        self.components = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance_ = None
        self.means = {}

    def fit(self, X, y):
        labels = np.unique(y)

        self.S_B = np.zeros((X.shape[1], X.shape[1]))
        self.S_W = np.zeros((X.shape[1], X.shape[1]))

        overall_mean = np.mean(X, axis=0)

        for label in labels:
            mean = np.mean(X[np.where(y == label)], axis=0)
            self.means[label] = mean

        for label in labels:
            class_data = X[np.where(y == label)]
            mean_vector = self.means[label]
            n_i = class_data.shape[0]

            scatter_matrix = np.sum(
                [np.outer((x - mean_vector), (x - mean_vector)) for x in class_data],
                axis=0,
            )

            self.S_W += scatter_matrix

        for label in labels:
            mean_vector = self.means[label]
            n_i = X[np.where(y == label)].shape[0]

            self.S_B += n_i * np.outer(
                (mean_vector - overall_mean), (mean_vector - overall_mean)
            )

        S_W_inv = np.linalg.inv(self.S_W)

        self.eigenvalues, self.eigenvectors = np.linalg.eigh(np.dot(S_W_inv, self.S_B))

    def transform(self, X, n_components):
        self.n_components = n_components
        eiglist = [
            (self.eigenvalues[i], self.eigenvectors[:, i])
            for i in range(len(self.eigenvalues))
        ]

        eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)

        self.W = np.array([eiglist[i][1] for i in range(self.n_components)])
        self.W = np.asarray(self.W).T
        return X @ self.W
