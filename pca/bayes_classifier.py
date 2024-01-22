import numpy as np


def mahalanobis_distance(x, mean, covariance):
    diff = x - mean

    distance = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(covariance)), diff.T))

    return distance


class LDA:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_means = []
        self.shared_covariance = np.zeros((X.shape[1], X.shape[1]))

        for c in self.classes:
            X_class = X[y == c]
            mean = np.mean(X_class, axis=0)
            self.class_means.append(mean)
            self.shared_covariance += np.cov(X_class, rowvar=False)

        self.shared_covariance /= len(self.classes)

    def predict(self, X):
        predictions = []
        for x in X:
            distances = []
            for _, mean in enumerate(self.class_means):
                distance = mahalanobis_distance(x, mean, self.shared_covariance)
                distances.append(distance)

            predictions.append(self.classes[np.argmin(distances)])

        return np.array(predictions)

    def get_parameters_count(self, X, y):
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        lda_params_count = n_features * n_classes + n_features**2
        return lda_params_count
