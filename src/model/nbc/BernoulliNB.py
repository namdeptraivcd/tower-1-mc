import numpy as np
class BernoulliNB:
    def __init__(self):
        self.class_priors = None
        self.feature_probs = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        self.class_priors = {}
        self.feature_probs = {}

        for c in classes:
            X_c = X[y==c]
            self.class_priors[c] = X_c.shape[0] / n_samples
            self.feature_probs[c] = 