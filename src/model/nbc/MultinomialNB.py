import numpy as np
class MultinomialNB:
    def __init__(self, alpha = 1.0):
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
    
        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y==c]
            class_count[idx] = X_c.shape[0]
            feature_count[idx, :] = X_c.sum(axis = 0)
        
        self.class_log_prior_ = np.log(class_count / n_samples)


        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis = 1).reshape(-1,1)
        self.feature_log_prob = np.log(smoothed_fc / smoothed_cc)

        return self 
    
    def _log_likelihood(self, X):
        return X.dot(self.feature_log_prob_.T) + self.class_log_prior_
    
    def predict(self, X):
        log_likelihood = self._log_likelihood(X)
        return self.classes_[np.argmax(log_likelihood, axis=1)]
    
    def predict_proba(self, X):
        log_likelihood = self._log_likelihood(X)
        log_prob = log_likelihood - log_likelihood.max(axis=1, keepdims=True)
        prob = np.exp(log_prob)
        return prob / prob.sum(axis=1, keepdims=True)