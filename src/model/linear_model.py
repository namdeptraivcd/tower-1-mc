import numpy as np

class LinearRegression:
    """
    Simple Linear Regression implementation similar to sklearn's LinearRegression.
    Supports fit, predict, and coef_/intercept_ attributes.
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if self.fit_intercept:
            X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_ = X
        # Closed-form solution (Normal Equation)
        theta = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
        if self.fit_intercept:
            self.intercept_ = theta[0].squeeze()
            self.coef_ = theta[1:].squeeze()
        else:
            self.intercept_ = 0.0
            self.coef_ = theta.squeeze()
        return self

    def predict(self, X):
        X = np.asarray(X)
        y_pred = X @ self.coef_.T
        if self.fit_intercept:
            y_pred = y_pred + self.intercept_
        return y_pred

    def score(self, X, y):
        """R^2 score."""
        y = np.asarray(y)
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v
    pass
