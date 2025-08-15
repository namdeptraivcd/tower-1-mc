import numpy as np
def predict (w, X):
    return np.sign(X.dot(w))

def perceptron (X, y, w_init):
    w = w_init
    while True:
        pred = predict(w, X)
        mis_idxs = np.where(np.equal(pred, y) == False)[0]
        num_mis = mis_idxs.shape[0]
        if num_mis == 0:
            return w
        random_id = np.random.choice(mis_idxs, 1)[0]
        w = w + y[random_id] * X[random_id]