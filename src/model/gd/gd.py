import numpy as np

def GD(grad, eta, w_init):
    w = [w_init]
    for i in range(1000):
        w_new = w[-1] - eta * grad(w[-1])
        if np.linalg.norm(grad(w_new))/ len(w_new) < 1e-6:
            break
        w.append(w_new)
    return (w, i+1)