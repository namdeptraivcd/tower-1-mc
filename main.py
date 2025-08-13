import numpy as np
#init 
X = np.random.rand(1000)
y = 3 + 2*X
y = y.reshape(-1,1)
one = np.ones((X.size, 1))
Xbar = np.concatenate((X.reshape(-1,1), one), axis=1)
def grad(w):
    N = y.size
    return 1/N * Xbar.T .dot(Xbar.dot(w) - y)
def myGD(grad, eta, w_init):
    w = [w_init]
    for i in range(1000):
        w_new = w[-1] - eta * grad(w[-1])
        if np.linalg.norm(grad(w_new))/ len(w_new) < 1e-6:
            break
        w.append(w_new)
    return (w, i+1)
w_init = np.array([[2], [1]])
w_result, iter_result = myGD(grad, eta=1, w_init=w_init)
print(f"the final param is {w_result[-1].T}\n the number of iterations is {iter_result}")
