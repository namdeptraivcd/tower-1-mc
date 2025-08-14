import numpy as np
from src.model.gd.gd_nag import GD_NAG
#init 
X = np.random.rand(1000)
y = 3 + 2*X
y = y.reshape(-1,1)
one = np.ones((X.size, 1))
Xbar = np.concatenate((X.reshape(-1,1), one), axis=1)
def grad(w):
    N = y.size
    return 1/N * Xbar.T .dot(Xbar.dot(w) - y)

theta_init = np.array([[2], [1]])
theta_result, iter_result = GD_NAG(grad, eta=1, theta_init=theta_init)
print(f"the final param is {theta_result[-1].T}\n the number of iterations is {iter_result}")
