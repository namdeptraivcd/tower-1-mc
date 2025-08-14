import numpy as np 

def GD_NAG (grad, theta_init, eta, gamma = 0.9):
    theta = [theta_init]
    v_old = np.zeros(theta_init.shape)
    for i in range (100):
        v_new = gamma * v_old + eta* grad (theta[-1] - gamma* v_old)
        theta_new = theta[-1] - v_new
        if np.linalg.norm(theta_new - theta[-1]) < 1e-6:
            break
        theta.append(theta_new)
        v_old = v_new
    return (theta, i + 1)