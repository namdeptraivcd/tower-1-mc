from __future__ import print_function
import numpy as np
from time import time 

#setup để so sánh time giữa các cách để xác định k nearest neighbors
d, N = 1000, 100000
X = np.random.randn(N, d)
z = np.random.randn(d) # điểm dữ liệu đang cần xét đến 


#Tính bình hương khoảng cách giữa z và mỗi hàng của X
def dis_pp(x, z):
    d = z - x.reshape(z.shape)
    return np.sum(d*d)

def dis_ps_naive(z, X):
    N = X.shape[0]
    res = np.zeros((1, N))
    for i in range (N):
        res[0][i] = dis_pp(X[i], z)
    return res



#Tính mỗi tích vô hướng giữa X[i].T và z, lưu bình phương của các euclid X[i]. 
def dis_pp_fast(z, X):
    X2 = np.sum(X*X, axis = 1)
    z2 = np.sum(z*z, axis = 1)
    res_2 = X2 + z2 - 2*np.dot(X.T, z) # có thể bỏ cả z2
    return res_2 

t1 = time()
Distance1 = dis_ps_naive(z, X)
print("Naive time: ", time() - t1)

t2 = time()
Distance2 = dis_pp_fast(z, X)
print("Fast time: ", time() - t2)
print("result_difference: ", np.linalg.norm(Distance1-Distance2))