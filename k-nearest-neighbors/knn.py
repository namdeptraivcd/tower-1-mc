from __future__ import print_function
import numpy as np
from time import time 

#setup để so sánh time giữa các cách để xác định k nearest neighbors
d, N = 1000, 100000
X = np.random.randn(N, d)
z = np.random.randn(d) # điểm dữ liệu đang cần xét đến 


#Tính bình hương khoảng cách giữa z và mỗi hàng của X
def dis_pp(x, z):
    d = z - x.reshape(z.shape, axis = 0)
    return np.sum(d*d)

def dis_ps_naive(z, X):
    N = X.shape[0]
    res = np.zeros((1, N))
    for i in range (N):
        res[0][i] = dis_pp[X[i], z]
    return res



#Tính mỗi tích vô hướng giữa X[i].T và z, lưu bình phương của các euclid X[i]. 
def_dis_fast(z, X):