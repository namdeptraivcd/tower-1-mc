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
def dis_ps_fast(z, X):
    X2 = np.sum(X*X, axis = 1)
    z2 = np.sum(z*z)
    res_2 = X2 + z2 - 2*np.dot(X, z) # có thể bỏ cả z2
    return res_2 

t1 = time()
Distance1 = dis_ps_naive(z, X)
print("Naive time point to set: ", time() - t1)

t2 = time()
Distance2 = dis_ps_fast(z, X)
print("Fast time point to set: ", time() - t2)
print("result_difference: ", np.linalg.norm(Distance1-Distance2))


# tính khoảng cách của các Z tới ma trận X
M = 100
Z = np.random.randn(M, d)
def dis_ss_naive(Z, X):
    N = X.shape[0]
    M = Z.shape[0]
    res = np.zeros((M,N))
    for i in range (M):
        res[i] = dis_ps_naive(Z[i], X)
    return res

def dis_ss_fast(Z, X):
    X2 = np.sum(X*X, axis = 1)
    Z2 = np.sum(Z*Z, axis = 1)
    return X2.reshape(1, -1) + Z2.reshape(-1,1) - 2*np.dot(Z, X.T) # có thể bỏ cả Z2

t3 = time()
dis_ss_naive(Z, X)
print("Naive time set to set: ", time() - t3)


t4 = time()
dis_ss_fast(Z,X)
print("Fast time set to set: ", time() - t4)
print("Result_difference: ", np.linalg.norm(dis_ss_naive(Z, X) - dis_ss_fast(Z, X)))






#Sử dụng model KNN
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
