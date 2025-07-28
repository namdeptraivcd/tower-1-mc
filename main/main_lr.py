from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt 
from src.model.linear_model import LinearRegression

from src.config.config import DEVICE

print(DEVICE)

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

plt.plot(X, y, 'ro')
plt.axis([140, 190, 40, 80])
plt.xlabel('Height (cm)')
plt.ylabel('weight (kg)')
# plt.savefig('show_data.png')
plt.show()


# từ show_data ta thấy rằng linear_regression là một mô hình phù hợp với dữ liệu này

# bắt đầu xây dựng hàm loss và tìm ra hệ số w

one = np.ones((X.shape[0], 1))
X_bar = np.concatenate ((one, X), axis = 1) # thêm 1 cột để tính hệ số w1

A = np.dot(X_bar.T, X_bar)
b = np.dot(X_bar.T, y)
w = np.dot(np.linalg.inv(A), b)


# hàm đường regression
w_0 = w[0][0]
w_1 = w[1][0]
x_0 = np.linspace(140, 190, 2)
y_0 = w_0 + w_1 * x_0

# vẽ đường regression 
plt.plot(X,y, 'ro')
plt.plot(x_0, y_0)
plt.axis([140, 190, 40, 80])
plt.xlabel('Height (cm)')
plt.ylabel('weight (kg)')
# plt.savefig('output.png')
plt.show()

# check hệ số thu được 
print('w', w)

# dùng model của nam
model = LinearRegression(fit_intercept= False)
model.fit(X_bar,y)
print('my w', model.coef_)

