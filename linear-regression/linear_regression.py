from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt 

X = np.array([[147, 150, 153, 155, 158, 160, 163, 165, 168, 170, 170, 173, 175, 178, 180, 183]]).T

y = np.array([[49, 50, 51, 52, 54, 56, 58, 59, 60, 72, 63, 64, 66, 67, 68]]).T

plt.plot(X, y, 'ro')
plt.axis([140, 190, 40, 80])
plt.xlabel('Height (cm)')
plt.ylabel('weight (kg)')
plt.savefig('show_data.png')
