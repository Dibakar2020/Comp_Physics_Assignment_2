# RK4 method for 2nd order differentail equation 

import numpy as np 
import matplotlib.pylab as plt 

# given 2nd order differentail function 
def f(x, y, z):
    return 2 * z - y + x * np.exp(x) - x

# RK4 method
def rk4(a, b, yi, zi, h):
    n = int((b - a)/h + 1)
    xx = np.linspace(a, b, n)
    yy = np.zeros(n)
    zz = np.zeros(n)
    yy[0] = yi
    zz[0] = zi
    for i in range(1, n):
        k1y = h * zz[i - 1]
        k1z = h * f(xx[i - 1], yy[i - 1], zz[i - 1])
        k2y = h * (zz[i - 1] + k1z / 2)
        k2z = h * f(xx[i - 1] + h / 2, yy[i - 1] + k1y / 2, zz[i - 1] + k1z / 2)
        k3y = h * (zz[i - 1] + k2z / 2)
        k3z = h * f(xx[i - 1] + h / 2, yy[i - 1] + k2y / 2, zz[i - 1] + k2z / 2)
        k4y = h * (zz[i - 1] + k3z)
        k4z = h * f(xx[i - 1] + h, yy[i - 1] + k3y, zz[i - 1] + k3z)

        yy[i] = yy[i - 1] + (k1y + 2 * k2y + 2 * k3y + k4y)
        zz[i] = zz[i - 1] + (k1z + 2 * k2z + 2 * k3z + k4z)
    return xx, yy

# given initial condition and step size 
a, b = 0, 1
yi, zi, h = 0, 0, 0.1

# RK4 solution 
xx, yy = rk4(a, b, yi, zi, h)

print("Solution using RK4 method: ")
print(yy)

# plotting of the solution 
plt.plot(xx, yy)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of the 2nd order ODE using RK4')
plt.grid(True)
plt.show()


