# Backward integration with Euler’s Method

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import root

# defining the given differential function
def f1(x, y):
    return - 9 * y

def f2(x, y):
    return -20 * (y - x)**2 + 2 * x



# backward euler method
def backward_euler(f, a, b, yi, h):
    n = int((b - a)/h + 1)
    xx = np.linspace(a, b, n)
    ww = np.zeros(n)
    ww[0] = yi
    # Initial guess for root finding 
    y0 = 0.5
    for i in range(1, n):
        def func(y):
            return y - ww[i - 1] - h * f(xx[i], y)
        sol = root(func, y0)
        ww[i] = sol.x[0]

    return xx, ww


# given initial condition and step size 
a, b = 0, 1
yi1, h = np.e , 0.1
yi2, h = 1/3 , 0.01

# backward solution 
xx, ww1 = backward_euler(f1, a, b, yi1, h)   # first differential equantion 
xx, ww2 = backward_euler(f2, a, b, yi2, h)   # second differential equantion 

# plotting the solution 
plt.plot(xx, ww1, label = "solution of y'=-9y")
plt.plot(xx, ww2, label = "solution of y'=-20(y-x)^2+2x")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of the 1st order ODE using backward euler method')
plt.grid(True)
plt.legend()
plt.show()


        
