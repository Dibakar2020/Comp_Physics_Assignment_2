# Euler methods 

import numpy as np
import matplotlib.pyplot as plt

# defining the given differential function
def f(y, t):
    return y / t - (y / t) ** 2

# solution using analytical methods 
def y(t):
    return t/(1+np.log(t))

# euler method
def euler(a, b, yi, h):
    n = int((b - a) / h + 1)
    tt = np.linspace(a, b, n)
    ww = np.zeros(n)
    ww[0] = yi
    for i in range(1, n):
        ww[i] = ww[i - 1] + h * f(ww[i - 1], tt[i - 1])
    return tt, ww

# given initial condition and step size 
a, b, yi, h = 1, 2, 1, 0.1

tt, ww = euler(a, b, yi, h)
analytical_solution = y(tt)
# Analytical solution and solution using Euler method
print("Analytical solution: ")
print(analytical_solution)
print("Solution using Euler method: ")
print(ww)

# Absolute error 
absolute_error = np.abs(analytical_solution - ww)
print("Absolute error in solution obtained using Euler method: ")
print(absolute_error)

# Relative error 
relative_error = absolute_error/np.abs(analytical_solution)
print("Relative error in solution obtained using Euler method: ")
print(relative_error)


# plotting the solution 
plt.plot(tt, ww, label = "Solution using Euler method")
plt.plot(tt, y(tt), label = "Analytical solution")
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution using Euler Method')
plt.grid()
plt.legend()
plt.show()

