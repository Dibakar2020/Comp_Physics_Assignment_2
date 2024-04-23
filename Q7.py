# relaxation methods for 2nd order ODE 

import matplotlib.pyplot as plt 
import numpy as np 

# define the analytic solution 
def x(t): 
    return -5 * (-10 * t + t**2)
    


# define the given function 
def relaxation_method(a, b, xi, xf, h):
    n = int((b - a)/h + 1)
    t_values = np.linspace(a, b, n)
    x_values = np.zeros(n)
    x_values[0] = xi
    x_values[-1] = xf 
    err = 1
    eps = 0.001
    i = 1
    while err >= eps:
        x_last = x_values.copy()
        x_values[1:-1] = 0.5 * (x_values[2:] + x_values[:-2]) + 0.5 * g * h**2
        err = np.linalg.norm(x_values - x_last)
        if i in [500, 1500, 3000, 5000, 7500]:
            plt.plot(t_values, x_values, label = "candidate solution", color = "red")
        i += 1
    return t_values, x_values, i

# initial condition 
a = 0
b = 10
g = 10
xi = 0
xf = 0
h = 0.1
t_values, x_values, i =  relaxation_method(a, b, xi, xf, h)

print(i)

# solution using relaxation methods 
plt.plot(t_values, x_values, label = "final numerical solution", color = "green")

# plot the analytic solution 
plt.plot(t_values, x(t_values), label = "Exact solution", linestyle = "-.", color = "pink")
plt.xlabel('t')
plt.ylabel('x')
plt.title('Solution of the boundary value problem using Relaxation method')
plt.legend()
plt.grid(True)
plt.show()