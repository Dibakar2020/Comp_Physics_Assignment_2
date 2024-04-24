# fourth order runge kutta method for 1st order ODE 

import matplotlib.pyplot as plt 
import numpy as np 

# define the given differential function 
def f(t, x):
        return 1/(x**2 + t**2)

# define runge kutta fourth order method
def RK4_step(a, b, x0, h):
    n = int((b - a)/h + 1)
    t_values = np.linspace(a, b, n)
    x_values = np.zeros(n)
    x_values[0] = x0
    for i in range(1, n):
        k1 = h * f(t_values[i - 1], x_values[i - 1])
        k2 = h * f(t_values[i - 1] + h/2, x_values[i - 1] + k1/2)
        k3 = h * f(t_values[i - 1] + h/2, x_values[i - 1] + k2/2)
        k4 = h * f(t_values[i - 1] + h, x_values[i - 1] + k3)
        x_values[i] = x_values[i - 1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t_values, x_values

# given initial condition and step size 
a = 0
b = 3.5 * 1e6
x0 = 1
h = 0.1

# RK4 solution 
t_values, x_values = RK4_step(a, b, x0, h) 

# Solution at t = 3.5e6
print("Solution at t = 3.5e6:", x_values[-1])

#plotting the solution 
plt.plot(t_values, x_values, label = "x(t)")
plt.xlabel('t')
plt.ylabel('x')
plt.title("Solution of x'(t)=1/(x^2+t^2)")
plt.grid(True)
plt.legend()
plt.show()

