# solution of system of differential equation with RK4 method 

import matplotlib.pyplot as plt 
import numpy as np 

# define given differential equation 
def f(t, u):
    u1, u2, u3 = u
    du1dt = u1 + 2*u2 - 2*u3 + np.exp(-t)
    du2dt = u2 + u3 - 2 * np.exp(-t)
    du3dt = u1 + 2 * u2 + np.exp(-t)
    return [du1dt, du2dt, du3dt]

# forth order runge kutta method 
def rk4(f, u0, a, b, h):
    n = int((b-a)/h + 1)
    t_values = np.linspace(a, b, n)
    u_values = np.zeros((n, len(u0)))
    u_values[0] = u0
    for i in range(1, n):
        k1 = f(t_values[i - 1], u_values[i - 1])
        k2 = f(t_values[i - 1] + 0.5 * h, u_values[i - 1] + 0.5 * h * np.array(k1))
        k3 = f(t_values[i - 1] + 0.5 * h, u_values[i - 1] + 0.5 * h * np.array(k2))
        k4 = f(t_values[i - 1] + h, u_values[i - 1] + h * np.array(k3))
        u_values[i] = u_values[i - 1] + h * (np.array(k1) + 2 * np.array(k2) + 2 * np.array(k3) + np.array(k4)) / 6

    return t_values, u_values

# initial condition
u0 = [3, -1, 1]
a = 0
b = 1
h = 0.1

# solve using runge kutta method 
t_values, u_values = rk4(f, u0, a, b, h)

# plot the solution 
plt.plot(t_values, u_values[:, 0], label = "u1")
plt.plot(t_values, u_values[:, 1], label = "u2")
plt.plot(t_values, u_values[:, 2], label = "u3")
plt.xlabel('t')
plt.ylabel('u')
plt.title('Solution of the Initial Value Problem')
plt.legend()
plt.grid(True)
plt.show()


