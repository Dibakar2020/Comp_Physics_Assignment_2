# adaptive step size control with RK4 method

import numpy as np 
import matplotlib.pyplot as plt 

# define given differential function 
def f(t, y):
    return (y**2 + y)/t 

# define value of y for next step using RK4  
def RK4(t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
# define adaptive Runge Kutta method
def adaptive_RK4(t0, y0, t_end, h0, tol):
    t_values = [t0]
    y_values = [y0]
    h = h0
    t = t0
    while t < t_end: 
        y_pred = RK4(t_values[-1], y_values[-1], h) 
        y_step_half = RK4(t_values[-1], y_values[-1], h/2)
        y_step_full = RK4(t_values[-1] + h/2, y_step_half, h/2)
        
        error = np.abs(y_pred - y_step_full)
        if error < tol: 
            t += h
            t_values.append(t)
            y_values.append(y_step_full)
        h = 0.9 * h * (tol / error) ** 0.2

    return np.array(t_values), np.array(y_values)\
    
# given inital condition 
t0 = 1
y0 = -2
t_end = 3
h0 = 1
tol = 1e-4

# solve using adaptive step size control method
t_values, y_values = adaptive_RK4(t0, y0, t_end, h0, tol)

# plotting the solution
plt.plot(t_values, y_values, label = "y(x)")
plt.scatter(t_values, y_values, label = "Mesh points")
plt.title('Solution of y\' = (y^2 + y) / t')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
