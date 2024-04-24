# euler method in 2nd order ODE 
import matplotlib.pyplot as plt 
import numpy as np 

# define given 2nd order differential function 
def f(t, y, z):
    return (t**3 * np.log(t) + 2 * t * z - 2 * y) / t**2

# define the analytical solution 
def y(t):
    return 7*t/4 + (t**3)/2 * np.log(t) - (3/4) * t**3

# define euler method 
def euler_method(a, b, y0, z0, h):
    n = int((b-a)/h + 1)
    t_values = np.linspace(a, b, n)
    y_values = np.zeros(n)
    z_values = np.zeros(n)
    y_values[0] = y0
    z_values[0] = z0
    for i in range(1, n):
        z_values[i] = z_values[i - 1] + h * f(t_values[i - 1], y_values[i - 1], z_values[i - 1])
        y_values[i] = y_values[i - 1] + h * z_values[i]

    return t_values, y_values

# initial condition 
a = 1
b = 2
y0 = 1
z0 = 0
h = 0.001

# solve using euler method 
t_values, y_values =  euler_method(a, b, y0, z0, h)

# plot the solution 
plt.plot(t_values, y_values, label = "Numerical solution")
plt.plot(t_values, y(t_values), label = "Analytical solution", linestyle = "--")
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of the Initial Value Problem')
plt.legend()
plt.grid(True)
plt.show()

