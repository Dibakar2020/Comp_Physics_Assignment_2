import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Define the differential equations and boundary conditions for each problem

# First Boundary Value Problem
def ode1(x, y):
    return np.vstack((y[1], -np.exp(-2*y[0])))

def bc1(ya, yb):
    return np.array([ya[0], yb[0] - np.log(2)])

# Second Boundary Value Problem
def ode2(x, y):
    return np.vstack((y[1], y[1] * np.cos(x) - y[0] * np.log(np.maximum(y[0], 1e-10))))

def bc2(ya, yb):
    return np.array([ya[0] - 1, yb[0] - np.e])

# Third Boundary Value Problem
def ode3(x, y):
    return np.vstack((y[1], -(2 * y[1]**3 + y[0]**2 * y[1]) * (1/np.cos(x))))

def bc3(ya, yb):
    return np.array([ya[0] - 2**(-1/4), yb[0] - (12**(1/4))/2])

# Fourth Boundary Value Problem
def ode4(x, y):
    return np.vstack((y[1], 1/2 - (y[1]**2)/2 - y[0] * np.sin(x)/2))

def bc4(ya, yb):
    return np.array([ya[0] - 2, yb[0] - 2])


# analytical solution using WolframAlpha
def y1(x):
    return np.log(x)
def y2(x):
    return np.exp(np.sin(x))
def y3(x):
    return (np.sin(x))**0.5
def y4(x):
    return 2 + np.sin(x)


# Define the x values for each problem
x1 = np.linspace(1, 2, 100)
x2 = np.linspace(1e-10, np.pi/2, 100)
x3 = np.linspace(np.pi/4, np.pi/3, 100)
x4 = np.linspace(0, np.pi, 100)

# Initial guess for y and y' for each problem
y_guess1 = np.zeros((2, x1.size))
y_guess2 = np.array([np.ones(100), np.ones(100)])
y_guess3 = np.zeros((2, x3.size))
y_guess4 = np.zeros((2, x4.size))

# Solve each Boundary Value Problem
sol1 = solve_bvp(ode1, bc1, x1, y_guess1)
sol2 = solve_bvp(ode2, bc2, x2, y_guess2)
sol3 = solve_bvp(ode3, bc3, x3, y_guess3)
sol4 = solve_bvp(ode4, bc4, x4, y_guess4)

# Plot solutions

# Plot for First Boundary Value Problem
plt.plot(sol1.x, sol1.y[0], label='y(x)')
plt.plot(x1, y1(x1), label = "Analytical solution", linestyle = "--")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution for y\'\' = -exp(-2y)')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Second Boundary Value Problem
plt.plot(sol2.x, sol2.y[0], label='y(x)')
plt.plot(x2, y2(x2), label = "Analytical solution", linestyle = "--")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution for y\'\' = y\'cos(x) - yln(y)')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Third Boundary Value Problem
plt.plot(sol3.x, sol3.y[0], label='y(x)')
plt.plot(x3, y3(x3), label = "Analytical solution", linestyle = "--")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution for y\'\' = -(2(y\')^3 + y^2y\') sec(x)')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Fourth Boundary Value Problem
plt.plot(sol4.x, sol4.y[0], label='y(x)')
plt.plot(x4, y4(x4), label = "Analytical solution", linestyle = "--")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution for y\'\' = 1/2 - (y\')^2/2 - ysin(x)/2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
