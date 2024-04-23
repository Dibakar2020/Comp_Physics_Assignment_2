# solution using scipy.integrate.solve_ivp 

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp

# defining the given differential equation 
def f1(t, y):
    return t * np.exp(3 * t) - 2 * y
def f2(t, y):
    return 1 - (t - y)**2 
def f3(t, y):
    return 1 + y / t
def f4(t, y):
    return np.cos(2 * t) + np.sin(3 * t)

# analytical solution using WolframAlpha
def y1(t):
    return (1/25) * np.exp(-2 * t) * (np.exp(5 * t) * (5 * t - 1) + 1)
def y2(t):
    return (t**2 - 3 * t + 1)/(t - 3)
def y3(t):
    return t * (np.log(t) + 2)
def y4(t):
    return (1/6) * (3 * np.sin(2 * t) - 2 * np.cos(3 * t) + 8)



# define the different time range for different function 
t_span1 = (0, 1)
t_span2 = (2, 3)
t_span3 = (1, 2)
t_span4 = (0, 1)

# define time evalution for different function 
t_eval1 = np.linspace(0, 1, 100)
t_eval2 = np.linspace(2, 3, 100)
t_eval3 = np.linspace(1, 2, 100)
t_eval4 = np.linspace(0, 1, 100)

# difine initial condition for different function 
yi1 = [0]
yi2 = [1]
yi3 = [2]
yi4 = [1]

# solve the initial value problem 
sol1 = solve_ivp(f1, t_span1, yi1, t_eval = t_eval1)
sol2 = solve_ivp(f2, t_span2, yi2, t_eval = t_eval2)
sol3 = solve_ivp(f3, t_span3, yi3, t_eval = t_eval3)
sol4 = solve_ivp(f4, t_span4, yi4, t_eval = t_eval4)

# 

# plots of the solutions 
plt.plot(sol1.t, sol1.y[0], label = "Numerical solution")
plt.plot(t_eval1, y1(t_eval1), label = "Analytical solution", linestyle = "--")
plt.xlabel('x')
plt.ylabel('y')
plt.title("solution of y'=te^(3t)-2y")
plt.grid()
plt.legend()
plt.show()
plt.plot(sol2.t, sol2.y[0], label = "Numerical solution")
plt.plot(t_eval2, y2(t_eval2), label = "Analytical solution", linestyle = "--")
plt.xlabel('x')
plt.ylabel('y')
plt.title("solution of y'=1-(t-y)^2")
plt.grid()
plt.legend()
plt.show()
plt.plot(sol3.t, sol3.y[0], label = "Numerical solution")
plt.plot(t_eval3, y3(t_eval3), label = "Analytical solution", linestyle = "--")
plt.xlabel('x')
plt.ylabel('y')
plt.title("solution of y'=1+y/t")
plt.grid()
plt.legend()
plt.show()
plt.plot(sol4.t, sol4.y[0], label = "Numerical solution")
plt.plot(t_eval4, y4(t_eval4), label = "Analytical solution", linestyle = "--")
plt.xlabel('x')
plt.ylabel('y')
plt.title("solution of y'=cos2t+sin3t")
plt.grid()
plt.legend()
plt.show()