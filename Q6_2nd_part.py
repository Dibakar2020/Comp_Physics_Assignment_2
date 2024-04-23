import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# define the given function 
def f(t, x):
    g = 10  # gravity constant
    return x[1], -g

# define the analytic solution 
def x(t): 
    return -5 * (-10 * t + t**2)

# define the value of the function at boundary for a given initial slope 
def solution(a, b, x0, z_guess, h):
    t_range = [a, b]
    n = int((b - a)/h + 1)
    initial_value = [x0, z_guess] 
    t_evaluate = np.linspace(a, b, n)  # Use t_range here
    x_values = solve_ivp(lambda t, x: f(t, x), t_range, initial_value, t_eval=t_evaluate)
    x_values = x_values.y[0]
    return x_values

# define shooting method
def shooting(beta, a, b, x0, z_range, h):
    diff = np.zeros(len(z_range))
    for i in range(len(z_range)):
        diff[i] = np.abs(solution(a, b, x0, z_range[i], h)[-1] - beta)
    min_arg = np.argmin(diff)
    z_correct = z_range[min_arg]
    sol = solution(a, b, x0, z_correct, h)
    return sol, z_correct

# given initial condition 
a = 0
b = 10
x0 = 0
beta = 0
h = 0.01
n = int((b - a)/h + 1)
t_range = np.linspace(a, b, n)
z_range = np.arange(0, 60, 0.01)

#solution using numpy.argmin in shooting methods 
sol, z_correct = shooting(beta, a, b, x0, z_range, h)

# plot the solution 
plt.plot(t_range, sol, label='Solution for correct boundary condition', color = "green")  

# plot of the candidate solution 
z_values = [z_correct - 3, z_correct - 2, z_correct - 1, z_correct + 1, z_correct + 2]
for i in range(len(z_values)):
    plt.plot(t_range, solution(a, b, x0, z_values[i], h), linestyle = "--", label = "candidate solution", color = "red")

# plot the analytic solution 
plt.plot(t_range, x(t_range), label = "Analytic solution", linestyle = "dotted", color = "pink")

########
plt.xlabel('t')
plt.ylabel('x')
plt.title('Solution of the boundary value problem using shooting method')
plt.legend()
plt.grid(True)
plt.show()