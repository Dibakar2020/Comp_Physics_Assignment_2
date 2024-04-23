import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import bisect
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
def shooting_method(beta, a, b, x0, h):  # Pass a, b, x0, and h as arguments

    def hf(z_guess):
        return solution(a, b, x0, z_guess, h)[-1] - beta
    
    z_correct = bisect(hf, -100, 100)
    return z_correct

# given initial condition 
a = 0
b = 10
x0 = 0
beta = 0
h = 0.01

# solve for correct initial slope 
z_value = round(shooting_method(beta, a, b, x0, h))
print("Solution for x'(0):", z_value)

# integrated ODE with the correct slope 
n = int((b - a)/h + 1)
sol = solve_ivp(lambda t, x: f(t, x), [a , b], [0, z_value], t_eval=np.linspace(a, b, n))

# plot the solution 
plt.plot(sol.t, sol.y[0], label='Solution for correct boundary condition', color = "blue")  

# plot the candidate solutions 
z_values = [z_value - 3, z_value - 2, z_value - 1, z_value + 1, z_value + 2]
for z in z_values:
    candidate_solution = solution(a, b, x0, z, h)
    plt.plot(sol.t, candidate_solution, linestyle='--', label=f'Initial Slope = {z}', color = "red")

# plot the analytic solution 
plt.plot(sol.t, x(sol.t), label = "Analytic solution", linestyle = "dotted", color = "pink")

########
plt.xlabel('t')
plt.ylabel('x')
plt.title('Solution of the boundary value problem using shooting method')
plt.legend()
plt.grid(True)
plt.show()
