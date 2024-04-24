#include <stdio.h>
#include <math.h>

// Function representing the derivative y' = y - t^2 + 1
double derivative(double t, double y) {
    return y - t * t + 1;
}

// Exact solution y = (t + 1)^2 - 0.5 * e^t
double exact_solution(double t) {
    return pow(t + 1, 2) - 0.5 * exp(t);
}

int main() {
    double t0 = 0; // Initial value of t
    double y0 = 0.5; // Initial value of y
    double h = 0.2; // Step size
    double t, y, y_exact, error, error_bound;

    printf("t\t\tApproximate\tExact\t\tError\t\tError Bound\n");
    printf("----------------------------------------------------------------------------\n");

    // Iterate over each step
    for (t = t0, y = y0; t <= 2; t += h) {
        y += h * derivative(t, y); // Euler's method
        y_exact = exact_solution(t); // Exact solution
        error = fabs(y - y_exact); // Absolute error
        error_bound = 0.2 * fabs(exact_solution(t + h) - exact_solution(t)); // Error bound
        printf("%.2lf\t\t%.6lf\t%.6lf\t%.6lf\t%.6lf\n", t, y, y_exact, error, error_bound);
    }

    return 0;
}



