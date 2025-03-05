import numpy as np
import time
import random


class Assignment4:
    def __init__(self):
        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        start_time = time.time()
        n_samples = 100000  # Number of points to sample
        x_values = np.linspace(a, b, n_samples)
        y_values = []

        # Sample noisy data
        for x in x_values:
            if time.time() - start_time > (3 / 4) * maxtime:
                break
            y_values.append(f(x))

        x_values = x_values[:len(y_values)]  # Match sampled points
        y_values = np.array(y_values)

        # Construct Vandermonde matrix for polynomial fitting
        V = np.vander(x_values, d + 1)

        # Solve the least squares problem V * c = y using Gaussian elimination
        VT_V = np.dot(V.T, V)
        VT_y = np.dot(V.T, y_values)

        # Perform Gaussian elimination
        n = VT_V.shape[0]
        A = np.hstack([VT_V, VT_y.reshape(-1, 1)])
        for i in range(n):
            A[i] /= A[i, i]
            for j in range(i + 1, n):
                A[j] -= A[i] * A[j, i]

        coefficients = np.zeros(n)
        for i in range(n - 1, -1, -1):
            coefficients[i] = A[i, -1] - np.sum(A[i, :-1] * coefficients)

        # Define the fitted polynomial function
        def fitted_model(x):
            return sum(c * x ** i for i, c in enumerate(reversed(coefficients)))

        return fitted_model