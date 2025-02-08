import numpy as np
import time
from typing import Callable, Tuple

class Assignment4:
    def __init__(self):
        pass

    def _sample_points(self, f: Callable[[float], float], a: float, b: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points from the function with uniform spacing.
        """
        x = np.linspace(a, b, num_points)
        y = np.array([f(xi) for xi in x])  # Function evaluation
        return x, y

    def _gaussian_elimination(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve Ax = b using Gaussian elimination without relying on NumPy's built-in solvers.
        """
        n = len(b)
        for i in range(n):
            # Partial pivoting to improve numerical stability
            max_row = i + np.argmax(abs(A[i:, i]))
            if i != max_row:
                A[[i, max_row]] = A[[max_row, i]]
                b[[i, max_row]] = b[[max_row, i]]

            # Make the diagonal element 1
            for j in range(i + 1, n):
                factor = A[j, i] / A[i, i]
                A[j, i:] -= factor * A[i, i:]
                b[j] -= factor * b[i]

        # Back-substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        return x

    def _fit_polynomial(self, x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        """
        Fit a polynomial using normal equations, solved manually via Gaussian elimination.
        """
        A = np.vander(x, degree + 1)
        ATA = A.T @ A
        ATy = A.T @ y
        return self._gaussian_elimination(ATA, ATy)

    def fit(self, f: Callable[[float], float], a: float, b: float, d: int, maxtime: float) -> Callable[[float], float]:
        """
        Build a function that fits the noisy data points sampled from f.
        """
        start_time = time.time()

        # Determine number of sample points
        num_points = min(max(10 * d, 50), 1000)
        x, y = self._sample_points(f, a, b, num_points)

        # If running out of time, return a linear approximation
        if time.time() - start_time > maxtime - 0.2:
            slope = (y[-1] - y[0]) / (x[-1] - x[0])
            intercept = y[0] - slope * x[0]
            return lambda x_val: slope * x_val + intercept

        # Fit polynomial using manually solved normal equations
        coeffs = self._fit_polynomial(x, y, d)

        # Polynomial function using Horner's method
        def fitted_function(x_val: float) -> float:
            result = coeffs[0]
            for c in coeffs[1:]:
                result = result * x_val + c
            return float(result)

        return fitted_function
