

import numpy as np
import time


class Assignment4:
    def __init__(self):
        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:

        # Start the timer
        begin_time = time.time()

        # Generate x values within the given range
        x_samples = np.linspace(a, b, 100000)
        y_samples = []

        # Collect noisy data points within the time limit
        idx = 0
        while idx < len(x_samples) and (time.time() - begin_time) < (7 / 8) * maxtime:
            y_samples.append(f(x_samples[idx]))
            idx += 1

        # Trim x_samples to match the number of y_samples collected
        x_samples = x_samples[:len(y_samples)]
        y_samples = np.array(y_samples)

        # Build the Vandermonde matrix for polynomial fitting
        vander_matrix = np.vander(x_samples, d + 1)

        # Compute the transpose of the Vandermonde matrix
        vander_transpose = vander_matrix.T

        # Prepare the system of equations for least squares
        A = np.dot(vander_transpose, vander_matrix)
        b = np.dot(vander_transpose, y_samples)

        # Combine A and b into an augmented matrix
        augmented_matrix = np.hstack([A, b.reshape(-1, 1)])

        # Perform Gaussian elimination on the augmented matrix
        rows = augmented_matrix.shape[0]
        for row in range(rows):
            # Normalize the current row
            augmented_matrix[row] /= augmented_matrix[row, row]

            # Eliminate the current variable from subsequent rows
            for next_row in range(row + 1, rows):
                augmented_matrix[next_row] -= augmented_matrix[row] * augmented_matrix[next_row, row]

        # Back-substitution to find the polynomial coefficients
        coeffs = np.zeros(rows)
        for row in range(rows - 1, -1, -1):
            coeffs[row] = augmented_matrix[row, -1] - np.sum(augmented_matrix[row, :-1] * coeffs)

        # Define the fitted polynomial function
        def fitted_poly(x):
            return sum(coeff * x ** power for power, coeff in enumerate(reversed(coeffs)))

        return fitted_poly
