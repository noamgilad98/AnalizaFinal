import time
import numpy as np

class Assignment4:
    def __init__(self):
        pass

    def gaussian_elimination(self, matrix: np.ndarray) -> np.ndarray:
        size = matrix.shape[0]
        for row in range(size):
            matrix[row] /= matrix[row, row]
            for below in range(row + 1, size):
                matrix[below] -= matrix[row] * matrix[below, row]
        return matrix

    def back_substitution(self, matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]
        result = np.zeros(n)
        for i in range(n - 1, -1, -1):
            result[i] = matrix[i, -1] - np.sum(matrix[i, :-1] * result)
        return result

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        start = time.time()
        points = np.linspace(a, b, 100000)
        values = []

        i = 0
        while i < len(points) and (time.time() - start) < (0.8 * maxtime):
            values.append(f(points[i]))
            i += 1

        points = points[:len(values)]
        values = np.array(values)
        vander = np.vander(points, d + 1)
        lhs = vander.T @ vander
        rhs = vander.T @ values
        augmented = np.column_stack((lhs, rhs))
        reduced = self.gaussian_elimination(augmented)
        coeffs = self.back_substitution(reduced)

        def polynomial_model(x):
            return sum(coef * x ** exp for exp, coef in enumerate(coeffs[::-1]))

        return polynomial_model
