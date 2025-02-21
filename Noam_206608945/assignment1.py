import numpy as np


class Assignment1:
    def __init__(self):
        pass

    def _get_chebyshev_points(self, a: float, b: float, n: int) -> np.ndarray:
        k = np.arange(n)
        points = np.cos((2 * k + 1) * np.pi / (2 * n))
        return 0.5 * (b - a) * points + 0.5 * (b + a)

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        # Get Chebyshev nodes
        x = self._get_chebyshev_points(a, b, n)
        y = np.array([f(xi) for xi in x])

        # Compute barycentric weights in O(n) complexity
        w = np.zeros(n)
        for i in range(n):
            # Direct computation of barycentric weights for Chebyshev points
            w[i] = (-1) ** i * np.sin((2 * i + 1) * np.pi / (2 * n))

        def interpolant(x_eval):
            # Handle exact node matches
            if isinstance(x_eval, (float, int)):
                idx = np.abs(x - x_eval) < 1e-14
                if np.any(idx):
                    return y[idx][0]

                # Compute interpolation
                diff = x_eval - x
                weights = w / diff
                return np.sum(weights * y) / np.sum(weights)
            else:
                return np.array([interpolant(xe) for xe in x_eval])

        return interpolant