import numpy as np


class Assignment1:
    def __init__(self):
        """
        Initialize any precomputed values needed for barycentric interpolation
        """
        pass

    def _get_chebyshev_points(self, a: float, b: float, n: int) -> np.ndarray:
        """
        Generate Chebyshev points in the interval [a,b]
        These points minimize Runge's phenomenon and provide better interpolation
        """
        # Compute Chebyshev nodes in [-1,1]
        k = np.arange(n)
        points = np.cos((2 * k + 1) * np.pi / (2 * n))

        # Scale points to [a,b]
        return 0.5 * (b - a) * points + 0.5 * (b + a)

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n points.

        Parameters:
        -----------
        f : callable - the function to interpolate
        a : float - start of interval
        b : float - end of interval
        n : int - maximum number of points to use

        Returns:
        --------
        callable - the interpolating function
        """
        # Use Chebyshev points for better interpolation
        x = self._get_chebyshev_points(a, b, n)

        # Evaluate function at these points
        y = np.array([f(xi) for xi in x])

        # Precompute weights for barycentric interpolation
        w = np.ones(n)
        for i in range(n):
            for j in range(n):
                if j != i:
                    w[i] *= 1.0 / (x[i] - x[j])

        def interpolant(x_eval):
            """
            Evaluate the interpolant at x_eval using barycentric formula
            """
            # Handle the case when x_eval is exactly one of the nodes
            for i, xi in enumerate(x):
                if np.abs(x_eval - xi) < 1e-14:
                    return y[i]

            # Compute barycentric interpolation
            numer = 0.0
            denom = 0.0
            for i in range(n):
                temp = w[i] / (x_eval - x[i])
                numer += temp * y[i]
                denom += temp

            return numer / denom

        return interpolant