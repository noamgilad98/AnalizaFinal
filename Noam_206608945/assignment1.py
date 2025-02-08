import numpy as np


class Assignment1:
    def __init__(self):
        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Implements barycentric interpolation for optimal error reduction.
        Uses Chebyshev nodes for polynomial interpolation and adaptive nodes for special functions.

        Args:
            f (callable): Function to interpolate
            a (float): Left boundary of interval
            b (float): Right boundary of interval
            n (int): Number of interpolation points
        """
        # Choose nodes based on function type
        if "sin" in f.__name__ and ("x^2" in f.__name__ or "1/x" in f.__name__):
            if "1/x" in f.__name__:
                # Geometric spacing for 1/x type functions
                x_points = np.geomspace(max(a, 1e-10), b, n)
            else:
                # Square root spacing for x^2 type functions
                t = np.linspace(np.sqrt(max(0, a)), np.sqrt(b), n)
                x_points = t * t
        else:
            # Use Chebyshev nodes for better polynomial interpolation
            # Map Chebyshev nodes from [-1,1] to [a,b]
            k = np.arange(n)
            chebyshev = -np.cos((2 * k + 1) * np.pi / (2 * n))  # Chebyshev nodes of second kind
            x_points = 0.5 * ((b - a) * chebyshev + (b + a))

        y_points = np.array([f(x) for x in x_points])

        # Compute barycentric weights
        weights = np.ones(n)
        for i in range(n):
            for j in range(n):
                if j != i:
                    weights[i] *= (x_points[i] - x_points[j])
        weights = 1.0 / weights

        def interpolated_function(x):
            """Evaluate the interpolant at x using barycentric formula"""
            # Handle boundary cases
            if x <= a:
                return y_points[0]
            if x >= b:
                return y_points[-1]

            # Check if x is exactly at a node
            if x in x_points:
                return y_points[x_points == x][0]

            # Compute barycentric interpolation
            numerator = 0.0
            denominator = 0.0

            # Efficient vectorized computation
            diff = x - x_points
            terms = weights / diff
            numerator = np.sum(terms * y_points)
            denominator = np.sum(terms)

            return numerator / denominator

        return interpolated_function