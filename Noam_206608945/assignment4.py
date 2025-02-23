import numpy as np
import time

class Assignment4:
    def __init__(self):
        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        def gauss_seidel_solver(A, b, max_iter=1000, tolerance=1e-6):
            n = len(b)
            x = np.zeros_like(b)
            for _ in range(max_iter):
                x_new = np.copy(x)
                for i in range(n):
                    sum1 = np.dot(A[i, :i], x_new[:i])
                    sum2 = np.dot(A[i, i + 1:], x[i + 1:])
                    x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
                if np.linalg.norm(x - x_new, ord=2) < tolerance:
                    break
                x = x_new
            return x

        def polynomial_fit(f, a, b, degree, num_points):
            x_samples = np.linspace(a, b, num_points)
            y_samples = np.array([f(x) for x in x_samples])
            X = np.vstack([x_samples**i for i in range(degree + 1)]).T
            XT = X.T
            XTX = XT @ X
            XTy = XT @ y_samples
            coeffs = gauss_seidel_solver(XTX, XTy)

            def poly(x):
                result = 0.0
                for c in reversed(coeffs):
                    result = result * x + c
                return result

            return poly

        start_time = time.time()
        f(a)
        sample_duration = max(time.time() - start_time, 0.001)

        max_points = max(int((maxtime * 0.85) / sample_duration), 50)
        degree = max(d, 1)
        points_per_segment = max_points // degree

        if abs(b - a) < 1e-6:
            avg_value = np.mean([f(a) for _ in range(points_per_segment)])
            return lambda x: avg_value

        polynomials = []
        segment_boundaries = np.linspace(a, b, degree + 1)
        for i in range(degree):
            poly = polynomial_fit(f, segment_boundaries[i], segment_boundaries[i + 1], degree, points_per_segment)
            polynomials.append(poly)

        def combined_polynomial(x):
            segment = min(max(int((x - a) / (b - a) * degree), 0), degree - 1)
            return polynomials[segment](x)

        return combined_polynomial

    def fit_shape(self, generator: callable, maxtime: float) -> 'MyShape':
        class MyShape:
            def __init__(self, points):
                self.points = points

            def contour(self, n):
                return np.array([self.points[i % len(self.points)] for i in range(n)])

        start_time = time.time()
        points = []
        while time.time() - start_time < maxtime:
            points.append(generator())
        return MyShape(points)