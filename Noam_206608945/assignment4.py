import numpy as np
import time
import random


class Assignment4:
    def __init__(self):
        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        start = time.time()
        f(a)
        call_time = max(time.time() - start, 0.001)

        safe_time = maxtime * 0.85
        n_samples = min(int(safe_time / call_time), 200)

        # Handle edge cases
        if abs(b - a) < 1e-6:
            y_vals = [f(a) for _ in range(min(10, n_samples))]
            return lambda x: np.median(y_vals)  # More robust than mean for noisy data

        # Adaptive sampling - more points where function changes rapidly
        x_initial = np.linspace(a, b, min(20, n_samples))
        y_initial = np.array([f(x) for x in x_initial])
        differences = np.abs(np.diff(y_initial))
        weights = np.concatenate([differences, [differences[-1]]])
        weights = weights / np.sum(weights)

        # Generate points with higher density in high-variation regions
        extra_points = n_samples - len(x_initial)
        if extra_points > 0:
            additional_x = np.random.choice(x_initial, size=extra_points, p=weights)
            noise = (b - a) * 0.005 * np.random.randn(extra_points)
            additional_x = np.clip(additional_x + noise, a, b)
            x_points = np.sort(np.concatenate([x_initial, additional_x]))
            y_points = np.array([f(x) for x in x_points])
        else:
            x_points, y_points = x_initial, y_initial

        # Normalize with robust scaling
        x_center = (a + b) / 2
        x_range = max(abs(b - a), 1e-6)
        x_scaled = (x_points - x_center) / (x_range / 2)

        y_median = np.median(y_points)
        y_range = np.percentile(np.abs(y_points - y_median), 95)
        y_scaled = (y_points - y_median) / (y_range + 1e-6)

        best_coeffs = np.zeros(d + 1)
        best_error = float('inf')

        # Multiple fitting attempts with different strategies
        remaining_time = maxtime - (time.time() - start)
        n_attempts = min(15, max(5, int(remaining_time / 0.1)))

        for attempt in range(n_attempts):
            if time.time() - start > maxtime * 0.9:
                break

            # Vary subset size based on attempt number
            subset_ratio = 0.5 + 0.3 * (attempt / (n_attempts - 1))
            subset_size = max(20, int(len(x_points) * subset_ratio))
            indices = np.random.choice(len(x_points), subset_size, replace=False)

            x_subset = x_scaled[indices]
            y_subset = y_scaled[indices]

            # Progressive fitting with regularization
            coeffs = np.zeros(d + 1)
            coeffs[0] = np.median(y_subset)

            # Adaptive regularization
            reg_strength = 1e-6 * (0.5 ** attempt)

            for degree in range(1, d + 1):
                for _ in range(3):  # Multiple passes for each degree
                    residuals = y_subset - sum(c * x_subset ** i for i, c in enumerate(coeffs))
                    x_power = x_subset ** degree
                    denom = np.mean(x_power * x_power) + reg_strength
                    update = np.mean(residuals * x_power) / denom
                    coeffs[degree] = update

                    # Damping for higher degrees
                    if degree > 2:
                        coeffs[degree] *= 0.95

            # Evaluate on full dataset
            predictions = sum(c * x_scaled ** i for i, c in enumerate(coeffs))
            current_error = np.mean(np.abs(predictions - y_scaled))

            if current_error < best_error:
                best_error = current_error
                best_coeffs = coeffs.copy()

        def fitted_function(x):
            x_norm = (x - x_center) / (x_range / 2)
            result = sum(c * x_norm ** i for i, c in enumerate(best_coeffs))
            return result * y_range + y_median

        return fitted_function