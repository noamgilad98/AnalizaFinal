import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        def calc_derivative(func, point):
            delta = 1e-6
            return (func(point + delta) - func(point)) / delta

        def newton_iteration(func, start, end_point, error_bound, iterations):
            x = start
            for _ in range(iterations):
                y_val = func(x)
                if abs(y_val) < error_bound:
                    return x

                d = calc_derivative(func, x)
                if d == 0:
                    return None

                x = x - y_val / d
                if not (start < x < end_point):
                    return None

            return None

        def search_segment(func, start, end, step_size, error_bound, iterations):
            found = np.array([])
            x_points = np.arange(start, end, step_size)

            if x_points[-1] < end:
                x_points = np.append(x_points, end)

            for i in range(len(x_points) - 1):
                result = newton_iteration(func, x_points[i], x_points[i + 1], error_bound, iterations)
                if result is not None:
                    found = np.append(found, result)

            return found

        def search_range(func, start, end, error_bound):
            found = np.array([])
            range_size = abs(end - start)

            if range_size <= 2:
                return search_segment(func, start, end, 0.01, error_bound, 5)

            if range_size <= 10:
                segments = np.linspace(start, end, 6)
                for i in range(5):
                    curr = segments[i]
                    values = np.array([abs(func(curr + j * 0.15)) for j in range(4)])
                    diff = values.max() - values.min()

                    step = 0.02 if diff > 1.5 and abs(values.mean()) <= 2 else 0.1
                    found = np.append(found, search_segment(func, segments[i], segments[i + 1], step, error_bound, 10))
                return found

            if range_size <= 50:
                segments = np.linspace(start, end, 15)
                for i in range(14):
                    curr = segments[i]
                    values = np.array([abs(func(curr + j * 0.001)) for j in range(4)])
                    diff = values.max() - values.min()
                    deriv = calc_derivative(func, curr)

                    if diff > 0.5 and abs(values.mean()) <= 1.5:
                        step = 0.02
                        iters = 10
                    elif (diff < 0.5 and deriv > 0 and func(curr) > 10) or (
                            diff < 0.5 and deriv < 0 and func(curr) < 10):
                        step = 1
                        iters = 20
                    else:
                        step = 0.2
                        iters = 20
                    found = np.append(found,
                                      search_segment(func, segments[i], segments[i + 1], step, error_bound, iters))
                return found

            segments = np.linspace(start, end, 25)
            for i in range(24):
                curr = segments[i]
                values = np.array([abs(func(curr + j * 0.001)) for j in range(4)])
                diff = values.max() - values.min()
                deriv = calc_derivative(func, curr)

                if diff > 1 and abs(values.mean()) <= 1.5:
                    step = 0.02
                    iters = 10
                elif (diff < 0.5 and deriv > 0 and func(curr) > 15) or (diff < 0.5 and deriv < 0 and func(curr) < 15):
                    step = 3
                    iters = 50
                else:
                    step = 1
                    iters = 50
                found = np.append(found, search_segment(func, segments[i], segments[i + 1], step, error_bound, iters))
            return found

        return search_range(lambda x: f1(x) - f2(x), a, b, maxerr)