import numpy as np
import time
import random


class Assignment3:
    def __init__(self):
        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        if n == 1:
            mid_point = (b - a) / 2
            return np.float32(f(mid_point) * (b - a))

        if n == 2:
            return np.float32((f(b) + f(a)) * (b - a) / 2)

        if n < 7 or (b - a) < 5:
            if n % 2 == 0:
                n -= 1
            points = np.linspace(a, b, n)
            values = np.array([f(x) for x in points])
            step = (b - a) / (n - 1)

            sum_even = values[0]
            sum_odd = 0
            sum_last = values[-1]

            for i in range(1, n - 1):
                if i % 2 == 0:
                    sum_even += values[i]
                    sum_last += values[i]
                else:
                    sum_odd += values[i]

            return np.float32(step / 3 * (sum_even + 4 * sum_odd + sum_last))

        if n % 2 == 0:
            n -= 1

        interval = 5
        if n > 20:
            temp = int((1 / ((b - a) / n)) * 1.5)
            interval = max(interval, temp)

        sample_points = np.array([[x, None] for x in np.linspace(a, b, n)])

        val1 = f(sample_points[interval][0])
        pos1 = sample_points[interval][0]
        sample_points[interval][1] = val1

        mid_idx = int((n - interval) / 2)
        val2 = f(sample_points[mid_idx][0])
        pos2 = sample_points[mid_idx][0]
        sample_points[mid_idx][1] = val2

        val3 = f(sample_points[-1][0])
        pos3 = sample_points[-1][0]
        sample_points[-1][1] = val3

        evaluated = 3

        if np.abs((val2 - val1) / (pos2 - pos1)) <= 0.25 and np.abs((val3 - val2) / (pos3 - pos2)) <= 0.15:
            area = ((pos3 - pos2) * ((val3 + val2) / 2)) + ((pos2 - pos1) * ((val2 + val1) / 2))
            val4 = f(sample_points[2][0])
            pos4 = sample_points[2][0]
            evaluated += 1

            while (n - evaluated) >= 2:
                area += ((pos1 - pos4) * ((val1 + val4) / 2))
                val1, pos1 = val4, pos4
                pos4 = (pos1 + a) / 2
                val4 = f(pos4)
                evaluated += 1

            area += ((pos1 - pos4) * ((val1 + val4) / 2))
            val1, pos1 = val4, pos4
            val4, pos4 = f(a), a
            area += ((pos1 - pos4) * ((val1 + val4) / 2))

            return np.float32(area)

        for point in sample_points:
            if point[1] is None:
                point[1] = f(point[0])

        values = np.transpose(sample_points)[1]
        step = (b - a) / (n - 1)

        sum_even = values[0]
        sum_odd = 0
        sum_last = values[-1]

        for i in range(1, n - 1):
            if i % 2 == 0:
                sum_even += values[i]
                sum_last += values[i]
            else:
                sum_odd += values[i]

        return np.float32(step / 3 * (sum_even + 4 * sum_odd + sum_last))

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        def calc_slope(f, x):
            h = 1e-6
            return (f(x + h) - f(x)) / h

        def find_root_newton(f, start, maxerr=0.0001, max_steps=15):
            x = start
            for _ in range(max_steps):
                y = f(x)
                if abs(y) < maxerr:
                    return x

                slope = calc_slope(f, x)
                if slope == 0:
                    return None

                x = x - y / slope
            return None

        def find_root_binary(f, start, end, tol=0.0001):
            mid = (start + end) / 2
            if np.abs(f(mid)) < tol:
                return mid
            elif np.sign(f(start)) == np.sign(f(mid)):
                return find_root_binary(f, mid, end, tol)
            elif np.sign(f(end)) == np.sign(f(mid)):
                return find_root_binary(f, start, mid, tol)

        def locate_root(f, start, end):
            root = find_root_newton(f, start)
            if root is None:
                root = find_root_binary(f, start, end)
            return root

        func_diff = lambda x: f1(x) - f2(x)
        x_points = np.linspace(1, 100, 150)
        y_vals = np.array([func_diff(x) for x in x_points])

        root1 = root2 = roots_found = 0

        if y_vals[0] != 0:
            for i in range(len(y_vals) - 1):
                if y_vals[i] * y_vals[i + 1] < 0:
                    root1 = locate_root(func_diff, x_points[i], x_points[i + 1])
                    roots_found += 1

                    if y_vals[-1] != 0:
                        for j in range(len(y_vals) - 1, i, -1):
                            if y_vals[j] * y_vals[j - 1] < 0:
                                root2 = locate_root(func_diff, x_points[j - 1], x_points[j])
                                roots_found += 1
                                break
                    else:
                        root2, roots_found = x_points[-1], roots_found + 1
                    break
        else:
            root1, roots_found = x_points[0], roots_found + 1
            if y_vals[-1] != 0:
                for j in range(len(y_vals) - 1, 0, -1):
                    if y_vals[j] * y_vals[j - 1] < 0:
                        root2 = locate_root(func_diff, x_points[j - 1], x_points[j])
                        roots_found += 1
                        break
            else:
                root2, roots_found = x_points[-1], roots_found + 1

        if roots_found <= 1 or abs(root2 - root1) <= 0.002:
            return np.float32(None)

        abs_diff = lambda x: abs(f1(x) - f2(x))
        n_points = int((root2 - root1) * 40)
        if n_points % 2 == 0:
            n_points += 1

        step = (root2 - root1) / (n_points - 1)
        x_vals = np.linspace(root1, root2, n_points)
        y_vals = np.array([abs_diff(x) for x in x_vals])

        sum_even = y_vals[0]
        sum_odd = 0
        sum_last = y_vals[-1]

        for i in range(1, n_points - 1):
            if i % 2 == 0:
                sum_even += y_vals[i]
                sum_last += y_vals[i]
            else:
                sum_odd += y_vals[i]

        return np.float32(step / 3 * (sum_even + 4 * sum_odd + sum_last))