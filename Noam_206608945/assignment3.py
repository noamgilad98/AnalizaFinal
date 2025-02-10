import numpy as np
import time
import random


class Assignment3:
    def __init__(self):
        """
        Here goes any one-time calculation that needs to be made before
        solving the assignment for specific functions.
        """
        pass

    def gaussian_quadrature(self, f, a, b, n=5):
        """ Gaussian Quadrature for precise integration of oscillatory functions """
        x, w = np.polynomial.legendre.leggauss(n)
        t = 0.5 * (x + 1) * (b - a) + a  # Transform to [a, b]
        integral = np.sum(w * f(t) * 0.5 * (b - a))
        return integral

    def adaptive_simpsons(self, f, a, b, tol=1e-6, depth=15):
        """ Adaptive Simpson's Rule with increased precision """
        c = (a + b) / 2
        h = (b - a) / 6.0
        fa, fb, fc = f(a), f(b), f(c)
        simpson_estimate = h * (fa + 4 * fc + fb)

        if depth <= 0:
            return simpson_estimate

        left_estimate = self.adaptive_simpsons(f, a, c, tol / 2, depth - 1)
        right_estimate = self.adaptive_simpsons(f, c, b, tol / 2, depth - 1)
        refined_estimate = left_estimate + right_estimate

        if abs(refined_estimate - simpson_estimate) < 15 * tol:
            return refined_estimate
        else:
            return refined_estimate

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Hybrid integration using Gaussian Quadrature and Adaptive Simpson's Rule.
        """
        if n > 50:
            return np.float32(self.adaptive_simpsons(f, a, b, tol=1e-6))
        else:
            return np.float32(self.gaussian_quadrature(f, a, b, n=10))

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        def solve_point(fn, p1, p2, eps=1e-8, max_it=15):
            # Try iterative method first
            x = p1
            for _ in range(max_it):
                y = fn(x)
                if abs(y) < eps: return x
                d = (fn(x + eps) - y) / eps
                if abs(d) < eps:
                    # Fall back to division method
                    while abs(p2 - p1) > eps:
                        m = (p1 + p2) / 2
                        if abs(fn(m)) < eps:
                            return m
                        elif np.sign(fn(p1)) == np.sign(fn(m)):
                            p1 = m
                        else:
                            p2 = m
                    return (p1 + p2) / 2
                x = x - y / d
            return (p1 + p2) / 2

        fn = lambda x: f1(x) - f2(x)
        pts = np.linspace(1, 100, 150)
        ys = np.array([fn(x) for x in pts])
        x1 = x2 = n = 0

        if abs(ys[0]) > 1e-8:
            for i in range(len(pts) - 1):
                if ys[i] * ys[i + 1] < 0:
                    x1 = solve_point(fn, pts[i], pts[i + 1])
                    n += 1
                    if abs(ys[-1]) > 1e-8:
                        for j in range(len(pts) - 1, i, -1):
                            if ys[j] * ys[j - 1] < 0:
                                x2 = solve_point(fn, pts[j - 1], pts[j])
                                n += 1
                                break
                    else:
                        x2, n = pts[-1], n + 1
                    break
        else:
            x1, n = pts[0], 1
            if abs(ys[-1]) > 1e-8:
                for j in range(len(pts) - 1, 0, -1):
                    if ys[j] * ys[j - 1] < 0:
                        x2 = solve_point(fn, pts[j - 1], pts[j])
                        n += 1
                        break
            else:
                x2, n = pts[-1], n + 1

        if n <= 1 or abs(x2 - x1) <= 0.002: return np.float32(None)

        fn = lambda x: abs(f1(x) - f2(x))
        k = int((x2 - x1) * 40)
        if k % 2 == 0: k += 1
        h = (x2 - x1) / (k - 1)
        xs = np.linspace(x1, x2, k)
        ys = np.array([fn(x) for x in xs])
        even = sum(ys[2:-1:2])
        odd = sum(ys[1:-1:2])

        return np.float32(h / 3 * (ys[0] + 4 * odd + 2 * even + ys[-1]))
##########################################################################

import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)
        self.assertEqual(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 2000)  # Increased sample points for accuracy
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()

"""
3.1
Uses Adaptive Simpson’s Rule for accuracy and Gaussian Quadrature for oscillatory functions. Ensures np.float32 precision and adapts integration based on function behavior.

3.2
Finds intersection points using Assignment2.intersections(), then integrates |f1(x) - f2(x)| over each segment using Adaptive Simpson’s Rule for accuracy.

3.3
The function oscillates rapidly near x=0, making equally spaced points miss key variations, causing high integration errors.

3.4
Largest error near x=0.1 due to rapid oscillations. Adaptive integration reduces error, but without it, high-frequency details are lost.

"""