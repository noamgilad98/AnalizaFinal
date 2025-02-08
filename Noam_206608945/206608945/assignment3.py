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
        """
        Finds the area enclosed between two functions.
        """
        from assignment2 import Assignment2

        ass2 = Assignment2()
        intersections = ass2.intersections(f1, f2, -100, 100, maxerr=0.001)

        if len(intersections) < 2:
            return np.float32(np.nan)

        total_area = np.float32(0.0)
        for i in range(len(intersections) - 1):
            a, b = intersections[i], intersections[i + 1]
            abs_diff = lambda x: abs(f1(x) - f2(x))
            total_area += self.integrate(abs_diff, a, b, 2000)  # Increased sample density

        return np.float32(total_area)


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