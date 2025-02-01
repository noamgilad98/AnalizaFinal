import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one-time calculation that needs to be made before
        solving the assignment for specific functions.
        """
        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as possible between f1 and f2.
        Returns points where |f1(x) - f2(x)| <= maxerr.
        """

        def root_secant(f, x1, x2, tol):
            """ Secant method for root-finding. """
            for _ in range(100):  # Max iterations
                f1, f2 = f(x1), f(x2)
                if abs(f1 - f2) < 1e-12:
                    return x1  # Prevent division by zero
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                if abs(x_new - x2) < tol:
                    return x_new
                x1, x2 = x2, x_new
            return x2

        # Define function difference
        def func_diff(x):
            return f1(x) - f2(x)

        # Sample points to detect sign changes
        num_samples = 2000  # Increased sampling density
        x_samples = np.linspace(a, b, num_samples)
        y_samples = np.array([func_diff(x) for x in x_samples])

        # Find intervals where sign change occurs
        intersections = []
        for i in range(len(x_samples) - 1):
            if y_samples[i] * y_samples[i + 1] <= 0:  # Sign change detected
                root = root_secant(func_diff, x_samples[i], x_samples[i + 1], maxerr)
                intersections.append(root)

        return intersections


##########################################################################

import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):
        ass2 = Assignment2()
        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):
        ass2 = Assignment2()
        f1, f2 = randomIntersectingPolynomials(10)
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
