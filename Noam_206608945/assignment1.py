import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one-time calculation that needs to be made before
        starting to interpolate arbitrary functions.
        """
        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.

        Parameters
        ----------
        f : callable. The given function
        a : float - start of the interpolation range.
        b : float - end of the interpolation range.
        n : int - number of sample points allowed.

        Returns
        -------
        A function g(x) that approximates f(x) via interpolation.
        """
        # Use Chebyshev nodes for better accuracy
        cheb_nodes = np.cos((2 * np.arange(n) + 1) / (2 * n) * np.pi) * (b - a) / 2 + (b + a) / 2
        y_points = np.array([f(x) for x in cheb_nodes])  # Sampled function values

        # Compute barycentric weights
        w = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    w[i] /= (cheb_nodes[i] - cheb_nodes[j])

        def interpolated_function(x):
            """
            Uses Barycentric Lagrange interpolation with Chebyshev nodes.
            """
            numer = 0
            denom = 0
            for i in range(n):
                term = w[i] / (x - cheb_nodes[i]) if x != cheb_nodes[i] else float('inf')
                numer += term * y_points[i]
                denom += term
            return numer / denom if denom != 0 else y_points[np.argmin(np.abs(cheb_nodes - x))]

        return interpolated_function


##########################################################################

import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)
            f = np.poly1d(a)
            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200) * 20 - 10  # Scale to [-10,10]
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20) * 20 - 10  # Scale to [-10,10]
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()

"""
The interpolation function uses Chebyshev nodes to minimize Runge’s phenomenon and improve accuracy.
It applies Barycentric Lagrange interpolation, which allows for efficient evaluation in 𝑂(𝑛) time.
The method ensures that the function is sampled at most n times while maintaining a low interpolation error.
The implementation is optimized for both accuracy and speed, making it highly effective for polynomial functions.
"""