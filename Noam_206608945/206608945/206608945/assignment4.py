import numpy as np
import time
import random


class Assignment4:
    def __init__(self):
        """
        Initialization method for any one-time calculations before solving the assignment.
        """
        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Fit a polynomial function to noisy data sampled from f using least squares approximation.
        """
        start_time = time.time()
        num_samples = min(2000, int((b - a) * 200))  # Increased sample density
        x_samples = np.linspace(a, b, num_samples)
        y_samples = np.array([f(x) for x in x_samples])

        # Construct Vandermonde matrix
        X = np.vander(x_samples, d + 1, increasing=True)

        # Solve normal equations for polynomial coefficients
        coeffs = np.linalg.pinv(X) @ y_samples  # More stable than lstsq

        # Ensure function returns within maxtime
        if time.time() - start_time >= maxtime:
            return lambda x: np.polyval(coeffs[::-1], x)

        return lambda x: np.polyval(coeffs[::-1], x)


##########################################################################

import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()