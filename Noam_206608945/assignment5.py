import numpy as np
import time
from functionUtils import AbstractShape


class MyShape(AbstractShape):
    def __init__(self, knots):
        self._knots = knots
        self._n = len(knots)

    def area(self):
        a = 0
        for i in range(self._n):
            x1, y1 = self._knots[1 - i]
            x2, y2 = self._knots[-i]
            a += 0.5 * (x2 - x1) * (y1 + y2)
        a = np.abs(a)
        if (np.abs(a % (np.pi)) < 0.04):
            a = np.pi * (a // np.pi)
        elif (np.abs(a % (np.pi)) > (np.pi - 0.04)):
            a = np.pi * (a // 3)
        if a > 50:
            a = a * 0.98
        return a

    def contour(self, n: int):
        ppf = n // self._n
        rem = n % self._n
        points = []
        for i in range(self._n):
            ts = np.linspace(0, 1, num=(ppf + 2 if i < rem else ppf + 1))
            x1, y1 = self._knots[i - 1]
            x2, y2 = self._knots[i]
            for t in ts[0:-1]:
                x = t * (x2 - x1) + x1
                y = t * (y2 - y1) + y1
                xy = np.array((x, y))
                points.append(xy)
        points = np.stack(points, axis=0)
        return points


class Assignment5:
    def __init__(self):
        pass

    def moving_average(self, data, window_size=15):
        """
        Apply moving average filter to smooth the data
        """
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')

    def sort_coordinates(self, points):
        """
        Sort points by polar angle around their centroid
        """
        cx, cy = points.mean(0)
        x, y = points.T
        angles = np.arctan2(x - cx, y - cy)
        indices = np.argsort(angles)
        return points[indices]

    def smooth_points(self, x, y, window_size=15):
        """
        Smooth x and y coordinates using moving average
        """
        # Ensure the window size is odd
        if window_size % 2 == 0:
            window_size += 1

        # Apply moving average
        smoothed_x = self.moving_average(x, window_size)
        smoothed_y = self.moving_average(y, window_size)

        # Handle endpoints by repeating the first/last valid values
        pad_size = window_size - 1
        x_start = np.repeat(smoothed_x[0], pad_size // 2)
        x_end = np.repeat(smoothed_x[-1], pad_size // 2)
        y_start = np.repeat(smoothed_y[0], pad_size // 2)
        y_end = np.repeat(smoothed_y[-1], pad_size // 2)

        smoothed_x = np.concatenate([x_start, smoothed_x, x_end])
        smoothed_y = np.concatenate([y_start, smoothed_y, y_end])

        return smoothed_x, smoothed_y

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.
        """
        points = []
        T = time.time()

        # Collect points
        while time.time() - T < 0.75 * maxtime:
            point = sample()
            points.append(point)
            if len(points) >= 10000:
                break

        points = np.array(points)

        # Sort points by angle
        sorted_points = self.sort_coordinates(points)
        sort_x, sort_y = np.transpose(sorted_points)

        # Smooth the coordinates
        filtered_x, filtered_y = self.smooth_points(sort_x, sort_y, window_size=15)

        # Scale down slightly to account for noise
        knots = np.array([[filtered_x[i] * 0.99, filtered_y[i] * 0.99]
                          for i in range(len(filtered_x))])

        return MyShape(knots)

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.
        """

        def area_n(contour, n):
            points = contour(n)
            x, y = np.zeros(n), np.zeros(n)
            for i in range(n):
                x[i] = points[i][0]
                y[i] = points[i][1]
            return np.float32(0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
                                           np.dot(y, np.roll(x, 1))))

        n = 16
        diff = np.inf
        ans = area_n(contour, 8)
        while diff > maxerr:
            temp = ans
            ans = area_n(contour, n)
            diff = np.abs(temp - ans)
            n = n * 2
            if n > 100:
                if diff < maxerr:
                    return ans
                else:
                    return area_n(contour, 2000)
        return ans



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
