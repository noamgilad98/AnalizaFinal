import csv
import gc
import os
import sys
import time
import importlib.util
import numpy as np
import traceback

from numpy.random.mtrand import uniform, seed

from commons import *
import threading
import _thread as thread

from functionUtils import RESTRICT_INVOCATIONS


def quit_function(fn_name):
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush()
    thread.interrupt_main()


def exit_after(s):
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer


@exit_after(60)
def run_func(fnc, args):
    return fnc(*args.values())


class Grader:
    def __init__(self, dir_path):
        self.reports = []
        self.dir_path = dir_path
        # Last year's error values
        self.last_year_errors = [
            0.102471,
            0.101261,
            0.099983,
            0.09998,
            0.100464,
            0.148035,
            0.142357
        ]

    def create_res_file(self, res_path):
        """Creates res.csv with additional columns."""
        fieldnames = ['time', 'error', 'last_year_error', 'error_diff']
        with open(res_path, 'w', newline='', encoding='utf-16') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
            print(f"Created file: {res_path}")

    def grade_assignment(self, to_grade_func, params, assignment_name, err_funcs, expected_results, repeats=1):
        execnum = 0

        for p, result, func_error, last_year_error in zip(
                params, expected_results, err_funcs, self.last_year_errors
        ):
            execnum += 1
            report = {
                'execnum': execnum,
            }

            error = "None"
            res = "None"
            start = time.time()

            try:
                start = time.time()
                for i in range(repeats):
                    gc.collect()
                    res = run_func(to_grade_func, p)
                end = time.time()
                error = func_error(res, result)
                print(res, result, error)
            except (KeyboardInterrupt, Exception) as e:
                end = time.time()
                errors = traceback.format_exc()
                error = str(e)

            report['error'] = error
            report['time'] = end - start
            report['last_year_error'] = last_year_error

            # Calculate error difference
            try:
                report['error_diff'] = float(error) - last_year_error
            except (ValueError, TypeError):
                report['error_diff'] = 'N/A'

            self.reports.append(report)

    def add_error_report(self, assignment, place, error, repeates):
        report = {
            'error': error,
            'time': 'ERROR',
            'last_year_error': 'ERROR',
            'error_diff': 'ERROR'
        }
        self.reports.append(report)

    def grade_assignment_5_area(self):
        try:
            import assignment5

            names = ('contour', 'maxerr')
            valss = [
                (shape1().contour, 0.001),
                (shape2().contour, 0.001),
                (shape3().contour, 0.001),
                (shape4().contour, 0.001),
                (shape5().contour, 0.001),
                (shape6().contour, 0.001),
                (shape7().contour, 0.001),
            ]
            params = [dict(zip(names, vals)) for vals in valss]

            expected_results = [
                shape1().area(),
                shape2().area(),
                shape3().area(),
                shape4().area(),
                shape5().area(),
                shape6().area(),
                shape7().area(),
            ]

            func_error = [
                lambda res, exp:
                abs(abs(res) - abs(exp)) / abs(exp)
                for c, e in valss
            ]

            repeats = 1

            ass = assignment5.Assignment5()
            self.grade_assignment(ass.area, params, 'Assignment 5 area', func_error, expected_results, repeats)

        except Exception as e:
            self.add_error_report('Assignment 5 area', 'area', e, 1)


    def report(self):
        with open(os.path.join(self.dir_path, 'res5area.csv'), 'w', newline='', encoding='utf-16') as f:
            fieldnames = ['time', 'error', 'last_year_error', 'error_diff']
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
            for report in self.reports:
                writer.writerow({
                    'time': report.get('time', 'N/A'),
                    'error': report.get('error', 'N/A'),
                    'last_year_error': report.get('last_year_error', 'N/A'),
                    'error_diff': report.get('error_diff', 'N/A')
                })

    def grade(self):
        self.grade_assignment_5_area()

        self.report()
        sys.path.remove(self.dir_path)


if __name__ == '__main__':
    grdr = Grader(os.path.dirname(os.path.abspath(__file__)))
    grdr.grade()