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
            0.024475617,
            0.007580753,
            0.006200465,
            0,
            5,
            0.019201701,
            0.055837773,
            0.016720072,
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

    def grade_assignment_4(self):
        try:
            import assignment4

            names = ('f', 'a', 'b', 'd', 'maxtime')
            valss = [(f2_noise, 0, 5, 10, 5),
                     (f4_noise, -2, 4, 20, 10),
                     (f3_noise, -1, 5, 50, 20),
                     (f13_noise, 3, 10, 20, 15),
                     (f1_noise, 2, 5, 20, 20),
                     (f7_noise, 3, 16, 10, 10),
                     (f8_noise, 1, 3, 10, 15),
                     (f9_noise, 5, 10, 10, 20)
                     ]
            params = [dict(zip(names, vals)) for vals in valss]
            expected_results = [f2, f4, f3, f13, f1, f7, f8, f9]

            func_error = [  # mean absolute error at 2n points within the [a,b] range
                SAVEARGS(f=f, a=a, b=b, n=n, t=t)(
                    lambda fres, fexp, f, a, b, n, t:
                    sum([
                        abs(fres(x) - fexp(x))
                        for x in uniform(low=a, high=b, size=2 * n)
                    ]) / 2 / n
                )
                for f, a, b, n, t in valss
            ]

            repeats = 1

            ass = assignment4.Assignment4()
            self.grade_assignment(ass.fit, params, 'Assignment 4', func_error, expected_results, repeats)
        except Exception as e:
            self.add_error_report('Assignment 4', 'fit', e, 1)

    def report(self):
        with open(os.path.join(self.dir_path, 'res4.csv'), 'w', newline='', encoding='utf-16') as f:
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
        self.grade_assignment_4()
        self.report()
        sys.path.remove(self.dir_path)


if __name__ == '__main__':
    grdr = Grader(os.path.dirname(os.path.abspath(__file__)))
    grdr.grade()