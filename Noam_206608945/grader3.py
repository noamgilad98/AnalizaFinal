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
            0.00E+00, 5.09E-06, 4.08E-01, 1.24E-06, 0.00E+00
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

    def grade_assignment_3(self):
        try:
            import assignment3

            names = ('f', 'a', 'b', 'n')
            valss = [(f1, 2, 5, 4),
                     (f2, 2, 10, 10),
                     (f3, 1, 1.5, 4),
                     (f13_nr, 2, 4, 20),
                     (f12_nr, 5, 8, 10)]
            params = [dict(zip(names, vals)) for vals in valss]
            expected_results = [
                15,
                202.6666666666667,
                0.469565,
                35.3333333,
                37.5]
            func_error = [
                SAVEARGS(f=f, a=a, b=b, n=n)(
                    lambda res, expected, f, a, b, n:
                    abs(res - expected)
                )
                for f, a, b, n in valss
            ]

            repeats = 15
            ass = assignment3.Assignment3()
            self.grade_assignment(ass.integrate, params, 'Assignment 3 integrate', func_error, expected_results,
                                  repeats)
        except Exception as e:
            self.add_error_report('Assignment 3', 'integrate', e, 1)

    def grade_assignment_3_areabetween(self):
        try:
            import assignment3

            repeats = 3
            R = RESTRICT_INVOCATIONS
            names = ('f1', 'f2')
            valss = [
                (f10, f2),
            ]
            params = [dict(zip(names, vals)) for vals in valss]
            expected_results = [
                0.731257,
            ]
            func_error = [
                SAVEARGS(f1=f1, f2=f2)(
                    lambda res, expected, f1, f2:
                    abs(res - expected)
                )
                for f1, f2 in valss
            ]

            ass = assignment3.Assignment3()
            self.grade_assignment(ass.areabetween, params, 'Assignment 3 areabetween', func_error, expected_results,
                                  repeats)
        except Exception as e:
            self.add_error_report('Assignment 3 areabetween', 'areabetween', e, 1)

    def report(self):
        with open(os.path.join(self.dir_path, 'res3.csv'), 'w', newline='', encoding='utf-16') as f:
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
        self.grade_assignment_3()
        self.report()
        sys.path.remove(self.dir_path)


if __name__ == '__main__':
    grdr = Grader(os.path.dirname(os.path.abspath(__file__)))
    grdr.grade()