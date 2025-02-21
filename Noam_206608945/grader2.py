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

def test_roots(f1, f2, res, gt, maxerr):
    err = 0

    # count non roots
    res = [x for x in res]
    print(res)

    for x in res:
        if (abs(f1(x) - f2(x)) > maxerr):
            err += 1
            print("ERR:", x, f1(x), f2(x), abs(f1(x) - f2(x)))

    # count missing roots
    gt = [x for x in gt]

    res = np.array(res)
    for gtr in gt:
        # find the closest entry in res:
        dist = abs(res - gtr)
        i = np.argmin(dist)
        x = float(res[i])
        for y in np.linspace(min(x, gtr), max(x, gtr), 10):
            if abs(f1(y) - f2(y)) > maxerr:
                err += 1
                print("MISS:", gtr, x, abs(f1(x) - f2(x)))
                break
    return err

class Grader:
    def __init__(self, dir_path):
        self.reports = []
        self.dir_path = dir_path
        # Last year's error values
        self.last_year_errors = [
            0.006122, 0.000255, 0.000714, 0.006987, 2.66E-16,
            0.001902, 132.2131, 7.44E-05
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



    def grade_assignment_2(self):
        try:
            import assignment2

            names = ('f1', 'f2', 'a', 'b')
            valss = [(f2_nr, f3_nr, 0.5, 2),
                     (f3_nr, f10, 1, 10),
                     (f1, f2_nr, -2, 5),
                     (f12, f13, -0.5, 1.5)
                     ]
            params = [dict(zip(names, vals)) for vals in valss]

            expected_results = [
                [0.671718, 1.8147],
                [1.62899, 2.69730, 3.725809, 3.7914655],
                [-0.79128, 3.79128],
                [-0.175390, 1.42539]
            ]
            func_error = [  # total number of non roots and missing roots
                SAVEARGS(f1=f1, f2=f2, a=a, b=b)(
                    lambda res, exp, f1, f2, a, b:
                    test_roots(f1, f2, res, exp, 0.001)
                )
                for f1, f2, a, b in valss
            ]
            repeats = 15

            ass = assignment2.Assignment2()
            self.grade_assignment(ass.intersections, params, 'Assignment 2', func_error, expected_results, repeats)
        except Exception as e:
            self.add_error_report('Assignment 2', 'intersections', e, 1)

    def report(self):
        with open(os.path.join(self.dir_path, 'res2.csv'), 'w', newline='', encoding='utf-16') as f:
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
        self.grade_assignment_2()
        self.report()
        sys.path.remove(self.dir_path)


if __name__ == '__main__':
    grdr = Grader(os.path.dirname(os.path.abspath(__file__)))
    grdr.grade()