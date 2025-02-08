import csv
import gc
import os
import sys
import traceback
import numpy as np
import threading
import _thread as thread
import time

from numpy.random.mtrand import uniform
from commons import *
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
                result = fn(*args.values())
            finally:
                timer.cancel()
            return result

        return inner

    return outer


@exit_after(60)
def run_func(fnc, args):
    return fnc(*args.values())


def compute_score(X, median_X):
    return min(1, max(0, 1 - np.log((X + 1e-200) / (median_X + 1e-200))))


def test_roots(f1, f2, res, gt, maxerr):
    err = 0
    res = np.array(res)

    for x in res:
        if abs(f1(x) - f2(x)) > maxerr:
            err += 1

    for gtr in gt:
        dist = abs(res - gtr)
        if np.min(dist) > maxerr:
            err += 1
    return err


class Grader():
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.reports = []
        self.old_results = self.load_old_results()

    def load_old_results(self):
        old_results = []
        file_path = os.path.join(self.dir_path, 'res_old.csv')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-16') as f:
                reader = csv.DictReader(f, delimiter='\t')
                old_results = [row for row in reader]
        return old_results

    def grade_assignment(self, to_grade_func, params, assignment_name, err_funcs, expected_results, repeats=1):
        execnum = 0
        current_results = []
        for p, result, func_error in zip(params, expected_results, err_funcs):
            execnum += 1
            report = {'assignemnt': assignment_name, 'function': to_grade_func.__name__, 'execnum': execnum, **p}
            error, res = "None", "None"
            start = time.time()
            try:
                for _ in range(repeats):
                    gc.collect()
                    res = run_func(to_grade_func, p)
                end = time.time()
                error = func_error(res, result)
            except Exception as e:
                end = time.time()
                error = str(e)
            report.update({'output': str(res), 'error': error, 'repeats': repeats, 'time': end - start})
            current_results.append(report)
        self.compute_scores(assignment_name, current_results)
        self.reports.extend(current_results)

    def compute_scores(self, assignment_name, current_results):
        old_results = [r for r in self.old_results if r['assignemnt'] == assignment_name]
        if not old_results:
            return
        time_values = [float(r['time']) for r in old_results]
        error_values = [float(r['error']) if r['error'] != "None" else 1 for r in old_results]
        median_time = np.median(time_values) if time_values else 1
        median_error = np.median(error_values) if error_values else 1

        for report in current_results:
            report['score'] = compute_score(float(report['error']) if report['error'] != "None" else 1, median_error)
            if assignment_name == 'Assignment 3 areabetween':
                report['score'] = (compute_score(float(report['time']), median_time) + 1) * report['score'] / 2
            elif assignment_name == 'Assignment 5 area':
                report['score'] = 0 if float(report['error']) > 0.001 else compute_score(float(report['time']),
                                                                                         median_time)

    def report(self):
        file_path = os.path.join(self.dir_path, 'res.csv')
        with open(file_path, 'w', newline='', encoding='utf-16') as f:
            fieldnames = ['assignemnt', 'function', 'execnum', 'output', 'error', 'repeats', 'time', 'score']
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
            for report in self.reports:
                writer.writerow(report)
        os.rename(file_path, os.path.join(self.dir_path, 'res_old.csv'))

    def grade(self):
        self.grade_assignment_1()
        self.grade_assignment_2()
        self.grade_assignment_3()
        self.grade_assignment_4()
        self.grade_assignment_5_area()
        self.report()
        sys.path.remove(self.dir_path)


if __name__ == '__main__':
    grdr = Grader(os.path.dirname(os.path.abspath(__file__)))
    grdr.grade()
