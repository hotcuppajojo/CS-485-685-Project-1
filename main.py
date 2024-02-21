# JoJo Petersky
# CS 485/685 Spring '24 Project1
# 2024/2/20
# main.py

import unittest
import coverage

if __name__ == '__main__':
    cov = coverage.Coverage()
    cov.start()

    loader = unittest.TestLoader()
    suite = loader.discover('tests')

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

    cov.stop()
    cov.save()
    cov.report(show_missing=True)