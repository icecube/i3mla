"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

import unittest
from unittest import mock

import numpy as np

from context import mla
from mla import injectors


def inject_n_signal_events(injector: injectors.PsInjector, n_signal: int,
                           *args, **kwargs) -> np.ndarray:
    """Docstring"""
    with mock.patch('mla.scipy.stats.poisson.rvs', return_value=n_signal):
        return injector.inject_signal_events(*args, **kwargs)


class TestPsInjector(unittest.TestCase):
    """Docstring"""

    def test_ns_bias(self):
        """Docstring"""


class TestTimeDependentPsInjector(unittest.TestCase):
    """Docstring"""

    def test_ns_bias(self):
        """Docstring"""


if __name__ == '__main__':
    unittest.main()
