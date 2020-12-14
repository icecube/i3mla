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

if __name__ == '__main__':
    from context import mla
else:
    from .context import mla
from mla import time_profiles


class TestGenericProfile(unittest.TestCase):
    """Docstring"""


class TestGaussProfile(unittest.TestCase):
    """Docstring"""


class TestUniformProfile(unittest.TestCase):
    """Docstring"""


class TestCustomProfile(unittest.TestCase):
    """Docstring"""


if __name__ == '__main__':
    unittest.main()
