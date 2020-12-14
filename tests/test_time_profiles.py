"""Docstring"""

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
