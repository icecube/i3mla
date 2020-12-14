"""Docstring"""

import unittest

if __name__ == '__main__':
    from context import mla
else:
    from .context import mla
from mla import spectral


class TestBaseSpectrum(unittest.TestCase):
    """Docstring"""


class TestPowerLaw(unittest.TestCase):
    """Docstring"""


class TestCustomSpectrum(unittest.TestCase):
    """Docstring"""


if __name__ == '__main__':
    unittest.main()
