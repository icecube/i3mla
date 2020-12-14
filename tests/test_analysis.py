"""Docstring"""

import unittest

if __name__ == '__main__':
    from context import mla
else:
    from .context import mla
from mla import analysis


class TestAnalysis(unittest.TestCase):
    """Docstring"""


class TestPsAnalysis(unittest.TestCase):
    """Docstring"""


class TestTimeDependentPsAnalysis(unittest.TestCase):
    """Docstring"""


class TestThreeMLPsAnalysis(unittest.TestCase):
    """Docstring"""


if __name__ == '__main__':
    unittest.main()
