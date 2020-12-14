"""Docstring"""

import unittest

if __name__ == '__main__':
    from context import mla
else:
    from .context import mla
from mla import models


class TestEventModel(unittest.TestCase):
    """Docstring"""


class TestThreeMLEventModel(unittest.TestCase):
    """Docstring"""


if __name__ == '__main__':
    unittest.main()
