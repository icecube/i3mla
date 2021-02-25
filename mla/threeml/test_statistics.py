"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import ClassVar, Optional

import dataclasses
import numpy as np

from .. import _test_statistics
from . import models


@dataclasses.dataclass
class ThreeMLPreprocessing(_test_statistics.TdPreprocessing):
    """Docstring"""
    event_model: Optional[models.ThreeMLEventModel] = None


@dataclasses.dataclass
class ThreeMLPreprocessor(_test_statistics.TdPreprocessor):
    """Docstring"""
    factory_type: ClassVar = ThreeMLPreprocessing


def get_sob(params: np.ndarray, prepro: ThreeMLPreprocessing) -> np.array:
    """Docstring"""
    sob = prepro.sob_spatial.copy()
    sob *= prepro.event_model.get_energy_sob(
        prepro.events[prepro.drop_index])
    sob *= _test_statistics.get_sob_time(params, prepro)
    return sob
