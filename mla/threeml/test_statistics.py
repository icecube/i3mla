"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, Tuple

import dataclasses
import numpy as np

from . import models
from .. import sources
from .. import test_statistics


@dataclasses.dataclass
class ThreeMLEnergyTerm(test_statistics.SoBTerm):
    """Docstring"""
    _energy_sob: Callable = dataclasses.field(init=False)

    def preprocess(
        self,
        params: np.ndarray,
        bounds: test_statistics.Bounds,
        events: np.ndarray,
        event_model: models.ThreeMLEventModel,
        source: sources.Source,
        drop_index: np.ndarray,
    ) -> Tuple[np.ndarray, test_statistics.Bounds]:
        """Docstring"""
        # no-ops
        len(params)
        len(events)
        len(source)

        self._energy_sob = event_model.get_sob_energy
        return drop_index, bounds

    def __call__(
        self,
        params: np.ndarray,
        events: np.ndarray,
        drop_index: np.ndarray,
    ) -> np.ndarray:
        """Docstring"""
        return self._energy_sob(events[drop_index])
