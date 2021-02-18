"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import ClassVar

import dataclasses
import numpy as np
import numpy.lib.recfunctions as rf

from . import test_statistics
from . import threeml_models


@dataclasses.dataclass
class ThreeMLPreprocessing(test_statistics.TdPreprocessing):
    """Docstring"""
    event_model: threeml_models.ThreeMLEventModel


@dataclasses.dataclass
class ThreeMLPreprocessor(test_statistics.TdPreprocessor):
    """Docstring"""
    factory_type: ClassVar = ThreeMLPreprocessing


def threeml_ps_test_statistic(params: np.ndarray,
                              prepro: ThreeMLPreprocessing,
                              return_ns: bool = False) -> float:

    """(ThreeML version) Evaluates the ts for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params:
        event_model:
        events:
        prepro:
        return_ns:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """
    temp_params = rf.unstructured_to_structured(
        params, dtype=prepro.params.dtype, copy=True)

    sob_energy = prepro.event_model.get_energy_sob(
        prepro.events[prepro['drop_index']])

    sob = prepro.sob_spatial * \
        test_statistics.cal_sob_time(params, prepro) * sob_energy

    ns_ratio = None
    if 'ns' in temp_params.dtype.names:
        ns_ratio = temp_params['ns'] / prepro.n_events
    return test_statistics.i3_ts(sob, prepro, return_ns, ns_ratio)
