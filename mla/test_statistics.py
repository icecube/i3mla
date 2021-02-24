"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, ClassVar, List

import dataclasses
import numpy as np
import numpy.lib.recfunctions as rf
import scipy.optimize

from . import sources
from . import _models
from . import _test_statistics


TestStatistic = Callable[[np.ndarray, _test_statistics.Preprocessing], float]


@dataclasses.dataclass
class I3Preprocessing(_test_statistics.TdPreprocessing):
    """Docstring"""
    splines: List[scipy.interpolate.UnivariateSpline]


@dataclasses.dataclass
class I3Preprocessor(_test_statistics.TdPreprocessor):
    """Docstring"""
    factory_type: ClassVar = I3Preprocessing

    def _preprocess(self, event_model: _models.EventModel,
                    source: sources.Source, events: np.ndarray) -> dict:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            event_model: An object containing data and preprocessed parameters.
            source:
            events:
        """
        super_prepro_dict = super()._preprocess(event_model, source, events)

        splines = event_model.log_sob_gamma_splines(
            events[super_prepro_dict['drop_index']])

        return {**super_prepro_dict, 'splines': splines}


def i3_test_statistic(params: np.ndarray,
                      prepro: I3Preprocessing,
                      return_ns: bool = False,
                      ns_newton_iters: int = 20) -> float:
    """Evaluates the test-statistic for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params: An array containing (*time_params, gamma).
        prepro:
        return_ns:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """
    temp_params = rf.unstructured_to_structured(
        params, dtype=prepro.params.dtype, copy=True)
    sob = prepro.sob_spatial * _test_statistics.cal_sob_time(
        temp_params, prepro) * np.exp(
        [spline(temp_params['gamma'], ext=3) for spline in prepro.splines])

    ns_ratio = None
    if 'ns' in temp_params.dtype.names:
        ns_ratio = temp_params['ns'] / prepro.n_events
    return _test_statistics.i3_ts(
        sob, prepro, return_ns, ns_ratio, ns_newton_iters=ns_newton_iters)
