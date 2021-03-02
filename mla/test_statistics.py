"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, ClassVar, List, Optional

import dataclasses
import numpy as np
import numpy.lib.recfunctions as rf
import scipy.optimize

from . import sources
from . import _models
from . import _test_statistics


TestStatistic = Callable[[
    np.ndarray,
    _test_statistics.Preprocessing,
    bool,
    int,
], float]


@dataclasses.dataclass
class I3Preprocessing(_test_statistics.TdPreprocessing):
    """Docstring"""
    splines: Optional[List[scipy.interpolate.UnivariateSpline]] = None
    gamma: float = -2
    event_spline_idxs: Optional[np.ndarray] = None
    splines: Optional[np.ndarray] = None


@dataclasses.dataclass
class I3Preprocessor(_test_statistics.TdPreprocessor):
    """Docstring"""
    gamma: float

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

        event_spline_idxs, splines = event_model.log_sob_spline_prepro(
            events[super_prepro_dict['drop_index']],
        )

        return {
            **super_prepro_dict,
            'gamma': self.gamma,
            'event_spline_idxs': event_spline_idxs,
            'splines': splines,
        }


def llh_test_statistic(params: np.ndarray,
                       prepro: _test_statistics.Preprocessing,
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
    if prepro.n_events == 0:
        return 0

    temp_params = rf.unstructured_to_structured(
        params, dtype=prepro.params.dtype, copy=True)

    sob = prepro.sob_func(temp_params, prepro)

    if 'ns' in params.dtype.names:
        ns_ratio = params['ns'] / prepro.n_events
    else:
        ns_ratio = _test_statistics.newton_ns_ratio(
            sob, prepro, ns_newton_iters)

    llh, drop_term = _test_statistics.get_i3_llh(sob, ns_ratio)
    ts = -2 * (llh.sum() + prepro.n_dropped * drop_term)

    if ts < prepro.best_ts:
        prepro.best_ts = ts
        prepro.best_ns = ns_ratio * prepro.n_events

    return ts
