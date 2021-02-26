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


SobFunc = Callable[[np.ndarray, _test_statistics.Preprocessing], np.array]
TestStatistic = Callable[[
    np.ndarray,
    _test_statistics.Preprocessing,
    SobFunc,
    bool,
    int,
], float]


@dataclasses.dataclass
class I3Preprocessing(_test_statistics.TdPreprocessing):
    """Docstring"""
    event_model: Optional[_models.EventModel] = None
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

        splines = event_model.log_sob_gamma_splines(
            events[super_prepro_dict['drop_index']])
        
        event_spline_idxs, splines = event_model.log_sob_spline_prepro(
            events[super_prepro_dict['drop_index']],
        )

        return {
            **super_prepro_dict,
            'splines': splines,
            'gamma': self.gamma,
            'event_spline_idxs': event_spline_idxs,
            'splines': splines,
            'event_model': event_model,
        }


def _get_sob_energy(params: np.ndarray, prepro: I3Preprocessing) -> np.array:
    """Docstring"""
    if 'gamma' in params.dtype.names:
        gamma = params['gamma']
    else:
        gamma = prepro.gamma
    return np.maximum(
        np.exp([spline(gamma) for spline in prepro.splines]),
        0,
    )


def i3_sob(params: np.ndarray, prepro: I3Preprocessing) -> np.array:
    """Docstring"""
    sob = prepro.sob_spatial.copy()
    sob *= _test_statistics.get_sob_time(params, prepro)
    #sob *= _get_sob_energy(params, prepro)
    sob *= prepro.event_model.get_sob_energy(params, prepro)
    return sob


def llh_test_statistic(params: np.ndarray,
                       prepro: _test_statistics.Preprocessing,
                       sob_func: SobFunc,
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

    sob = sob_func(temp_params, prepro)

    ns_ratio = _test_statistics.get_ns_ratio(
        sob, prepro, temp_params, ns_newton_iters)

    if return_ns:
        return ns_ratio * prepro.n_events

    if prepro.n_events == 0:
        return 0

    llh, drop_term = _test_statistics.get_i3_llh(sob, ns_ratio)
    return -2 * (llh.sum() + prepro.n_dropped * drop_term)
