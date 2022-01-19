"""
Top-level analysis code, and functions that are generic enough to not belong
in any class.
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Dict, List, Optional, Union
from typing import TYPE_CHECKING

import functools
import dataclasses

import numpy as np
import numpy.lib.recfunctions as rf
import scipy.optimize

if TYPE_CHECKING:
    from .test_statistics import LLHTestStatistic, LLHTestStatisticFactory
    from .params import Params
else:
    LLHTestStatistic = object  # pylint: disable=invalid-name
    LLHTestStatisticFactory = object  # pylint: disable=invalid-name
    Params = object  # pylint: disable=invalid-name

@dataclasses.dataclass
class Minimizer:
    """Docstring"""
    config: dict
    test_statistic: LLHTestStatistic

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        return {}


@dataclasses.dataclass
class GridSearchMinimizer(Minimizer):
    """Docstring"""
    def __call__(
        self,
        params: Params,
        events: np.ndarray,
        fitting_params: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Docstring"""
        if fitting_params is None:
            fitting_params = params.names

        if self.test_statistic.n_kept == 0:
            return np.array([(0, 0)], dtype=[('ts', np.float64), ('ns', np.float64)])
            
    @staticmethod
    def _reconstruct_params(params: list, dtype: list) -> np.ndarray:
        """Docstring"""
        return np.array(params, dtype=dtype)

    @staticmethod
    def _destruct_params(params: np.ndarray) -> tuple:
        """Docstring"""
        return rf.structured_to_unstructured(params, copy=True), params.dtype


def _default_minimizer(
    ts,
    unstructured_params,
    unstructured_param_names,
    structured_params,
    bounds=None,
    gridsearch=False,
    gridsearch_points=5,
) -> scipy.optimize.OptimizeResult:
    """Docstring"""
    f = functools.partial(
        _unstructured_ts,
        ts=ts,
        structured_params=structured_params,
        unstructured_param_names=unstructured_param_names,
    )
    x0 = unstructured_params

    if gridsearch:
        grid = (np.linspace(a, b, gridsearch_points)
                for (a, b) in bounds)
        points = np.array(np.meshgrid(*grid)).T
        results = np.zeros(len(points))
        for i, p in enumerate(points):
            results[i] = f(p)
        x0 = points[results.argmin()]  # pylint: disable=unsubscriptable-object
    result = scipy.optimize.minimize(
        f,
        x0=x0,
        bounds=bounds,
        method='L-BFGS-B',
    )
    return result


def minimize_ts(
    test_statistic_factory: LLHTestStatisticFactory,
    events: np.ndarray,
    test_params: np.ndarray = np.empty(1, dtype=[('empty', int)]),
    to_fit: Union[List[str], str, None] = 'all',
    minimizer: Minimizer = _default_minimizer,
) -> Dict[str, float]:
    """Calculates the params that minimize the ts for the given events.

    Accepts guess values for fitting the n_signal and spectral index, and
    bounds on the spectral index. Uses scipy.optimize.minimize() to fit.
    The default method is 'L-BFGS-B', but can be overwritten by passing
    kwargs to this fuction.

    Args:
        test_params:
        events:
        minimizer:

    Returns:
        A dictionary containing the minimized overall test-statistic, the
        best-fit n_signal, and the best fit gamma.
    """
    if to_fit == 'all':
        to_fit = list(test_params.dtype.names)
    elif to_fit is None:
        try:
            test_params = rf.append_fields(
                test_params,
                'empty',
                test_params[test_params.dtype.names[0]],
                usemask=False,
            )
        except ValueError:
            pass
        to_fit = ['empty']
    elif not hasattr(to_fit, '__len__'):
        to_fit = [to_fit]

    ts = test_statistic_factory(test_params[0], events, bounds=bounds)

    if ts.n_kept == 0:
        return np.array(
            [(0, 0)] * len(test_params),
            dtype=[(name, np.float64) for name in ['ts', 'ns']],
        )

    unstructured_params = rf.structured_to_unstructured(test_params[to_fit], copy=True)

    tuple_names = []
    if 'ns' not in to_fit:
        tuple_names.append('ns')
    if to_fit != ['empty']:
        tuple_names.extend(to_fit)

    minimize = functools.partial(
        _minimizer_wrapper,
        unstructured_param_names=to_fit,
        ts=ts,
        minimizer=minimizer,
        tuple_names=tuple_names,
    )

    return_list = [
        minimize(unstructured_params=fit_params, structured_params=params)
        for fit_params, params in zip(unstructured_params, test_params)
    ]

    return np.array(
        return_list,
        dtype=[
            ('ts', np.float64),
            *[(name, np.float64) for name in tuple_names],
        ],
    )


def _minimizer_wrapper(
    unstructured_params: np.array,
    structured_params: np.ndarray,
    unstructured_param_names: List[str],
    ts: LLHTestStatistic,
    minimizer: Minimizer,
    tuple_names: Optional[List[str]] = None,
    **kwargs,
) -> dict:
    """Docstring"""
    output = {}
    for name in structured_params.dtype.names:
        output[name] = structured_params[name]
    ts.update(structured_params)

    if 'empty' in unstructured_param_names:
        output['ts'] = -ts(structured_params, **kwargs)
        output['ns'] = ts.best_ns
    else:
        bounds = [
            bound for i, bound in enumerate(ts.bounds)
            if structured_params.dtype.names[i] in unstructured_param_names
        ]

        result = minimizer(
            ts=ts,
            unstructured_params=unstructured_params,
            unstructured_param_names=unstructured_param_names,
            structured_params=structured_params,
            bounds=bounds,
            **kwargs,
        )

        output['ts'] = -result.fun

        if 'ns' not in unstructured_param_names:
            output['ns'] = ts.best_ns

        for param, val in zip(unstructured_param_names, result.x):
            if param != 'empty':
                output[param] = np.asscalar(val)

    if tuple_names is not None:
        return tuple(
            output[name]
            for name in ['ts', *tuple_names]
        )

    return output
