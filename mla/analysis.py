"""
Top-level analysis code, and functions that are generic enough to not belong
in any class.
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Dict, List, Optional, Union
from typing_extensions import Protocol

import copy
import dataclasses
import functools
import warnings
import numpy as np
import numpy.lib.recfunctions as rf
import scipy.optimize

from . import test_statistics
from . import _models
from . import sources


class Minimizer(Protocol):
    """Docstring"""
    @staticmethod
    def __call__(
        ts: test_statistics.LLHTestStatistic,
        unstructured_params: np.ndarray,
        unstructured_param_names: List[str],
        structured_params: np.ndarray,
        bounds: test_statistics.Bounds = None,
        **kwargs,
    ) -> scipy.optimize.OptimizeResult:
        ...


@dataclasses.dataclass(frozen=True)
class Analysis:
    """Stores the components of an analysis."""
    model: _models.EventModel
    test_statistic: test_statistics.LLHTestStatistic
    source: sources.Source


def generate_params(**kwargs) -> np.ndarray:
    """Docstring"""
    dtype = [(key, np.float64) for key in kwargs]
    max_len = 1

    for key in kwargs:
        if hasattr(kwargs[key], '__len__'):
            max_len = max(max_len, len(kwargs[key]))

    for key in kwargs:
        if not hasattr(kwargs[key], '__len__'):
            kwargs[key] = [kwargs[key]] * max_len

    params = np.empty(max_len, dtype=dtype)
    for key in kwargs:
        params[key][:] = kwargs[key][:]

    return params


def evaluate_ts(analysis: Analysis, events: np.ndarray,
                params: np.ndarray,
                ts: Optional[test_statistics.LLHTestStatistic] = None,
                **kwargs) -> float:
    """Docstring"""
    if ts is None:
        ts = copy.deepcopy(analysis.test_statistic)
    ts.preprocess(params, events, analysis.model, analysis.source)
    return ts(params, **kwargs)


def _default_minimizer(
    ts: test_statistics.LLHTestStatistic,
    unstructured_params: np.ndarray,
    unstructured_param_names: List[str],
    structured_params: np.ndarray,
    bounds: test_statistics.Bounds = None,
    **kwargs,
) -> scipy.optimize.OptimizeResult:
    """Docstring"""
    return scipy.optimize.minimize(
        functools.partial(
            _unstructured_ts,
            ts=ts,
            structured_params=structured_params,
            unstructured_param_names=unstructured_param_names,
            **kwargs,
        ),
        x0=unstructured_params,
        bounds=bounds,
        method='L-BFGS-B',
    )


def _unstructured_ts(
    unstructured_params: np.array,
    ts: test_statistics.LLHTestStatistic,
    structured_params: np.array,
    unstructured_param_names: List[str],
    **kwargs,
) -> float:
    """Docstring"""
    for name, val in zip(unstructured_param_names, unstructured_params):
        structured_params[name] = val

    return ts(structured_params, **kwargs)


def minimize_ts(
    analysis: Analysis,
    events: np.ndarray,
    test_params: np.ndarray = np.empty(1, dtype=[('empty', int)]),
    to_fit: Union[List[str], str, None] = 'all',
    bounds: test_statistics.Bounds = None,
    minimizer: Minimizer = _default_minimizer,
    ts: Optional[test_statistics.LLHTestStatistic] = None,
    verbose: bool = False,
    as_array: bool = False,
    **kwargs,
) -> Dict[str, float]:
    """Calculates the params that minimize the ts for the given events.

    Accepts guess values for fitting the n_signal and spectral index, and
    bounds on the spectral index. Uses scipy.optimize.minimize() to fit.
    The default method is 'L-BFGS-B', but can be overwritten by passing
    kwargs to this fuction.

    Args:
        analysis:
        test_params:
        events:
        minimizer:
        verbose:

    Returns:
        A dictionary containing the minimized overall test-statistic, the
        best-fit n_signal, and the best fit gamma.
    """
    if not as_array:
        warnings.warn(
            ''.join([
                'List[dict] return type will soon be deprecated. ',
                'Set the as_array flag to True to use the np.ndarray ',
                'return type instead.',
            ]),
            FutureWarning,
        )

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

    if verbose:
        print('Preprocessing...', end='', flush=True)

    if ts is None:
        ts = copy.deepcopy(analysis.test_statistic)

    ts.preprocess(
        test_params[0],
        events,
        analysis.model,
        analysis.source,
        bounds=bounds,
    )

    if ts.n_kept == 0:
        if as_array:
            return np.array(
                [(0, 0)] * len(test_params),
                dtype=[(name, np.float64) for name in ['ts', 'ns']],
            )
        else:
            return [{'ts': 0, 'ns': 0}] * len(test_params)

    unstructured_params = rf.structured_to_unstructured(
        test_params[to_fit],
        copy=True,
    )

    if verbose:
        print('done')

    tuple_names = None
    if as_array:
        tuple_names = ['ts']
        if 'ns' not in to_fit:
            tuple_names.append('ns')
        if to_fit != ['empty']:
            tuple_names.extend(to_fit)

    minimize = functools.partial(
        _minimizer_wrapper,
        unstructured_param_names=to_fit,
        ts=ts,
        verbose=verbose,
        minimizer=minimizer,
        tuple_names=tuple_names,
        **kwargs,
    )

    return_list = [
        minimize(unstructured_params=fit_params, structured_params=params)
        for fit_params, params in zip(unstructured_params, test_params)
    ]

    if as_array:
        return np.array(
            return_list,
            dtype=[
                ('ts', np.float64),
                *[(name, np.float64) for name in test_params.dtype.names],
            ],
        )
    return return_list


def _minimizer_wrapper(
    unstructured_params: np.array,
    structured_params: np.ndarray,
    unstructured_param_names: List[str],
    ts: test_statistics.LLHTestStatistic,
    verbose: bool,
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
        if verbose:
            print(
                f'Minimizing: {unstructured_param_names}...',
                end='',
                flush=True,
            )

        result = minimizer(
            ts=ts,
            unstructured_params=unstructured_params,
            unstructured_param_names=unstructured_param_names,
            structured_params=structured_params,
            bounds=ts.bounds,
            **kwargs,
        )

        output['ts'] = -result.fun

        if 'ns' not in unstructured_param_names:
            output['ns'] = ts.best_ns

        for param, val in zip(unstructured_param_names, result.x):
            if param != 'empty':
                output[param] = np.asscalar(val)

        if verbose:
            print('done')

    if tuple_names is not None:
        return tuple(
            output[name]
            for name in ['ts', *structured_params.dtype.names]
        )

    return output


def produce_trial(
        analysis: Analysis,
        flux_norm: float = 0,
        random_seed: Optional[int] = None,
        n_signal_observed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
) -> np.ndarray:
    """Produces a single trial of background+signal events based on inputs.

    Args:
        analysis:
        flux_norm: A flux normaliization to adjust weights.
        random_seed: A seed value for the numpy RNG.
        n_signal_observed:
        verbose: A flag to print progress.

    Returns:
        An array of combined signal and background events.
    """
    # kwargs no-op
    len(kwargs)

    if random_seed is not None:
        np.random.seed(random_seed)

    background = analysis.model.inject_background_events()
    background['time'] = analysis.model.scramble_times(background['time'])

    if flux_norm > 0 or n_signal_observed is not None:
        signal = analysis.model.inject_signal_events(
            analysis.source,
            flux_norm,
            n_signal_observed,
        )

        signal['time'] = analysis.model.scramble_times(
            signal['time'],
            background=False,
        )

    else:
        signal = np.empty(0, dtype=background.dtype)

    if verbose:
        print(f'number of background events: {len(background)}')
        print(f'number of signal events: {len(signal)}')

    # Because we want to return the entire event and not just the
    # number of events, we need to do some numpy magic. Specifically,
    # we need to remove the fields in the simulated events that are
    # not present in the data events. These include the true direction,
    # energy, and 'oneweight'.
    signal = rf.drop_fields(signal, [n for n in signal.dtype.names
                                     if n not in background.dtype.names])

    # Combine the signal background events and time-sort them.
    # Use recfunctions.stack_arrays to prevent numpy from scrambling entry order
    events = rf.stack_arrays([background, signal], autoconvert=True)
    return events


def produce_and_minimize(
    analysis: Analysis,
    n_trials: int = 1,
    as_array: bool = False,
    **kwargs,
) -> List[Dict[str, float]]:
    """Docstring"""
    ts = copy.deepcopy(analysis.test_statistic)
    return_list = [
        minimize_ts(
            analysis,
            produce_trial(
                analysis,
                **kwargs,
            ),
            ts=ts,
            as_array=as_array,
            **kwargs,
        )
        for _ in range(n_trials)
    ]

    if as_array:
        return np.concatenate(return_list)
    return return_list
