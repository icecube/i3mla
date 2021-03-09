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

from typing import Callable, Dict, List, Optional

import copy
import dataclasses
import numpy as np
import numpy.lib.recfunctions as rf
import scipy.optimize

from . import test_statistics
from . import _models
from . import sources

Minimizer = Callable[
    [
        test_statistics.LLHTestStatistic,
        np.ndarray,
        test_statistics.Bounds,
    ],
    scipy.optimize.OptimizeResult,
]


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
    ts: Callable,
    fit_params: np.ndarray,
    params: np.ndarray,
    bounds: test_statistics.Bounds = None,
) -> scipy.optimize.OptimizeResult:
    """Docstring"""
    return scipy.optimize.minimize(
        ts,
        x0=fit_params,
        args=(params),
        bounds=bounds,
        method='L-BFGS-B',
    )


def minimize_ts(
    analysis: Analysis,
    events: np.ndarray,
    test_params: np.ndarray = np.empty(1, dtype=[('empty', int)]),
    to_fit: Optional[List[str]] = ['all'],
    bounds: test_statistics.Bounds = None,
    minimizer: Minimizer = _default_minimizer,
    ts: Optional[test_statistics.LLHTestStatistic] = None,
    verbose: bool = False,
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
    if to_fit == ['all']:
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

    if verbose:
        print('Preprocessing...', end='')
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
        return [{'ts': 0, 'ns': 0}] * len(test_params)

    unstructured_params = rf.structured_to_unstructured(
        test_params[to_fit],
        copy=True,
    )

    if verbose:
        print('done')

    def fit_ts(fit_params: np.ndarray, params: np.ndarray) -> float:
        """Docstring"""
        for name, val in zip(to_fit, fit_params):
            params[name] = val
        return ts(params, **kwargs)

    to_return = []
    for fit_params, params in zip(unstructured_params, test_params):
        output = {}
        for name in params.dtype.names:
                output[name] = params[name]
        ts.update(params)

        if 'empty' in to_fit:
            output['ts'] = -ts(params, **kwargs)
            output['ns'] = ts.best_ns
        else:
            if verbose:
                print(f'Minimizing: {to_fit}...', end='')

            result = minimizer(fit_ts, fit_params, params, ts.bounds)
            output['ts'] = -result.fun

            if 'ns' not in to_fit:
                output['ns'] = ts.best_ns

            for param, val in zip(to_fit, result.x):
                if param != 'empty':
                    output[param] = np.asscalar(val)

            if verbose:
                print('done')

        to_return.append(output)
    return to_return


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
    events = np.concatenate([background, signal])

    return events


def produce_and_minimize(
    analysis: Analysis,
    n_trials: int = 1,
    **kwargs,
) -> List[Dict[str, float]]:
    """Docstring"""
    ts = copy.deepcopy(analysis.test_statistic)
    return [minimize_ts(
        analysis,
        produce_trial(
            analysis,
            **kwargs,
        ),
        ts=ts,
        **kwargs,
    ) for _ in range(n_trials)]
