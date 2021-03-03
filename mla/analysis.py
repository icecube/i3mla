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
import functools
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


def evaluate_ts(analysis: Analysis, events: np.ndarray,
                params: np.ndarray, **kwargs) -> float:
    """Docstring"""
    ts = copy.deepcopy(analysis.test_statistic)
    ts.preprocess(params, events, analysis.model, analysis.source)
    unstructured_params = rf.structured_to_unstructured(params, copy=True)[0]
    return ts(unstructured_params, **kwargs)


def _default_minimizer(
        ts: test_statistics.LLHTestStatistic,
        params: np.ndarray,
        bounds: test_statistics.Bounds = None,
):
    """Docstring"""
    return scipy.optimize.minimize(
        ts, x0=params, bounds=bounds, method='L-BFGS-B')


def minimize_ts(
    analysis: Analysis,
    events: np.ndarray,
    test_params: np.ndarray = np.empty(1, dtype=[('empty', int)]),
    bounds: test_statistics.Bounds = None,
    minimizer: Minimizer = _default_minimizer,
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
    ts = copy.deepcopy(analysis.test_statistic)
    ts.preprocess(
        test_params,
        events,
        analysis.model,
        analysis.source,
        bounds=bounds,
    )

    if verbose:
        print('Preprocessing...', end='')

    if verbose:
        print('done')

    output = {'ts': 0, 'ns': 0}

    if ts.n_kept == 0:
        return output

    ts_partial = functools.partial(ts, **kwargs)

    unstructured_params = rf.structured_to_unstructured(
        test_params,
        copy=True,
    )[0]

    if 'empty' in test_params.dtype.names:
        output['ts'] = -ts_partial(unstructured_params)
        output['ns'] = ts.best_ns

    else:

        if verbose:
            print(f'Minimizing: {test_params}...', end='')

        result = minimizer(ts_partial, unstructured_params, ts.bounds)
        output['ts'] = -result.fun

        res_params = rf.unstructured_to_structured(
            result.x, dtype=test_params.dtype, copy=True)

        if 'ns' not in test_params.dtype.names:
            output['ns'] = ts.best_ns

        for param in test_params.dtype.names:
            output[param] = np.asscalar(res_params[param])

        if verbose:
            print('done')

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
    events = np.concatenate([background, signal])

    return events


def produce_and_minimize(
    analysis: Analysis,
    n_trials: int = 1,
    **kwargs,
) -> List[Dict[str, float]]:
    """Docstring"""
    return [minimize_ts(
        analysis,
        produce_trial(
            analysis,
            **kwargs,
        ),
        **kwargs,
    ) for _ in range(n_trials)]
