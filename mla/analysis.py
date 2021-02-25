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

import functools
import dataclasses
import numpy as np
import numpy.lib.recfunctions as rf
import scipy.optimize

from . import sources
from . import test_statistics
from . import _test_statistics
from . import _models


Minimizer = Callable[
    [
        test_statistics.TestStatistic,
        np.ndarray,
        _test_statistics.Preprocessing,
        test_statistics.SobFunc,
        _test_statistics.Bounds,
    ],
    scipy.optimize.OptimizeResult,
]


@dataclasses.dataclass(frozen=True)
class Analysis:
    """Stores the components of an analysis."""
    model: _models.EventModel
    ts_preprocessor: _test_statistics.Preprocessor
    ts_sob: test_statistics.SobFunc
    test_statistic: test_statistics.TestStatistic
    source: sources.Source


def evaluate_ts(analysis: Analysis, events: np.ndarray,
                params: np.ndarray) -> float:
    """Docstring"""
    return analysis.test_statistic(
        params,
        analysis.ts_preprocessor(
            analysis.model, analysis.source, events, params, bounds=None),
    )


def _default_minimizer(ts: test_statistics.TestStatistic,
                       params: np.ndarray,
                       prepro: _test_statistics.Preprocessing,
                       sob: test_statistics.SobFunc,
                       bounds: _test_statistics.Bounds = None):
    """Docstring"""
    with np.errstate(divide='ignore', invalid='ignore'):
        # Set the seed values, which tell the minimizer
        # where to start, and the bounds. First do the
        # shape parameters.
        return scipy.optimize.minimize(
            ts, x0=params, args=(prepro, sob), bounds=bounds, method='L-BFGS-B')


def minimize_ts(
    analysis: Analysis,
    events: np.ndarray,
    test_params: np.ndarray = np.empty(0),
    bounds: _test_statistics.Bounds = None,
    minimizer: Minimizer = _default_minimizer,
    verbose: bool = False,
    ns_newton_iters: int = 20
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
    if verbose:
        print('Preprocessing...', end='')

    prepro = analysis.ts_preprocessor(
        analysis.model, analysis.source, events, test_params, bounds)

    if verbose:
        print('done')

    output = {'ts': 0, 'ns': 0}

    if prepro.n_events - prepro.n_dropped == 0:
        return output

    ts = functools.partial(analysis.test_statistic,
                           ns_newton_iters=ns_newton_iters)

    if len(test_params) != 0:
        params = rf.structured_to_unstructured(prepro.params, copy=True)[0]

        if verbose:
            print(f'Minimizing: {prepro.params}...', end='')

        result = minimizer(ts, params, prepro, analysis.ts_sob, prepro.bounds)
        output['ts'] = -result.fun
        output['ns'] = ts(result.x, prepro, analysis.ts_sob, return_ns=True)

        result.x = rf.unstructured_to_structured(
            result.x, dtype=prepro.params.dtype, copy=True)

        for param in prepro.params.dtype.names:
            output[param] = np.asscalar(result.x[param])

        if verbose:
            print('done')
    else:
        output['ts'] = -ts(params, prepro, analysis.ts_sob)
        output['ns'] = ts(params, prepro, analysis.ts_sob, return_ns=True)

    return output


def produce_trial(analysis: Analysis, flux_norm: float = 0,
                  random_seed: Optional[int] = None,
                  n_signal_observed: Optional[int] = None,
                  verbose: bool = False) -> np.ndarray:
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
    if random_seed is not None:
        np.random.seed(random_seed)

    background = analysis.model.inject_background_events()
    background['time'] = analysis.model.scramble_times(
        background['time'],
        analysis.model.background_time_profile,
    )

    if flux_norm > 0 or n_signal_observed is not None:
        signal = analysis.model.inject_signal_events(analysis.source,
                                                     flux_norm,
                                                     n_signal_observed)
        signal['time'] = analysis.model.scramble_times(
            signal['time'],
            analysis.model.signal_time_profile,
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
    test_params: np.ndarray,
    flux_norm: float = 0,
    bounds: _test_statistics.Bounds = None,
    minimizer: Minimizer = _default_minimizer,
    random_seed: Optional[int] = None,
    n_signal_observed: Optional[int] = None,
    verbose: bool = False,
    n_trials: int = 1,
) -> List[Dict[str, float]]:
    """Docstring"""
    return [minimize_ts(
        analysis,
        test_params,
        produce_trial(
            analysis,
            flux_norm=flux_norm,
            random_seed=random_seed,
            n_signal_observed=n_signal_observed,
            verbose=verbose,
        ),
        bounds=bounds,
        minimizer=minimizer,
        verbose=verbose,
    ) for _ in range(n_trials)]
