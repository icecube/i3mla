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

import dataclasses
import numpy as np
import numpy.lib.recfunctions as rf
import scipy.optimize

from . import sources
from . import models
from . import injectors
from . import test_statistics


Minimizer = Callable[
    [test_statistics.TestStatistic, test_statistics.Preprocessing],
    scipy.optimize.OptimizeResult,
]


@dataclasses.dataclass(frozen=True)
class Analysis:
    """Stores the components of an analysis."""
    model: models.EventModel
    injector: injectors.PsInjector
    ts_preprocessor: test_statistics.Preprocessor
    test_statistic: test_statistics.TestStatistic
    source: sources.Source


def evaluate_ts(analysis: Analysis, events: np.ndarray,
                params: np.ndarray) -> float:
    """Docstring"""
    return analysis.test_statistic(
        params,
        analysis.ts_preprocessor(
            analysis.model, analysis.injector, analysis.source, events, params),
    )


def default_minimizer(ts: test_statistics.TestStatistic,
                      prepro: test_statistics.Preprocessing):
    """Docstring"""
    with np.errstate(divide='ignore', invalid='ignore'):
        # Set the seed values, which tell the minimizer
        # where to start, and the bounds. First do the
        # shape parameters.
        return scipy.optimize.minimize(
            ts, x0=prepro.params.values, args=(prepro), bounds=prepro.bounds,
            method='L-BFGS-B'
        )


def minimize_ts(analysis: Analysis, test_params: List[float],
                events: np.ndarray,
                minimizer: Minimizer = default_minimizer,
                verbose: bool = False) -> Dict[str, float]:
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
        analysis.model, analysis.injector, analysis.source, events, test_params)

    if verbose:
        print('done')

    output = {'ts': 0, **prepro.params}

    if len(prepro.events) == 0:
        return output

    if verbose:
        print(f'Minimizing: {prepro.params}...', end='')

    result = minimizer(analysis.test_statistic, prepro)

    if verbose:
        print('done')

    # Store the results in the output array
    output['ts'] = -1 * result.fun
    for i, (param, _) in enumerate(prepro.params):
        output[param] = result.x[i]

    return output


def produce_trial(analysis: Analysis, flux_norm: float = 0,
                  random_seed: Optional[int] = None,
                  grl_filter: bool = True,
                  verbose: bool = False) -> np.ndarray:
    """Produces a single trial of background+signal events based on inputs.

    Args:
        analysis:
        flux_norm: A flux normaliization to adjust weights.
        random_seed: A seed value for the numpy RNG.
        grl_filter: cut out events that is not in grl
        verbose: A flag to print progress.

    Returns:
        An array of combined signal and background events.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    background = analysis.injector.inject_background_events(
        analysis.event_model)

    if flux_norm > 0:
        signal = analysis.injector.inject_signal_events(
            analysis.event_model, analysis.source, flux_norm)
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

    if grl_filter:
        sorting_indices = np.argsort(events['time'])
        events = events[sorting_indices]

        # We need to check to ensure that every event is contained within
        # a good run. If the event happened when we had deadtime (when we
        # were not taking data), then we need to remove it.
        grl_start = analysis.event_model.grl['start'].reshape(
            (1, len(analysis.event_model.grl)))
        grl_stop = analysis.event_model.grl['stop'].reshape(
            (1, len(analysis.event_model.grl)))
        after_start = np.less(grl_start, events['time'].reshape(len(events),
                                                                1))
        before_stop = np.less(events['time'].reshape(len(events), 1),
                              grl_stop)
        during_uptime = np.any(after_start & before_stop, axis=1)
        events = events[during_uptime]

    return events


def produce_and_minimize(
    analysis: Analysis,
    flux_norms: List[float],
    test_params_list: List[List[float]],
    minimizer: Minimizer = default_minimizer,
    random_seed: Optional[int] = None,
    grl_filter: bool = True,
    verbose: bool = False,
    n_trials: int = 1,
) -> List[List[List[Dict[str, float]]]]:
    """Docstring"""
    return [[[
        minimize_ts(
            analysis,
            test_params,
            produce_trial(
                analysis, flux_norm, random_seed, grl_filter, verbose),
            minimizer,
            verbose,
        ) for _ in range(n_trials)
    ] for test_params in test_params_list] for flux_norm in flux_norms]
