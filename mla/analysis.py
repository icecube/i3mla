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

from typing import Callable, Dict, List

import dataclasses
import numpy as np
import scipy.optimize

from . import sources
from . import models
from . import injectors
from . import test_statistics
from . import trial_generators


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
    trial_generator: trial_generators.PsTrialGenerator
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
                minimizer: Minimizer = default_minimizer) -> Dict[str, float]:
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

    Returns:
        A dictionary containing the minimized overall test-statistic, the
        best-fit n_signal, and the best fit gamma.
    """
    prepro = analysis.ts_preprocessor(
        analysis.model, analysis.injector, analysis.source, events, test_params)

    test_stat = analysis.test_statistic

    output = {'ts': 0, **prepro.params}

    if len(prepro.events) == 0:
        return output

    result = minimizer(test_stat, prepro)

    # Store the results in the output array
    output['ts'] = -1 * result.fun
    for i, (param, _) in enumerate(prepro.params):
        output[param] = result.x[i]

    return output


def produce_trial(analysis: Analysis, *args, **kwargs) -> np.ndarray:
    """Docstring"""
    return analysis.trial_generator.generate(
        analysis.model,
        analysis.injector,
        analysis.source,
        analysis.trial_generator.preprocess_trial(
            analysis.model, analysis.source, *args, **kwargs
        ),
        *args,
        **kwargs,
    )
