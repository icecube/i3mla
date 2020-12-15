"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Optional

import numpy as np
import numpy.lib.recfunctions as rf

from mla import models
from mla import injectors


class PsTrialGenerator:
    """Docstring"""

    def __init__(self) -> None:
        """Docstring"""

    @staticmethod
    def preprocess_trial() -> None:
        """Docstring"""

    @classmethod
    def ps_trial_gen(cls, event_model: models.EventModel,  # pylint: disable=too-many-locals, too-many-arguments
                     injector: injectors.PsInjector, flux_norm: float,
                     reduced_sim: Optional[np.ndarray] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                     gamma: float = -2, sampling_width: Optional[float] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                     random_seed: Optional[int] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                     disable_time_filter: Optional[bool] = False,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                     verbose: bool = False) -> np.ndarray:
        """Produces a single trial of background+signal events based on inputs.

        Args:
            event_model: An object containing data and preprocessed parameters.
            injector:
            reduced_sim: Reweighted and pruned simulated events near the source
                declination.
            flux_norm: A flux normaliization to adjust weights.
            gamma: A spectral index to adjust weights.
            sampling_width: The bandwidth around the source declination to cut
                events.
            random_seed: A seed value for the numpy RNG.
            disable_time_filter: do not cut out events that is not in grl
            verbose: A flag to print progress.

        Returns:
            An array of combined signal and background events.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        if reduced_sim is None:
            reduced_sim = injector.reduced_sim(
                flux_norm=flux_norm,
                gamma=gamma,
                sampling_width=sampling_width,
            )

        background = injector.inject_background_events(event_model)

        if flux_norm > 0:
            signal = injector.inject_signal_events(reduced_sim)
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
        if not disable_time_filter:
            sorting_indices = np.argsort(events['time'])
            events = events[sorting_indices]

            # We need to check to ensure that every event is contained within
            # a good run. If the event happened when we had deadtime (when we
            # were not taking data), then we need to remove it.
            grl_start = event_model.grl['start'].reshape((1, len(event_model.grl)))
            grl_stop = event_model.grl['stop'].reshape((1, len(event_model.grl)))
            after_start = np.less(grl_start, events['time'].reshape(len(events), 1))
            before_stop = np.less(events['time'].reshape(len(events), 1), grl_stop)
            during_uptime = np.any(after_start & before_stop, axis=1)
            events = events[during_uptime]

        return events
