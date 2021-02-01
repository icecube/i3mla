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

from . import core
from . import models
from . import injectors
from . import spectral


class PsTrialGenerator:
    """Docstring"""

    def __init__(self) -> None:
        """Docstring"""

    @staticmethod
    def preprocess_trial(event_model: models.EventModel, source: core.Source,  # This is fine... pylint: disable=too-many-locals, too-many-arguments, unused-argument
                         flux_norm: float = 0, gamma: float = -2,
                         spectrum: Optional[spectral.BaseSpectrum] = None,
                         sampling_width: Optional[float] = None,
                         **kwargs) -> np.ndarray:
        """Gets a small simulation dataset to use for injecting signal.

        Prunes the simulation set to only events close to a given source and
        calculate the weight for each event. Adds the weights as a new column
        to the simulation set.

        Args:
            event_model: Preprocessed data and simulation.
            source:
            flux_norm: A flux normaliization to adjust weights.
            gamma: A spectral index to adjust weights.
            sampling_width: The bandwidth around the source dec to cut events.

        Returns:
            A reweighted simulation set around the source declination.
        """
        # Pick out only those events that are close in
        # declination. We only want to sample from those.
        if sampling_width is not None:
            sindec_dist = np.abs(
                source['dec'] - event_model.sim['trueDec'])
            close = sindec_dist < sampling_width

            reduced_sim = rf.append_fields(
                event_model.sim[close].copy(),
                'weight',
                np.zeros(close.sum()),
                dtypes=np.float32)

            max_dec = np.min([np.sin(source['dec'] + sampling_width), 1])
            min_dec = np.max([np.sin(source['dec'] - sampling_width), -1])
            omega = 2 * np.pi * (max_dec - min_dec)

        else:
            reduced_sim = rf.append_fields(
                event_model.sim.copy(),
                'weight',
                np.zeros(len(event_model.sim)),
                dtypes=np.float32)
            omega = 4 * np.pi

        # Assign the weights using the newly defined "time profile"
        # classes above. If you want to make this a more complicated
        # shape, talk to me and we can work it out.
        if spectrum is None:
            rescaled_energy = (reduced_sim['trueE'] / 100.e3)**gamma
            reduced_sim['weight'] = reduced_sim['ow'] * flux_norm
            reduced_sim['weight'] *= rescaled_energy
        else:
            reduced_sim['weight'] = reduced_sim['ow'] * flux_norm
            reduced_sim['weight'] *= spectrum(reduced_sim['trueE'])

        # Apply the sampling width, which ensures that we
        # sample events from similar declinations.
        # When we do this, correct for the solid angle
        # we're including for sampling
        reduced_sim['weight'] /= omega

        # Randomly assign times to the simulation events within the data time
        # range.
        min_time = np.min(event_model.data['time'])
        max_time = np.max(event_model.data['time'])
        reduced_sim['time'] = np.random.uniform(min_time, max_time,
                                                size=len(reduced_sim))

        return reduced_sim

    @classmethod
    def generate(cls, event_model: models.EventModel,  # pylint: disable=too-many-locals, too-many-arguments
                 injector: injectors.PsInjector, source: core.Source,
                 preprocessing: np.ndarray, flux_norm: float = 0,
                 random_seed: Optional[int] = None,
                 disable_time_filter: Optional[bool] = False,
                 verbose: bool = False,
                 **kwargs) -> np.ndarray:
        """Produces a single trial of background+signal events based on inputs.

        Args:
            event_model: An object containing data and preprocessed parameters.
            injector:
            source:
            preprocessing: Reweighted and pruned simulated events near the
                source declination.
            flux_norm: A flux normaliization to adjust weights.
            random_seed: A seed value for the numpy RNG.
            disable_time_filter: do not cut out events that is not in grl
            verbose: A flag to print progress.

        Returns:
            An array of combined signal and background events.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        background = injector.inject_background_events(event_model)

        if flux_norm > 0 or 'n_signal_observed' in kwargs:
            signal = injector.inject_signal_events(source, preprocessing,
                                                   **kwargs)
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
            grl_start = event_model.grl['start'].reshape((1,
                                                          len(event_model.grl)))
            grl_stop = event_model.grl['stop'].reshape((1,
                                                        len(event_model.grl)))
            after_start = np.less(grl_start, events['time'].reshape(len(events),
                                                                    1))
            before_stop = np.less(events['time'].reshape(len(events), 1),
                                  grl_stop)
            during_uptime = np.any(after_start & before_stop, axis=1)
            events = events[during_uptime]

        return events
