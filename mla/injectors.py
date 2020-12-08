"""
The classes in this file are used to inject events for trials generation.
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Dict, Optional

import numpy as np
import numpy.lib.recfunctions as rf
import scipy

from mla import tools
from mla import models
from mla import time_profiles

class PsInjector:
    """A basic point-source injector.

    This injector class assumes a source, a gaussian signal spatial pdf based
    on angular distance from the source, and declination-dependent background
    spatial pdf given by an event model.

    Attributes:
        source (Dict[str, float]): The dictionary representation of a source.
    """
    def __init__(self, source: Dict[str, float]) -> None:
        """Initializes the PsInjector object and gives it a source.

        Args:
            source: The dictionary representation of a source.
        """
        self.source = source

    def signal_spatial_pdf(self, events: np.ndarray) -> np.array:
        """Calculates the signal probability of events.

        Gives a gaussian probability based on their angular distance from the
        source object.

        Args:
            events: An array of events including their positional data.

        Returns:
            The value for the signal spatial pdf for the given events angular
            distances.
        """
        sigma = events['angErr']
        dist = tools.angular_distance(events['ra'], events['dec'],
                                      self.source['ra'], self.source['dec'])
        return (1.0/(2*np.pi*sigma**2))*np.exp(-dist**2/(2*sigma**2))

    def background_spatial_pdf(self, events: np.ndarray, # This still belongs here even though it doesn't use self... pylint: disable=no-self-use
                               event_model: models.EventModel) -> np.array:
        """Calculates the background probability of events based on their dec.

        Uses the background_dec_spline() function from the given event_model to
        get the probabilities.

        Args:
            events: An array of events including their declination.
            event_model: Preprocessed data and simulation.

        Returns:
            The value for the background space pdf for the given events decs.
        """
        bg_densities = event_model.background_dec_spline(np.sin(events['dec']))
        return (1/(2*np.pi))*bg_densities

    def reduced_sim(self, event_model: models.EventModel, flux_norm: float = 0,
                    gamma: float = -2,
                    sampling_width: Optional[float] = None) -> np.ndarray: # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
        """Gets a small simulation dataset to use for injecting signal.

        Prunes the simulation set to only events close to a given source and
        calculate the weight for each event. Adds the weights as a new column
        to the simulation set.

        Args:
            event_model: Preprocessed data and simulation.
            flux_norm: A flux normaliization to adjust weights.
            gamma: A spectral index to adjust weights.
            sampling_width: The bandwidth around the source dec to cut events.

        Returns:
            A reweighted simulation set around the source declination.
        """
        # Pick out only those events that are close in
        # declination. We only want to sample from those.
        if sampling_width is not None:
            sindec_dist = np.abs(self.source['dec']-event_model.sim['trueDec'])
            close = sindec_dist < sampling_width

            reduced_sim = rf.append_fields(
                event_model.sim[close].copy(),
                'weight',
                np.zeros(close.sum()),
                dtypes=np.float32)

            max_dec = np.min([np.sin(self.source['dec']+sampling_width), 1])
            min_dec = np.max([np.sin(self.source['dec']-sampling_width), -1])
            omega = 2*np.pi * max_dec - min_dec

        else:
            reduced_sim = rf.append_fields(
                event_model.sim.copy(),
                'weight',
                np.zeros(len(event_model.sim)),
                dtypes=np.float32)
            omega = 4*np.pi

        # Assign the weights using the newly defined "time profile"
        # classes above. If you want to make this a more complicated
        # shape, talk to me and we can work it out.
        rescaled_energy = (reduced_sim['trueE']/100.e3)**gamma
        reduced_sim['weight'] = reduced_sim['ow'] * flux_norm * rescaled_energy

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

    def inject_background_events(self, # This still belongs here even though it doesn't use self... pylint: disable=no-self-use
                                 event_model: models.EventModel) -> np.ndarray:
        """Injects background events for a trial.

        Args:
            event_model: Preprocessed data and simulation.

        Returns:
            An array of injected background events.
        """
        # Get the number of events we see from these runs
        n_background = event_model.grl['events'].sum()
        n_background_observed = np.random.poisson(n_background)

        # How many events should we add in? This will now be based on the
        # total number of events actually observed during these runs
        background = np.random.choice(event_model.data,
                                      n_background_observed).copy()

        # Randomize the background RA
        background['ra'] = np.random.uniform(0, 2*np.pi, len(background))

        return background

    def inject_signal_events(self, reduced_sim: np.ndarray) -> np.ndarray:
        """Injects signal events for a trial.

        Args:
            reduced_sim: Reweighted and pruned simulated events near the source
                declination.

        Returns:
            An array of injected signal events.
        """
        # Pick the signal events
        total = reduced_sim['weight'].sum()

        n_signal_observed = scipy.stats.poisson.rvs(total)
        signal = np.random.choice(
            reduced_sim,
            n_signal_observed,
            p=reduced_sim['weight']/total,
            replace = False).copy()

        # Update this number
        n_signal_observed = len(signal)

        if n_signal_observed > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = tools.rotate(
                signal['trueRa'], signal['trueDec'],
                ones*self.source['ra'], ones*self.source['dec'],
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = tools.rotate(
                signal['trueRa'], signal['trueDec'],
                ones*self.source['ra'], ones*self.source['dec'],
                signal['trueRa'], signal['trueDec'])

        return signal

class TimeDependentPsInjector(PsInjector):
    """Docstring"""

    def __init__(self, source: Dict[str, float],
                 bg_time_profile: time_profiles.GenericProfile,
                 sig_time_profile: time_profiles.GenericProfile) -> None:
        """Docstring"""
        super().__init__(source)
        self.bg_time_profile = bg_time_profile
        self.sig_time_profile = sig_time_profile

    def inject_background_events(self,
                                 event_model: models.EventModel) -> np.ndarray:
        """Docstring"""
        raise NotImplementedError

    def inject_signal_events(self, reduced_sim: np.ndarray) -> np.ndarray:
        """Docstring"""
        raise NotImplementedError
