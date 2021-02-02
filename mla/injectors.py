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

from typing import Optional

import numpy as np
import scipy

from . import core
from . import sources
from . import models
from . import time_profiles


class PsInjector:
    """A basic point-source injector.

    This injector class assumes a source, a gaussian signal spatial pdf based
    on angular distance from the source, and declination-dependent background
    spatial pdf given by an event model.
    """
    def __init__(self) -> None:
        """Initializes the PsInjector object and gives it a source."""

    @staticmethod
    def signal_spatial_pdf(source: sources.Source,
                           events: np.ndarray) -> np.array:
        """Calculates the signal probability of events.

        Gives a gaussian probability based on their angular distance from the
        source object.

        Args:
            source:
            events: An array of events including their positional data.

        Returns:
            The value for the signal spatial pdf for the given events angular
            distances.
        """
        sigma = events['angErr']
        dist = core.angular_distance(events['ra'], events['dec'],
                                     source.r_asc, source.dec)
        norm = 1 / (2 * np.pi * sigma**2)
        return norm * np.exp(-dist**2 / (2 * sigma**2))

    @staticmethod
    def background_spatial_pdf(event_model: models.EventModel,
                               events: np.array) -> np.array:
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
        return (1 / (2 * np.pi)) * bg_densities

    @staticmethod
    def inject_background_events(event_model: models.EventModel) -> np.ndarray:
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
        background['ra'] = np.random.uniform(0, 2 * np.pi, len(background))

        return background

    @staticmethod
    def inject_signal_events(source: sources.Source,
                             trial_preprocessing: np.ndarray) -> np.ndarray:
        """Injects signal events for a trial.

        Args:
            source:
            trial_preprocessing: Reweighted and pruned simulated events near
            the source declination.

        Returns:
            An array of injected signal events.
        """
        # Pick the signal events
        total = trial_preprocessing['weight'].sum()

        n_signal_observed = scipy.stats.poisson.rvs(total)
        signal = np.random.choice(
            trial_preprocessing,
            n_signal_observed,
            p=trial_preprocessing['weight'] / total,
            replace=False).copy()

        # Update this number
        n_signal_observed = len(signal)

        if n_signal_observed > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = core.rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source.r_asc, ones * source.dec,
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = core.rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source.r_asc, ones * source.dec,
                signal['trueRa'], signal['trueDec'])

        return signal


class TimeDependentPsInjector(PsInjector):
    """Class info...

    More class info...

    Attributes:
        source (Dict[str, float]):
        event_model (EventModel):
    """
    def __init__(self, event_model: models.EventModel,
                 signal_time_profile: time_profiles.GenericProfile,
                 background_time_profile: time_profiles.GenericProfile,
                 background_window: Optional[float] = 14,
                 withinwindow: Optional[bool] = False) -> None:
        """Initialize TimeDependentPsInjector...

        Initialize and set the background and signal time profile
        Args:
            event_model: EventModel that holds the grl
            signal_time_profile: Signal time profile
            background_time_profile: Background time profile
            background_window: Days used to estimated the background rate
            withinwindow: Use data rate within time profile to estimate
                background rate
        Returns:
            None
        """
        super().__init__()
        # Find the run contian in the background time window
        start_time = background_time_profile.range[0]
        end_time = background_time_profile.range[1]
        if withinwindow:
            fully_contained = (
                event_model.grl['start'] >= start_time
            ) & (event_model.grl['stop'] < end_time)
            start_contained = (
                event_model.grl['start'] < start_time
            ) & (event_model.grl['stop'] > start_time)
            background_runs = (fully_contained | start_contained)
            if not np.any(background_runs):
                print('ERROR: No runs found in GRL for calculation of '
                      'background rates!')
                raise RuntimeError
            background_grl = event_model.grl[background_runs]
        else:
            fully_contained = (
                event_model.grl['start'] >= start_time - background_window
            ) & (event_model.grl['stop'] < start_time)
            start_contained = (
                event_model.grl['start'] < start_time - background_window
            ) & (event_model.grl['stop'] > start_time - background_window)
            background_runs = (fully_contained | start_contained)
            if not np.any(background_runs):
                print('ERROR: No runs found in GRL for calculation of '
                      'background rates!')
                raise RuntimeError
            background_grl = event_model.grl[background_runs]

        # Find the background rate
        n_background = background_grl['events'].sum()
        n_background /= background_grl['livetime'].sum()
        n_background *= background_time_profile.exposure
        self.n_background = n_background
        self.background_time_profile = background_time_profile
        self.signal_time_profile = signal_time_profile

    def inject_background_events(self,
                                 event_model: models.EventModel) -> np.ndarray:
        """Injects background events with specific time profile for a trial.

        Args:

        Returns:
            An array of injected background events.
        """
        n_background_observed = np.random.poisson(self.n_background)
        background = np.random.choice(event_model.data,
                                      n_background_observed).copy()
        background['ra'] = np.random.uniform(0, 2 * np.pi, len(background))
        background['time'] = self.background_time_profile.random(
            size=len(background))

        return background

    def inject_signal_events(self, source: sources.Source,  # Necessary for compatibility with trial_generator... pylint: disable=unused-argument
                             trial_preprocessing: np.ndarray,
                             n_signal_observed: Optional[int] = None,
                             **kwargs) -> np.ndarray:
        """Function info...

        More function info...

        Args:
            source:
            trial_preprocessing: Reweighted and pruned simulated events near
                the source declination.
            n_signal_observed: Optional,overriding the trial_preprocessing
                weighted total events

        Returns:
            An array of injected signal events.
        """
        # Pick the signal events
        weighttotal = trial_preprocessing['weight'].sum()
        normed_weight = trial_preprocessing['weight'] / weighttotal
        if n_signal_observed is None:
            total = weighttotal * self.signal_time_profile.exposure * 3600 * 24
            n_signal_observed = scipy.stats.poisson.rvs(total)

        signal = np.random.choice(trial_preprocessing, n_signal_observed,
                                  p=normed_weight, replace=False).copy()

        # Update this number
        n_signal_observed = len(signal)

        if n_signal_observed > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = core.rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source.r_asc, ones * source.dec,
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = core.rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source.r_asc, ones * source.dec,
                signal['trueRa'], signal['trueDec'])

        # Randomize the signal time
        signal['time'] = self.signal_time_profile.random(size=len(signal))

        return signal
