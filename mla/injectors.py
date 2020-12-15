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

import numpy as np
import scipy

from mla import mla
from mla import models


class PsInjector:
    """A basic point-source injector.

    This injector class assumes a source, a gaussian signal spatial pdf based
    on angular distance from the source, and declination-dependent background
    spatial pdf given by an event model.
    """
    def __init__(self) -> None:
        """Initializes the PsInjector object and gives it a source."""

    @staticmethod
    def signal_spatial_pdf(source: mla.Source, events: np.ndarray) -> np.array:
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
        dist = mla.angular_distance(events['ra'], events['dec'],
                                    source['ra'], source['dec'])
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
    def inject_signal_events(source: mla.Source,
                             trial_preprocessing: np.ndarray) -> np.ndarray:
        """Injects signal events for a trial.

        Args:
            source:
            reduced_sim: Reweighted and pruned simulated events near the source
                declination.

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

            signal['ra'], signal['dec'] = mla.rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source['ra'], ones * source['dec'],
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = mla.rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source['ra'], ones * source['dec'],
                signal['trueRa'], signal['trueDec'])

        return signal
