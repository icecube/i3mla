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

from typing import Optional, Tuple

import numpy as np
import scipy

from . import sources
from . import models
from . import time_profiles


def angular_distance(src_ra: float, src_dec: float, r_a: float,
                     dec: float) -> float:
    """Computes angular distance between source and location.

    Args:
        src_ra: The right ascension of the first point (radians).
        src_dec: The declination of the first point (radians).
        r_a: The right ascension of the second point (radians).
        dec: The declination of the second point (radians).

    Returns:
        The distance, in radians, between the two points.
    """
    sin_dec = np.sin(dec)

    cos_dec = np.sqrt(1. - sin_dec**2)

    cos_dist = (
        np.cos(src_ra - r_a) * np.cos(src_dec) * cos_dec
    ) + np.sin(src_dec) * sin_dec
    # handle possible floating precision errors
    cos_dist = np.clip(cos_dist, -1, 1)

    return np.arccos(cos_dist)


def rotate(ra1: float, dec1: float, ra2: float, dec2: float,
           ra3: float, dec3: float) -> Tuple[float, float]:
    """Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2).

    The rotation is performed on (ra3, dec3).

    Args:
        ra1: The right ascension of the point to be rotated from.
        dec1: The declination of the point to be rotated from.
        ra2: the right ascension of the point to be rotated onto.
        dec2: the declination of the point to be rotated onto.
        ra3: the right ascension of the point that will actually be rotated.
        dec3: the declination of the point that will actually be rotated.

    Returns:
        The rotated ra3 and dec3.

    Raises:
        IndexError: Arguments must all have the same dimension.
    """
    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    if not (
        len(ra1) == len(dec1) == len(ra2) == len(dec2) == len(ra3) == len(dec3)
    ):
        raise IndexError('Arguments must all have the same dimension.')

    cos_alpha = np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2) \
        + np.sin(dec1) * np.sin(dec2)

    # correct rounding errors
    cos_alpha[cos_alpha > 1] = 1
    cos_alpha[cos_alpha < -1] = -1

    alpha = np.arccos(cos_alpha)
    vec1 = np.vstack([np.cos(ra1) * np.cos(dec1),
                      np.sin(ra1) * np.cos(dec1),
                      np.sin(dec1)]).T
    vec2 = np.vstack([np.cos(ra2) * np.cos(dec2),
                      np.sin(ra2) * np.cos(dec2),
                      np.sin(dec2)]).T
    vec3 = np.vstack([np.cos(ra3) * np.cos(dec3),
                      np.sin(ra3) * np.cos(dec3),
                      np.sin(dec3)]).T
    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec**2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diagflat(np.ones(3))
    ntn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([
        np.roll(np.roll(np.diag(nv.ravel()), 1, 1), -1, 0) for nv in nvec])

    r = np.array([(1. - np.cos(a)) * ntn_i + np.cos(a) * one + np.sin(a) * nx_i
                  for a, ntn_i, nx_i in zip(alpha, ntn, nx)])
    vec = np.array([np.dot(r_i, vec_i.T) for r_i, vec_i in zip(r, vec3)])

    r_a = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    r_a += np.where(r_a < 0., 2. * np.pi, 0.)

    return r_a, dec


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
        dist = angular_distance(events['ra'], events['dec'], source.r_asc,
                                source.dec)
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

            signal['ra'], signal['dec'] = rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source.r_asc, ones * source.dec,
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = rotate(
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
                             **kwargs) -> np.ndarray:
        """Function info...

        More function info...

        Args:
            source:
            trial_preprocessing: Reweighted and pruned simulated events near
                the source declination.

        Returns:
            An array of injected signal events.
        """
        # Pick the signal events
        weighttotal = trial_preprocessing['weight'].sum()
        normed_weight = trial_preprocessing['weight'] / weighttotal
        total = weighttotal * self.signal_time_profile.exposure * 3600 * 24
        n_signal_observed = scipy.stats.poisson.rvs(total)

        signal = np.random.choice(trial_preprocessing, n_signal_observed,
                                  p=normed_weight, replace=False).copy()

        # Update this number
        n_signal_observed = len(signal)

        if n_signal_observed > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source.r_asc, ones * source.dec,
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source.r_asc, ones * source.dec,
                signal['trueRa'], signal['trueDec'])

        # Randomize the signal time
        signal['time'] = self.signal_time_profile.random(size=len(signal))

        return signal
