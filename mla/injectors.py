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
from mla import spectral


class PsInjector:
    """A basic point-source injector.

    This injector class assumes a source, a gaussian signal spatial pdf based
    on angular distance from the source, and declination-dependent background
    spatial pdf given by an event model.

    Attributes:
        source (Dict[str, float]): The dictionary representation of a source.
    """
    def __init__(self, source: Dict[str, float]) -> None:
        """Initializes the PsInjector object and gives it a source."""
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
        norm = 1 / (2 * np.pi * sigma**2)
        return norm * np.exp(-dist**2 / (2 * sigma**2))

    def background_spatial_pdf(self, events: np.ndarray,  # This still belongs here even though it doesn't use self... pylint: disable=no-self-use
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
        return (1 / (2 * np.pi)) * bg_densities

    def reduced_sim(self, event_model: models.EventModel, flux_norm: float = 0,
                    gamma: float = -2,
                    sampling_width: Optional[float] = None) -> np.ndarray:  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
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
            sindec_dist = np.abs(
                self.source['dec'] - event_model.sim['trueDec'])
            close = sindec_dist < sampling_width

            reduced_sim = rf.append_fields(
                event_model.sim[close].copy(),
                'weight',
                np.zeros(close.sum()),
                dtypes=np.float32)

            max_dec = np.min([np.sin(self.source['dec'] + sampling_width), 1])
            min_dec = np.max([np.sin(self.source['dec'] - sampling_width), -1])
            omega = 2 * np.pi * max_dec - min_dec

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
        rescaled_energy = (reduced_sim['trueE'] / 100.e3)**gamma
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

    def inject_background_events(self,  # This still belongs here even though it doesn't use self... pylint: disable=no-self-use
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
        background['ra'] = np.random.uniform(0, 2 * np.pi, len(background))

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
            p=reduced_sim['weight'] / total,
            replace=False).copy()

        # Update this number
        n_signal_observed = len(signal)

        if n_signal_observed > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = tools.rotate(
                signal['trueRa'], signal['trueDec'],
                ones * self.source['ra'], ones * self.source['dec'],
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = tools.rotate(
                signal['trueRa'], signal['trueDec'],
                ones * self.source['ra'], ones * self.source['dec'],
                signal['trueRa'], signal['trueDec'])

        return signal


class TimeDependentPsInjector(PsInjector):
    """Class info...

    More class info...

    Attributes:
        source (Dict[str, float]):
        event_model (EventModel):
    """
    def __init__(self, source: Dict[str, float],
                 background_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                 signal_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
    ) -> None:
        """Docstring"""
        super().__init__(source)
        self.background_time_profile = background_time_profile
        self.signal_time_profile = signal_time_profile

    def set_position(self, ra: float, dec: float) -> None:
        """set the position

        More function info...
        Args:
            ra
            dec
        """
        source = {'ra': ra, 'dec': dec}
        self.source = source

    def signal_spatial_pdf(self, events: np.ndarray) -> np.array:
        """Function info..

        Calculates the signal probability of events based on their angular
        distance from a source.

        Args:
            events: An array of events including their positional data.

        Returns:
            The value for the signal space pdf for the given events angular distances.
        """
        sigma = events['angErr']
        dist = tools.angular_distance(events['ra'], events['dec'],
                                      self.source['ra'], self.source['dec'])
        return (1 / (2 * np.pi * sigma**2)) * np.exp(-dist**2 / (2 * sigma**2))

    def background_spatial_pdf(self, events: np.ndarray,
                               event_model: models.EventModel) -> np.array:
        """Calculates the background probability of events based on their dec.

        More function info...

        Args:
            events: An array of events including their declination.

        Returns:
            The value for the background space pdf for the given events declinations.
        """
        return (1 / (2 * np.pi)) * event_model.background_dec_spline(
            np.sin(events['dec'])
        )

    def reduced_sim(self,
                    event_model: Optional[models.EventModel] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                    sim: Optional[np.ndarray] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                    flux_norm: Optional[float] = 0,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                    gamma: Optional[float] = -2,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                    livetime: float = None,
                    spectrum: Optional[spectral.BaseSpectrum] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                    sampling_width: Optional[float] = None) -> np.ndarray:  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
        """Function info...

        Prunes the simulation set to only events close to a given source and
        calculates the weight for each event. Adds the weights as a new column
        to the simulation set.

        Args:
            event_model: models.EventModel object that holds the simulation set
            sim: simulation set. Alternative for user not passing event_model.
            flux_norm: A flux normaliization to adjust weights.
            gamma: A spectral index to adjust weights.
            livetime: Livetime in days.It is equivalent of using flux per second
                instead of time-integrated flux.
            spectrum: spectral.BaseSpectrum object.Alternative for user not
                passing flux_norm and gamma.
            sampling_width: The bandwidth around the source declination to cut
                events.

        Returns:
            A reweighted simulation set around the source declination.
        """
        # Pick out only those events that are close in
        # declination. We only want to sample from those.
        if event_model is not None:
            sim = event_model.sim

        try:
            reduced_sim = rf.append_fields(sim, 'sindec', np.sin(sim['dec']),
                                           usemask=False)

            # The full simulation set,this is for the overall normalization of
            # the Energy S/B ratio

        except ValueError:  # sindec already exist
            reduced_sim = sim

        if sampling_width is not None:
            sindec_dist = np.abs(self.source['dec'] - sim['trueDec'])
            close = sindec_dist < sampling_width

            reduced_sim = rf.append_fields(
                reduced_sim[close].copy(),
                'weight',
                np.zeros(close.sum()),
                dtypes=np.float32,
                usemask=False)
            omega = 2 * np.pi * (
                np.min(
                    [np.sin(self.source['dec'] + sampling_width), 1]
                ) - np.max([np.sin(self.source['dec'] - sampling_width), -1])
            )
        else:
            reduced_sim = rf.append_fields(
                sim.copy(),
                'weight',
                np.zeros(len(sim)),
                dtypes=np.float32,
                usemask=False)
            omega = 4 * np.pi

        # Assign the weights using the newly defined "time profile"
        # classes above. If you want to make this a more complicated
        # shape, talk to me and we can work it out.
        if livetime is None:
            if spectrum is not None:
                reduced_sim['weight'] = reduced_sim['ow'] * spectrum(
                    reduced_sim['trueE'])
            else:
                reduced_sim['weight'] = reduced_sim['ow'] * flux_norm * (
                    reduced_sim['trueE'] / 100.e3)**gamma

        # Use flux per second instead of time-integrated flux
        else:
            if spectrum is not None:
                reduced_sim['weight'] = reduced_sim['ow'] * spectrum(
                    reduced_sim['trueE']) * livetime * 3600 * 24
            else:
                reduced_sim['weight'] = reduced_sim['ow'] * flux_norm * (
                    reduced_sim['trueE'] / 100.e3)**gamma * livetime * 3600 * 24

        # Apply the sampling width, which ensures that we
        # sample events from similar declinations.
        # When we do this, correct for the solid angle
        # we're including for sampling
        reduced_sim['weight'] /= omega
        reduced_sim['ow'] /= omega
        min_time = np.min(event_model.data['time'])
        max_time = np.max(event_model.data['time'])
        reduced_sim['time'] = np.random.uniform(min_time, max_time,
                                                size=len(reduced_sim))
        return reduced_sim

    def inject_background_events_No_time(self, event_model: models.EventModel) -> np.ndarray:
        """Function info...

        More function info...

        Returns:
            An array of injected background events.
        """
        # Get the number of events we see from these runs
        n_background = event_model.grl['events'].sum()
        n_background_observed = np.random.poisson(n_background)

        # How many events should we add in? This will now be based on the
        # total number of events actually observed during these runs
        background = np.random.choice(event_model.data, n_background_observed).copy()

        # Randomize the background RA
        background['ra'] = np.random.uniform(0, 2 * np.pi, len(background))

        return background

    def inject_background_events(self,
                                 event_model: models.EventModel,
                                 background_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                                 background_window: Optional[float] = 14,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                                 withinwindow: Optional[bool] = False,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                                 ) -> np.ndarray:
        """Function info...

        Inject Background Events

        Args:
            event_model: EventModel that holds the grl and data
            background_time_profile: Background time profile to do the injection
            background_window: Days used to estimated the background rate
            withinwindow: Use data rate within time profile to estimate background rate
        Returns:
            An array of injected background events.
        """
        if background_time_profile is not None:
            self.set_background_profile(event_model, background_time_profile,
                                        background_window, withinwindow)
        elif self.background_time_profile is not None:
            pass
        else:
            self.n_background = event_model.grl['events'].sum()

        # Get the number of events we see from these runs
        n_background = self.n_background
        n_background_observed = np.random.poisson(n_background)

        # How many events should we add in? This will now be based on the
        # total number of events actually observed during these runs
        background = np.random.choice(event_model.data,
                                      n_background_observed).copy()

        # Randomize the background RA
        background['ra'] = np.random.uniform(0, 2 * np.pi, len(background))

        # Randomize the background time
        if self.background_time_profile is not None:
            background['time'] = self.background_time_profile.random(
                size=len(background))
        return background

    def set_background_profile(self,
                               event_model: models.EventModel,
                               background_time_profile: time_profiles.GenericProfile,
                               background_window: Optional[float] = 14,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                               withinwindow: Optional[bool] = False) -> None:  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
        """Function info...

        Setting the background rate for the models
        Args:
            event_model: EventModel that holds the grl
            background_window: Days used to estimated the background rate
            background_time_profile: Background time profile to do the injection
            withinwindow: Use data rate within time profile to estimate background rate
        Returns:
            None
        """
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
                print("ERROR: No runs found in GRL for calculation of "
                      "background rates!")
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
                print("ERROR: No runs found in GRL for calculation of "
                      "background rates!")
                raise RuntimeError
            background_grl = event_model.grl[background_runs]

        # Find the background rate
        n_background = background_grl['events'].sum()
        n_background /= background_grl['livetime'].sum()
        n_background *= background_time_profile.exposure
        self.n_background = n_background
        self.background_time_profile = background_time_profile

    def set_signal_profile(self,
                           signal_time_profile: time_profiles.GenericProfile
    ) -> None:
        """Function info...

        Setting the signal time profile
        Args:
            signal_time_profile: Signal time profile to do the injection

        Returns:
            None
        """
        self.signal_time_profile = signal_time_profile

    def inject_signal_events(self,
                             reduced_sim: np.ndarray,
                             signal_time_profile: Optional[time_profiles.GenericProfile] = None  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
    ) -> np.ndarray:
        """Function info...

        More function info...

        Args:
            reduced_sim: Reweighted and pruned simulated events near the source declination.
            signal_time_profile: The time profile of the injected signal

        Returns:
            An array of injected signal events.
        """
        # Pick the signal events

        total = reduced_sim['weight'].sum()

        n_signal_observed = scipy.stats.poisson.rvs(total)
        signal = np.random.choice(reduced_sim, n_signal_observed,
                                  p=reduced_sim['weight'] / total,
                                  replace=False).copy()

        # Update this number
        n_signal_observed = len(signal)

        if n_signal_observed > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = tools.rotate(
                signal['trueRa'], signal['trueDec'],
                ones * self.source['ra'], ones * self.source['dec'],
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = tools.rotate(
                signal['trueRa'], signal['trueDec'],
                ones * self.source['ra'], ones * self.source['dec'],
                signal['trueRa'], signal['trueDec'])
        # set the signal time profile if provided
        if signal_time_profile is not None:
            self.set_signal_profile(signal_time_profile)

        # Randomize the signal time
        if self.signal_time_profile is not None:
            signal['time'] = self.signal_time_profile.random(size=len(signal))

        return signal

    def inject_background_and_signal(self,
                                     reduced_sim: np.ndarray,
                                     event_model: models.EventModel,
                                     nsignal: Optional[int] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                                     signal_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                                     background_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                                     background_window: Optional[float] = 14,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                                     ) -> np.ndarray:
        """Inject both signal and background

        Args:
            reduced_sim: Reweighted and pruned simulated events near the source declination.
            event_model: EventModel that holds the grl and data
            n: Number of events(Optional,use spectrum to get the nsignal if not pass.)
            signal_time_profile: The time profile of the injected signal
            background_time_profile: Background time profile to do the injection
            background_window: Days used to estimated the background rate

        Returns:
            Array containing injected signal and background
        """
        if nsignal is None:
            signal = self.inject_signal_events(reduced_sim, signal_time_profile)
        else:
            signal = self.inject_nsignal_events(
                reduced_sim, n=nsignal, signal_time_profile=signal_time_profile)
        background = self.inject_background_events(event_model,
                                                   background_time_profile,
                                                   background_window)
        signal = rf.drop_fields(signal, [n for n in signal.dtype.names
                                         if n not in background.dtype.names])

        return np.concatenate([background, signal])
