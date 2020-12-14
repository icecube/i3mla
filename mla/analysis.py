"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Dict, List, Optional, Tuple, Union

import abc
import scipy
import numpy as np
import numpy.lib.recfunctions as rf

from mla import models
from mla import injectors
from mla import time_profiles
from mla import spectral

TsPreprocess = Tuple[List[scipy.interpolate.UnivariateSpline], np.array]


class Analysis:
    """Abstract class defining the structure of analysis classes.

    Analysis classes based on this one have functions to produce trials one at
    a time for troubleshooting, and a function produce_and_minimize to run
    multiple trials.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initializes the analysis object."""

    @abc.abstractmethod
    def evaluate_ts(self, events: np.ndarray, event_model: models.EventModel,
                    *args, **kwargs) -> np.array:
        """Evaluates the test statistic for some events given an event model."""

    @abc.abstractmethod
    def minimize_ts(self, events: np.ndarray, event_model: models.EventModel,
                    *args, **kwargs) -> Dict[str, float]:
        """Finds the params that minimize the test statistic for some events."""

    @abc.abstractmethod
    def produce_trial(self, event_model: models.EventModel, *args,
                      **kwargs) -> np.ndarray:
        """Generates some events based on the given event model."""

    @abc.abstractmethod
    def produce_and_minimize(self, event_model: models.EventModel,
                             n_trials: int, *args, **kwargs) -> np.ndarray:
        """Generates events and finds the best fit params n_trials times."""


class PsAnalysis(Analysis):
    """A basic point-source analysis class.

    The is a time integrated analysis class that is useful for fitting the
    number of signal events and spectral index of a given a source and a time-
    integrated neutrino flux at the source.

    Attributes:
        injector (injectors.PsInjector): An injector object used to produce
            trial events from a given event model.
    """
    def __init__(self,
                 source: Optional[Dict[str, float]] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                 injector: Optional[injectors.PsInjector] = None) -> None:  # Python 3.9 bug... pylint: disable=unsubscriptable-object
        """Initializes the analysis object.

        Args:
            source: A dictionary containing the source location for this
                analysis.
        """
        super().__init__()
        if injector is not None:
            self.injector = injector
        else:
            self.injector = injectors.PsInjector(source)

    def _preprocess_ts(self, events: np.ndarray, event_model: models.EventModel
    ) -> TsPreprocess:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            events: An array of events to calculate the test-statistic for.
            event_model: An object containing data and preprocessed parameters.

        Returns:
            A list of log(sob) vs gamma splines for each event and the pre-
            processed signal over background.

        Raises:
            ValueError: There must be at least one event.
        """
        if len(events) == 0:
            raise ValueError('len(events) must be > 0.')

        sig = self.injector.signal_spatial_pdf(events)
        bkgr = self.injector.background_spatial_pdf(events, event_model)
        splines = event_model.get_log_sob_gamma_splines(events)

        return splines, sig / bkgr

    def evaluate_ts(self, events: np.ndarray, event_model: models.EventModel,  # Super class will never be called... pylint: disable=arguments-differ, too-many-arguments
                    n_signal: float, gamma: float,
                    n_events: Optional[int] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                    preprocessing: TsPreprocess = None) -> np.array:
        """Evaluates the test-statistic for the given events and parameters.

        Calculates the test-statistic using a given event model, n_signal, and
        gamma. This function does not attempt to fit n_signal or gamma.

        Args:
            events: An array of events to calculate the test statistic for.
            event_model: An object containing data and preprocessed parameters.
            n_signal: A guess for the number of signal events.
            n_events: The total number of events in the trial.
            gamma: A guess for the spectral index.

        Returns:
            The overall test-statistic value for the given events and
            parameters.
        """
        if n_events is None:
            n_events = len(events)
        if preprocessing is None:
            preprocessing = self._preprocess_ts(events, event_model)

        splines, sob = preprocessing
        sob_new = sob * np.exp([spline(gamma) for spline in splines])

        return np.log((n_signal / n_events * (sob_new - 1)) + 1)

    def minimize_ts(self, events: np.ndarray, event_model: models.EventModel,  # Super class will never be called... pylint: disable=arguments-differ, too-many-arguments, too-many-locals
                    test_ns: float, test_gamma: float,
                    gamma_bounds: Tuple[float] = (-4, -1),
                    **kwargs) -> Dict[str, float]:
        """Calculates the params that minimize the ts for the given events.

        Accepts guess values for fitting the n_signal and spectral index, and
        bounds on the spectral index. Uses scipy.optimize.minimize() to fit.
        The default method is 'L-BFGS-B', but can be overwritten by passing
        kwargs to this fuction.

        Args:
            events: A sample array of events to find the best fit for.
            event_model: An object containing data and preprocessed parameters.
            test_ns: An initial guess for the number of signal events
                (n_signal).
            test_gamma: An initial guess for the spectral index (gamma).

        Returns:
            A dictionary containing the minimized overall test-statistic, the
            best-fit n_signal, and the best fit gamma.
        """
        output = {'ts': np.nan, 'n_signal': test_ns, 'gamma': test_gamma}
        if len(events) == 0:
            return output
        preprocessing = self._preprocess_ts(events, event_model)

        # Check: n_signal cannot be larger than n_events
        n_events = len(events)
        if n_events <= test_ns:
            test_ns = n_events - 0.00001

        # Drop events with zero spatial or time llh
        # The contribution of those llh will be accounts in drop*np.log(1-n_signal/n_events)
        drop = n_events - np.sum(preprocessing[1] != 0)
        drop_index = preprocessing[1] != 0
        preprocessing = (preprocessing[0][drop_index],
                         preprocessing[1][drop_index])

        def get_ts(args):
            n_signal = args[0]
            gamma = args[1]
            llhs = self.evaluate_ts(events[drop_index], event_model, n_signal, 
                                    gamma, n_events=n_events,
                                    preprocessing=preprocessing)
            return -2 * (np.sum(llhs) + drop * np.log(1 - n_signal / n_events))

        with np.errstate(divide='ignore', invalid='ignore'):
            # Set the seed values, which tell the minimizer
            # where to start, and the bounds. First do the
            # shape parameters.
            x_0 = [test_ns, test_gamma]
            bounds = [(0, n_events), gamma_bounds]  # gamma [min, max]
            if 'method' not in kwargs:
                kwargs['method'] = 'L-BFGS-B'

            result = scipy.optimize.minimize(get_ts, x0=x_0, bounds=bounds,
                                             **kwargs)

            # Store the results in the output array
            output['ts'] = -1 * result.fun
            output['n_signal'] = result.x[0]
            output['gamma'] = result.x[1]

        return output

    def produce_trial(self, event_model: models.EventModel, flux_norm: float,  # Super class will never be called... pylint: disable=arguments-differ, too-many-locals, too-many-arguments
                      reduced_sim: Optional[np.ndarray] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                      gamma: float = -2, sampling_width: Optional[float] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                      n_signal: Optional[int] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                      random_seed: Optional[int] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                      disable_time_filter: Optional[bool] = False,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                      verbose: bool = False) -> np.ndarray:
        """Produces a single trial of background+signal events based on inputs.

        Args:
            event_model: An object containing data and preprocessed parameters.
            reduced_sim: Reweighted and pruned simulated events near the source
                declination.
            flux_norm: A flux normaliization to adjust weights.
            gamma: A spectral index to adjust weights.
            n_signal: How many signal events(Will overwrite flux_norm)
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
            reduced_sim = self.injector.reduced_sim(
                flux_norm=flux_norm,
                gamma=gamma,
                sampling_width=sampling_width)

        background = self.injector.inject_background_events(event_model)
        if n_signal is None:
            if flux_norm > 0:
                signal = self.injector.inject_signal_events(reduced_sim)
            else:
                signal = np.empty(0, dtype=background.dtype)
        else:
            signal = self.injector.inject_nsignal_events(reduced_sim, n_signal)

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

    def produce_and_minimize(self, event_model: models.EventModel,  # Super class will never be called... pylint: disable=arguments-differ, too-many-locals, too-many-arguments
                             n_trials: int, test_ns: float = 1,
                             test_gamma: float = -2,
                             random_seed: Optional[int] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                             flux_norm: float = 0, gamma: float = -2,
                             n_signal: Optional[int] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                             sampling_width: Optional[float] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                             disable_time_filter: Optional[bool] = False,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                             verbose: bool = False) -> np.ndarray:
        """Produces n trials and calculate a test statistic for each trial.

        Args:
            event_model: An object containing data and preprocessed parameters.
            n_trials: The number of times to repeat the trial + evaluate_ts
                process.
            test_ns: A guess for the number of signal events.
            test_gamma: A guess for best fit spectral index of the signal.
            random_seed: A seed value for the numpy RNG.
            flux_norm: A flux normaliization to adjust weights.
            gamma: A guess for best fit spectral index of the signal.
            n_signal: How many signal events(Will overwrite flux_norm)
            sampling_width: The bandwidth around the source declination to cut
                events.
            disable_time_filter:do not cut out events that is not in grl
            verbose: A flag to print progress.

        Returns:
            An array of test statistic values and their best-fit parameters for
            each trial.
        """
        if random_seed:
            np.random.seed(random_seed)

        # Cut down the sim. We're going to be using the same
        # source and weights each time, so this stops us from
        # needing to recalculate over and over again.
        reduced_sim = self.injector.reduced_sim(event_model,
                                                flux_norm=flux_norm,
                                                gamma=gamma,
                                                sampling_width=sampling_width)

        # Build a place to store information for the trial
        dtype = np.dtype([
            ('ts', np.float32),
            ('ntot', np.int),
            ('ninj', np.int),
            ('n_signal', np.float32),
            ('gamma', np.float32),
        ])
        fit_info = np.empty(n_trials, dtype=dtype)

        if verbose:
            print(f'Running Trials (N={flux_norm:3.2e}, gamma={gamma:2.1f})',
                  end='')
            prop_complete = 0

        for i in range(n_trials):
            # Produce the trial events
            trial = self.produce_trial(event_model, flux_norm, reduced_sim,
                                       n_signal=n_signal,
                                       random_seed=random_seed,
                                       disable_time_filter=disable_time_filter)

            # And get the weights
            bestfit = self.minimize_ts(trial, event_model, test_ns, test_gamma)

            fit_info['ts'][i] = bestfit['ts']
            fit_info['ntot'][i] = len(trial)
            fit_info['ninj'][i] = (trial['run'] > 200000).sum()
            fit_info['n_signal'][i] = bestfit['n_signal']
            fit_info['gamma'][i] = bestfit['gamma']

            if verbose and i / n_trials > prop_complete:
                prop_complete += .05
                print('.', end='')
        if verbose:
            print('done')

        return fit_info


class ThreeMLPsAnalysis(Analysis):
    """Class info...

    More class info...

    Attributes:
        injector (injectors.PsInjector):
    """
    def __init__(self,
                 source: Optional[Dict[str, float]] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                 injector: Optional[injectors.PsInjector] = None) -> None:  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
        """Function info...
        More function info...
        Args:
            source:
            injector:
        """
        super().__init__()
        if injector is not None:
            self.injector = injector
        else:
            self.injector = injectors.TimeDependentPsInjector(source)
        self.first_spatial = False

    def _init_event_model(self,
                          data: np.ndarray,
                          sim: np.ndarray,
                          grl: np.ndarray,
                          background_sin_dec_bins: Union[np.array, int] = 500,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                          signal_sin_dec_bins: Union[np.array, int] = 50,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                          log_energy_bins: Union[np.array, int] = 50,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                          spectrum: Optional[spectral.BaseSpectrum] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                          sampling_width: Optional[float] = np.radians(3),  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                          reduce: Optional[bool] = True,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                          verbose: bool = False) -> models.EventModel:
        """Set the event model
        More function info...
        Args:
            data: data
            sim: simulation set
            grl: grl
            background_sin_dec_bins: If an int, then the number of bins spanning
                -1 -> 1, otherwise, a numpy array of bin edges.
            signal_sin_dec_bins: If an int, then the number of bins spanning
                -1 -> 1, otherwise, a numpy array of bin edges.
            log_energy_bins: if an int, then the number of bins spanning 1 -> 8,
                otherwise, a numpy array of bin edges.
            spectrum: Spectrum for energy weighting
            reduce: whether reduce the simulation to narrow dec
            verbose: A flag to print progress.
        """
        event_model = models.ThreeMLEventModel(data,
                                               sim,
                                               grl,
                                               background_sin_dec_bins,
                                               signal_sin_dec_bins,
                                               log_energy_bins,
                                               spectrum,
                                               sampling_width,
                                               reduce,
                                               verbose)
        return event_model

    def set_data(self, data: np.ndarray) -> None:
        """Set data
        More function info...

        Args:
            data: data events
        """
        self.data = data
        self.n_events = len(data)
        self.first_spatial = True

    def calculate_TS(self,
                     event_model: models.EventModel,
                     data: Optional[np.ndarray] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                     n_signal: Optional[float] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                     test_signal_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                     test_background_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                     recalculate: Optional[bool] = True) -> float:  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
        """Calculate the signal pdf"""
        if self.n_events == 0:
            return 0, 0
        if data is not None:
            self.set_data(data)
        if recalculate:
            if data is not None:
                self.data = data
            if len(self.data) == 0:
                return 0, 0
            if self.first_spatial:
                self.spatial_s = self.injector.signal_spatial_pdf(self.data)
                mask = self.spatial_s != 0
                self.data = self.data[mask]
                self.spatial_s = self.spatial_s[mask]
                self.spatial_b = self.injector.background_spatial_pdf(
                    self.data, event_model)
                self.spatial_sob = self.spatial_s / self.spatial_b
                self.drop = self.n_events - mask.sum()
                self.first_spatial = False
                if test_signal_time_profile is None or test_background_time_profile is None:
                    if self.injector.signal_time_profile is None:
                        time_s = 1
                        time_b = 1
                    else:
                        time_s = self.injector.signal_time_profile.pdf(
                            self.data['time'])
                        time_b = self.injector.background_time_profile.pdf(
                            self.data['time'])
                else:
                    time_s = test_signal_time_profile.pdf(self.data['time'])
                    time_b = test_background_time_profile.pdf(self.data['time'])

                self.time_sob = time_s / time_b
                self.time_sob[np.isnan(self.time_sob)] = 0
            self.energy_sob = event_model.get_energy_sob(self.data)
        if n_signal is None:
            n_signal = event_model._spectrum(
                event_model._reduced_sim_truedec['trueE']
            ) * (
                event_model._reduced_sim_truedec['ow']
            ) * (
                self.injector.signal_time_profile.exposure
            ) * 3600 * 24
            n_signal = n_signal.sum()

        self.ts = (n_signal / self.n_events * (
            self.energy_sob * self.spatial_sob * self.time_sob - 1
        )) + 1
        ts_value = 2 * (
            np.sum(np.log(self.ts)) + self.drop * np.log(
                1 - n_signal / self.n_events)
        )
        if np.isnan(ts_value):
            self.ts = 0
        return ts_value, n_signal

    def produce_trial(self,
                      event_model: models.EventModel,
                      spectrum: Optional[spectral.BaseSpectrum] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                      n_signal: Optional[int] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                      random_seed: Optional[int] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                      signal_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                      background_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                      background_window: Optional[float] = 14,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                      withinwindow: Optional[bool] = False) -> np.ndarray:  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
        """Produce trial

        More function info...

        Args:
            event_model: An object containing data and preprocessed parameters.
            spectrum: Spectrum of the injection
            n_signal: How many signal events(Will overwrite flux_norm)
            random_seed: A seed value for the numpy RNG.
            signal_time_profile: The time profile of the injected signal.
            background_time_profile: Background time profile to do the
                injection.
            disable_time_filter:Cut out events that is not in grl

        Returns:
            An array of combined signal and background events.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        background = self.injector.inject_background_events(
            event_model, background_time_profile=background_time_profile,
            background_window=background_window, withinwindow=withinwindow
        )

        if signal_time_profile is not None:
            self.injector.set_signal_profile(signal_time_profile)

        livetime = self.injector.signal_time_profile.exposure

        if n_signal is None:
            if spectrum is None:
                try:
                    data = self.injector.inject_signal_events(
                        event_model._reduced_sim_truedec,
                        signal_time_profile=signal_time_profile
                    )
                except:
                    raise "No spectrum had even supplied"
            else:
                event_model._reduced_sim_truedec = self.injector.reduced_sim(
                    event_model=event_model, spectrum=spectrum,
                    livetime=livetime
                )
                data = self.injector.inject_signal_events(
                    event_model._reduced_sim_truedec,
                    signal_time_profile=signal_time_profile
                )
        else:
            if spectrum is None:
                try:
                    data = self.injector.inject_nsignal_events(
                        event_model._reduced_sim_truedec, n_signal,
                        signal_time_profile=signal_time_profile
                    )
                except:
                    raise "No spectrum had even supplied"
            else:
                event_model._reduced_sim_truedec = self.injector.reduced_sim(
                    event_model=event_model, spectrum=spectrum,
                    livetime=livetime,
                )
                data = self.injector.inject_nsignal_events(
                    event_model._reduced_sim_truedec, n_signal,
                    signal_time_profile=signal_time_profile,
                )

        bgrange = self.injector.background_time_profile.range
        contained_in_background = (
            (data['time'] >= bgrange[0]) & (data['time'] < bgrange[1])
        )
        data = data[contained_in_background]
        data = rf.drop_fields(data, [n for n in data.dtype.names
                                     if n not in background.dtype.names])

        return np.concatenate([background, data])

    def min_ns(self,
               event_model: models.EventModel,
               data: Optional[np.ndarray] = None,  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
               recalculate: Optional[bool] = True) -> float:  # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
        """minimize n_signal"""

        if data is not None:
            self.set_data(data)

        if self.n_events == 0:
            return 0, 0
        bounds = [[0, self.n_events], ]

        if recalculate:
            if self.first_spatial:
                self.spatial_s = self.injector.signal_spatial_pdf(self.data)
                mask = self.spatial_s != 0
                self.data = self.data[mask]
                self.spatial_s = self.spatial_s[mask]
                self.spatial_b = self.injector.background_spatial_pdf(
                    self.data, event_model)
                self.spatial_sob = self.spatial_s / self.spatial_b
                self.drop = self.n_events - mask.sum()
                self.first_spatial = False
                if self.injector.signal_time_profile is None:
                    time_s = 1
                    time_b = 1
                else:
                    time_s = self.injector.signal_time_profile.pdf(
                        self.data['time'])
                    time_b = self.injector.background_time_profile.pdf(
                        self.data['time'])
                self.time_sob = time_s / time_b
                self.time_sob[np.isnan(self.time_sob)] = 0
                if np.isinf(self.time_sob).sum() > 0:
                    print("Warning:Background time profile doesn't not cover si"
                          "gnal time profile.Discard events outside background "
                          "window!")
                    self.time_sob[np.isinf(self.time_sob)] = 0
            self.energy_sob = event_model.get_energy_sob(self.data)

        def cal_ts(n_signal):
            t_stat = (n_signal / self.n_events * (
                self.energy_sob * self.spatial_sob * self.time_sob - 1
            )) + 1
            ts_value = -2 * (
                np.sum(np.log(t_stat)) + self.drop * np.log(
                    1 - n_signal / self.n_events)
            )
            return ts_value

        result = scipy.optimize.minimize(cal_ts, x0=[1, ], bounds=bounds,
                                         method='SLSQP')
        fit_ns = result.x[0]
        fit_ts = -1 * result.fun
        if np.isnan(fit_ts):
            fit_ts = 0
            fit_ns = 0
        return fit_ts, fit_ns


class TimeDependentPsAnalysis(PsAnalysis):
    """Docstring"""

    def __init__(self,
                 source: Optional[Dict[str, float]],  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                 event_model: models.EventModel,
                 background_time_profile: Optional[Union[time_profiles.GenericProfile, Tuple[float, float]]],  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                 signal_time_profile: Optional[Union[time_profiles.GenericProfile, Tuple[float, float]]],  # Python 3.9 bug... pylint: disable=unsubscriptable-object 
                 test_background_time_profile: Optional[Union[time_profiles.GenericProfile, Tuple[float, float]]],  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                 test_signal_time_profile: Optional[Union[time_profiles.GenericProfile, Tuple[float, float]]],  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                 **kwargs
    ) -> None:
        """Function info...
        More function info...
        Args:
            source:
            background_time_profile: Background time profile for injection
            signal_time_profile:Signal time profile for injection
            test_background_time_profile:Background time profile for likelihood
            test_signal_time_profile:Background time profile for injection
            injector:
        """
        super().__init__()
        self.injector = injectors.TimeDependentPsInjector(source)

        if not issubclass(type(background_time_profile),
                          time_profiles.GenericProfile):
            background_time_profile = time_profiles.UniformProfile(
                background_time_profile[0], background_time_profile[1])

        if not issubclass(type(signal_time_profile),
                          time_profiles.GenericProfile):
            signal_time_profile = time_profiles.UniformProfile(
                signal_time_profile[0], signal_time_profile[1])

        if not issubclass(type(test_background_time_profile),
                          time_profiles.GenericProfile):
            test_background_time_profile = time_profiles.UniformProfile(
                test_background_time_profile[0],
                test_background_time_profile[1],
            )

        if not issubclass(type(test_signal_time_profile),
                          time_profiles.GenericProfile):
            test_signal_time_profile = time_profiles.UniformProfile(
                test_signal_time_profile[0], test_signal_time_profile[1])

        self.injector.set_signal_profile(signal_time_profile)
        self.injector.set_background_profile(event_model,
                                             background_time_profile, **kwargs)
        self.test_background_time_profile = test_background_time_profile
        self.test_signal_time_profile = test_signal_time_profile
        return

    def _preprocess_ts(self, events: np.ndarray, event_model: models.EventModel,
                       test_signal_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                       test_background_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
    ) -> Optional[Tuple[List[scipy.interpolate.UnivariateSpline], np.array]]:  # Python 3.9 bug... pylint: disable=unsubscriptable-object
        """Function info...
        More function info...
        Args:
            events:
            event_model: An object containing data and preprocessed parameters.
            test_signal_time_profile: Background time profile to calculate_TS.
            test_background_time_profile: The time profile of signals for calculate_TS.
        Returns:
            None if there are no events, otherwise the pre-processed signal over background.
        """
        if len(events) == 0:
            return

        sig_spatial = self.injector.signal_spatial_pdf(events)
        bg_spatial = self.injector.background_spatial_pdf(events, event_model)
        splines = event_model.get_log_sob_gamma_splines(events)
        sob = sig_spatial / bg_spatial
        sig_time = test_signal_time_profile.pdf(events['time'])
        bg_time = test_background_time_profile.pdf(events['time'])
        time_sob = sig_time / bg_time
        time_sob[np.isnan(time_sob)] = 0
        if np.isinf(time_sob).sum() > 0:
            print("Warning:Background time profile doesn't not cover signal tim"
                  "e profile.Discard events outside background window!")
            time_sob[np.isinf(time_sob)] = 0
        sob *= time_sob

        return np.array(splines), np.array(sob)
