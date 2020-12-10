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

    More class info...
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Docstring"""

    @abc.abstractmethod
    def evaluate_ts(self, events: np.ndarray, *args, **kwargs) -> np.array:
        """Docstring"""

    @abc.abstractmethod
    def minimize_ts(self, events: np.ndarray, *args,
                    **kwargs) -> Dict[str, float]:
        """Docstring"""

    @abc.abstractmethod
    def produce_trial(self, *args, **kwargs) -> np.ndarray:
        """Docstring"""

    @abc.abstractmethod
    def produce_and_minimize(self, n_trials: int, *args,
                             **kwargs) -> np.ndarray:
        """Docstring"""


class PsAnalysis(Analysis):
    """Class info...

    More class info...

    Attributes:
        injector (injectors.PsInjector):
    """
    def __init__(self,
                 source: Optional[Dict[str, float]] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                 injector: Optional[injectors.PsInjector] = None) -> None:  # Python 3.9 bug... pylint: disable=unsubscriptable-object
        """Function info...

        More function info...

        Args:
            source:
        """
        super().__init__()
        if injector is not None:
            self.injector = injector
        else:
            self.injector = injectors.PsInjector(source)

    def _preprocess_ts(self, events: np.ndarray, event_model: models.EventModel
    ) -> TsPreprocess:
        """Function info...

        More function info...

        Args:
            events:
            event_model: An object containing data and preprocessed parameters.

        Returns:
            None if there are no events, otherwise the pre-processed signal over
            background.

        Raises:
            ValueError: There must be at least one event.
        """
        if len(events) == 0:
            raise ValueError('len(events) must be > 0.')

        sig = self.injector.signal_spatial_pdf(events)
        bkgr = self.injector.background_spatial_pdf(events, event_model)
        splines = event_model.get_log_sob_gamma_splines(events)

        return splines, sig/bkgr

    def evaluate_ts(self, events: np.ndarray, event_model: models.EventModel,  # Super class will never be called... pylint: disable=arguments-differ, too-many-arguments
                    ns: float, gamma: float,
                    n_events:Optional[int] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                    preprocessing: TsPreprocess) -> np.array:
        """Function info...

        More function info...

        Args:
            events:
            event_model: An object containing data and preprocessed parameters.
            ns:
            n_events
            gamma:

        Returns:

        """
        if n_events is None:
            n_events = len(events)
        if preprocessing is None:
            preprocessing = self._preprocess_ts(events, event_model)

        splines, sob = preprocessing
        sob_new = sob*np.exp([spline(gamma) for spline in splines])
        
        return np.log((ns/n_events*(sob_new - 1)) + 1)

    def minimize_ts(self, events: np.ndarray, event_model: models.EventModel,  # Super class will never be called... pylint: disable=arguments-differ, too-many-arguments
                    test_ns: float, test_gamma: float,
                    gamma_bounds: Tuple[float] = (-4, -1),
                    **kwargs) -> Dict[str, float]:
        """Function info...

        More function info...

        Args:
            events:
            event_model: An object containing data and preprocessed parameters.
            test_ns:
            test_gamma:

        Returns:

        """
        output = {'ts': np.nan, 'ns': test_ns, 'gamma': test_gamma}
        if len(events) == 0:
            return output
        preprocessing = self._preprocess_ts(events, event_model)

        # Check: ns cannot be larger than n_events
        n_events = len(events)
        if n_events <= test_ns:
            test_ns = n_events - 0.00001
        
        
        #Drop events with zero spatial or time llh
        #The contribution of those llh will be accounts in drop*np.log(1-ns/n_events)
        drop = n_events - np.sum(preprocessing[1] != 0)
        drop_index = preprocessing[1] != 0
        preprocessing = (preprocessing[0][drop_index],preprocessing[1][drop_index])
        def get_ts(args):
            ns = args[0]
            gamma = args[1]
            llhs = self.evaluate_ts(events[drop_index], event_model, ns, gamma, n_events = n_events, preprocessing=preprocessing)
            return -2*(np.sum(llhs) + drop*np.log(1-ns/n_events))

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
            output['ts'] = -1*result.fun
            output['ns'] = result.x[0]
            output['gamma'] = result.x[1]

        return output

    def produce_trial(self, event_model: models.EventModel, flux_norm: float,  # Super class will never be called... pylint: disable=arguments-differ, too-many-locals, too-many-arguments
                      reduced_sim: Optional[np.ndarray] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                      gamma: float = -2, sampling_width: Optional[float] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                      random_seed: Optional[int] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                      verbose: bool = False) -> np.ndarray:
        """Produces a single trial of background+signal events based on inputs.

        More function info...

        Args:
            event_model: An object containing data and preprocessed parameters.
            reduced_sim: Reweighted and pruned simulated events near the source
                declination.
            flux_norm: A flux normaliization to adjust weights.
            gamma: A spectral index to adjust weights.
            sampling_width: The bandwidth around the source declination to cut
                events.
            random_seed: A seed value for the numpy RNG.
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

        background = self.injector.inject_background_events()
        if flux_norm > 0:
            signal = self.injector.inject_signal_events(reduced_sim)
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

        return events[during_uptime]

    def produce_and_minimize(self, event_model: models.EventModel,  # Super class will never be called... pylint: disable=arguments-differ, too-many-locals, too-many-arguments
                             n_trials: int, test_ns: float = 1,
                             test_gamma: float = -2,
                             random_seed: Optional[int] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                             flux_norm: float = 0, gamma: float = -2,
                             sampling_width: Optional[float] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                             verbose: bool = False) -> np.ndarray:
        """Produces n trials and calculate a test statistic for each trial.

        More function info...

        Args:
            event_model: An object containing data and preprocessed parameters.
            n_trials: The number of times to repeat the trial + evaluate_ts
                process.
            test_ns: A guess for the number of signal events.
            test_gamma: A guess for best fit spectral index of the signal.
            random_seed: A seed value for the numpy RNG.
            flux_norm: A flux normaliization to adjust weights.
            gamma: A guess for best fit spectral index of the signal.
            sampling_width: The bandwidth around the source declination to cut
                events.
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
            ('ns', np.float32),
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
                                       random_seed=random_seed)

            # And get the weights
            bestfit = self.minimize_ts(trial, event_model, test_ns, test_gamma)

            fit_info['ts'][i] = bestfit['ts']
            fit_info['ntot'][i] = len(trial)
            fit_info['ninj'][i] = (trial['run'] > 200000).sum()
            fit_info['ns'][i] = bestfit['ns']
            fit_info['gamma'][i] = bestfit['gamma']

            if verbose and i/n_trials > prop_complete:
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
                 source: Optional[Dict[str, float]] = None,
                 injector: Optional[injectors.PsInjector] = None) -> None:
        """Function info...
        More function info...
        Args:
            source:
            injector:
        """
        super().__init__()
        if injector is not None: self.injector = injector
        else: self.injector = injectors.TimeDependentPsInjector(source)
        self.first_spatial = False
        
    def _init_event_model(self,
                          data: np.ndarray,
                          sim: np.ndarray,
                          grl: np.ndarray,
                          background_sin_dec_bins: Union[np.array, int] = 500,
                          signal_sin_dec_bins: Union[np.array, int] = 50,
                          log_energy_bins: Union[np.array, int] = 50,
                          spectrum: Optional[spectral.BaseSpectrum] = None,
                          sampling_width: Optional[float] = np.radians(3),
                          reduce: Optional[bool] = True,
                          verbose: bool = False) -> models.EventModel:
        """Set the event model
        More function info...
        Args:
            data: data
            sim: simulation set
            grl: grl
            background_sin_dec_bins: If an int, then the number of bins spanning -1 -> 1,
                otherwise, a numpy array of bin edges.
            signal_sin_dec_bins: If an int, then the number of bins spanning -1 -> 1,
                otherwise, a numpy array of bin edges.
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
        
    def set_data(self, data: np.ndarray)->None:
        """Set data
        More function info...
        
        Args:
            data: data events
        """
        self.data = data
        self.N = len(data)
        self.first_spatial = True
        return
            
    def calculate_TS(self, 
                     event_model: models.EventModel, 
                     data: Optional[np.ndarray] = None, 
                     ns: Optional[float] = None,
                     test_signal_time_profile:Optional[time_profiles.GenericProfile] = None,
                     test_background_time_profile:Optional[time_profiles.GenericProfile] = None,
                     recalculate:Optional[bool] = True)-> float:
        """Calculate the signal pdf"""
        if self.N == 0:
            return 0,0
        if data is not None:
                self.set_data(data)
        if recalculate:
            if data is not None:
                self.data = data
            if len(self.data) == 0: return 0,0
            if self.first_spatial == True:
                self.Spatial_S = self.injector.signal_spatial_pdf(self.data)
                mask = self.Spatial_S!=0
                self.data = self.data[mask]
                self.Spatial_S = self.Spatial_S[mask]
                self.Spatial_B = self.injector.background_spatial_pdf(self.data,event_model)
                self.Spatial_SoB = self.Spatial_S/self.Spatial_B
                self.drop = self.N - mask.sum()
                self.first_spatial = False
                if test_signal_time_profile is None or test_background_time_profile is None:
                    if self.injector.signal_time_profile is None:
                        Time_S = 1
                        Time_B = 1
                    else:
                        Time_S = self.injector.signal_time_profile.pdf(self.data['time'])
                        Time_B = self.injector.background_time_profile.pdf(self.data['time'])
                else:
                    Time_S = test_signal_time_profile.pdf(self.data['time'])
                    Time_B = test_background_time_profile.pdf(self.data['time'])
                    
                self.Time_SOB = Time_S/Time_B
                self.Time_SOB[np.isnan(self.Time_SOB)] = 0
            self.energy_SoB = event_model.get_energy_sob(self.data)
        if ns is None:
            ns = event_model._spectrum(event_model._reduced_sim_truedec['trueE'])\
                 *event_model._reduced_sim_truedec['ow']\
                 *self.injector.signal_time_profile.effective_exposure()*3600*24
            ns = ns.sum()
            
        self.ts =( ns/self.N * (self.energy_SoB*self.Spatial_SoB*self.Time_SOB - 1))+1
        ts_value = 2*(np.sum(np.log(self.ts))+self.drop*np.log(1-ns/self.N))
        if np.isnan(ts_value):
            self.ts = 0
        return ts_value,ns
    
    def produce_trial(self,
                      event_model: models.EventModel,
                      spectrum:Optional[spectral.BaseSpectrum] = None, 
                      reduced_sim: Optional[np.ndarray] = None, 
                      nsignal: Optional[int] = None,
                      sampling_width: Optional[float] = None, 
                      random_seed: Optional[int] = None,
                      signal_time_profile:Optional[time_profiles.GenericProfile] = None,
                      background_time_profile: Optional[time_profiles.GenericProfile] = None,
                      background_window: Optional[float] = 14,
                      withinwindow: Optional[bool] = False,
                      verbose: bool = False) -> np.ndarray:    
        """Produce trial
        
        More function info...
        
        Args:
            event_model: An object containing data and preprocessed parameters.
            spectrum: Spectrum of the injection
            reduced_sim: Reweighted and pruned simulated events near the source declination.
            sampling_width: The bandwidth around the source declination to cut events.
            random_seed: A seed value for the numpy RNG.
            signal_time_profile: The time profile of the injected signal.
            background_time_profile: Background time profile to do the injection.
            disable_time_filter:Cut out events that is not in grl
            verbose: A flag to print progress.
            
        Returns:
            An array of combined signal and background events.
        """
        if random_seed is not None: np.random.seed(random_seed)
        
        background = self.injector.inject_background_events(event_model, 
                                                            background_time_profile = background_time_profile, 
                                                            background_window = background_window, 
                                                            withinwindow = withinwindow)
        
        
        if signal_time_profile is not None:
            self.injector.set_signal_profile(signal_time_profile)
        
        livetime = self.injector.signal_time_profile.effective_exposure()
        
        if nsignal is None:
            if spectrum is None:
                try:
                    data = self.injector.inject_signal_events(event_model._reduced_sim_truedec, signal_time_profile=signal_time_profile)
                except:
                    raise "No spectrum had even supplied"
            else:
                event_model._reduced_sim_truedec = self.injector.reduced_sim(event_model = event_model,spectrum=spectrum, livetime =livetime)
                data = self.injector.inject_signal_events(event_model._reduced_sim_truedec, signal_time_profile=signal_time_profile)    
        else:
            if spectrum is None:
                try:
                    data = self.injector.inject_nsignal_events(event_model._reduced_sim_truedec,nsignal, signal_time_profile=signal_time_profile)
                except:
                    raise "No spectrum had even supplied"
            else:
                event_model._reduced_sim_truedec = self.injector.reduced_sim(event_model = event_model, spectrum=spectrum, livetime = livetime)
                data = self.injector.inject_nsignal_events(event_model._reduced_sim_truedec, nsignal, signal_time_profile=signal_time_profile)
                
        bgrange = self.injector.background_time_profile.get_range()
        contained_in_background = ((data['time'] >= bgrange[0]) &\
                                   (data['time'] < bgrange[1]))
        data = data[contained_in_background]
        data = rf.drop_fields(data, [n for n in data.dtype.names \
                 if not n in background.dtype.names])    
        
        return np.concatenate([background,data])        
        
    def min_ns(self, 
               event_model: models.EventModel, 
               data: Optional[np.ndarray] = None,
               recalculate:Optional[bool] = True)-> float:
        """minimize ns"""
        
        if data is not None:
            self.set_data(data)
            
        if self.N == 0:
            return 0,0
        bounds= [[0, self.N ],]

        if recalculate:
            if self.first_spatial == True:
                self.Spatial_S = self.injector.signal_spatial_pdf(self.data)
                mask = self.Spatial_S!=0
                self.data = self.data[mask]
                self.Spatial_S = self.Spatial_S[mask]
                self.Spatial_B = self.injector.background_spatial_pdf(self.data,event_model)
                self.Spatial_SoB = self.Spatial_S/self.Spatial_B
                self.drop = self.N - mask.sum()
                self.first_spatial = False
                if self.injector.signal_time_profile is None:
                    Time_S = 1
                    Time_B = 1
                else:
                    Time_S = self.injector.signal_time_profile.pdf(self.data['time'])
                    Time_B = self.injector.background_time_profile.pdf(self.data['time'])
                self.Time_SOB = Time_S/Time_B
                self.Time_SOB[np.isnan(self.Time_SOB)] = 0
                if np.isinf(self.Time_SOB).sum() > 0:
                    print("Warning:Background time profile doesn't not cover signal time profile.Discard events outside background window!")
                    time_SoB[np.isinf(self.time_SoB)] = 0
            self.energy_SoB = event_model.get_energy_sob(self.data)
        
        def cal_ts(ns):
            ts =( ns/self.N * (self.energy_SoB*self.Spatial_SoB*self.Time_SOB - 1))+1
            ts_value = -2*(np.sum(np.log(ts))+self.drop*np.log(1-ns/self.N))
            return ts_value
        result = scipy.optimize.minimize(cal_ts,
                    x0 = [1,],
                    bounds = bounds,
                    method = 'SLSQP',
                    )     
        fit_ns = result.x[0]
        fit_ts = -1*result.fun
        if np.isnan(fit_ts):
            fit_ts = 0
            fit_ns = 0
        return fit_ts,fit_ns             
        

class TimeDependentPsAnalysis(Analysis):
    """Docstring"""

    def __init__(self,
                 source: Optional[Dict[str, float]] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                 background_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                 signal_time_profile: Optional[time_profiles.GenericProfile] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                 injector: Optional[injectors.TimeDependentPsInjector] = None,  # Python 3.9 bug... pylint: disable=unsubscriptable-object
                 **kwargs
    ) -> None:
        """Function info...
        More function info...
        Args:
            source:
            background_time_profile:
            signal_time_profile:
            injector:
        """
        super().__init__()
        if injector is not None: self.injector = injector
        else: self.injector = injectors.TimeDependentPsInjector(source)
        self.injector.background_time_profile = background_time_profile
        if self.injector.background_time_profile is not None: self.injector.set_background_profile(self.injector.background_time_profile,**kwargs)
        self.injector.signal_time_profile = signal_time_profile
    
    def _preprocess_ts(self, events: np.ndarray, event_model: models.EventModel,
                       test_signal_time_profile:Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                       test_background_time_profile:Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                       ) -> Optional[Tuple[List[scipy.interpolate.UnivariateSpline], np.array]]:# Python 3.9 bug... pylint: disable=unsubscriptable-object
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
        if len(events) == 0: return
            
        S = self.injector.signal_spatial_pdf(events)
        B = self.injector.background_spatial_pdf(events, event_model)
        splines = event_model.get_log_sob_gamma_splines(events)
        SoB = S/B
        if test_signal_time_profile is not None and test_background_time_profile is not None:
            S_time = test_signal_time_profile.pdf(events['time'])
            B_time = test_background_time_profile.pdf(events['time'])
            time_SoB = S_time/B_time
            time_SoB[np.isnan(time_SoB)] = 0
            if np.isinf(time_SoB).sum() > 0:
                print("Warning:Background time profile doesn't not cover signal time profile.Discard events outside background window!")
                time_SoB[np.isinf(time_SoB)] = 0
            SoB *= time_SoB

        
        return np.array(splines), np.array(SoB)
    
    def evaluate_ts(self, events: np.ndarray, event_model: models.EventModel, ns: float, gamma: float,
                    n_events:Optional[int] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                    test_signal_time_profile:Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                    test_background_time_profile:Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                    preprocessing: Optional[Tuple[List[scipy.interpolate.UnivariateSpline], np.array]] = None# Python 3.9 bug... pylint: disable=unsubscriptable-object
    ) -> Optional[np.array]:# Python 3.9 bug... pylint: disable=unsubscriptable-object
        """Function info...
        More function info...
        Args:
            events:
            event_model: An object containing data and preprocessed parameters.
            ns:
            gamma:
        Returns:
            
        """
        if n_events is None:
            n_events = len(events)
        if preprocessing is None:
            preprocessing = self._preprocess_ts(events, 
                                                event_model,
                                                test_signal_time_profile=test_signal_time_profile,
                                                test_background_time_profile=test_background_time_profile) 
            if preprocessing is None: return
        splines, sob = preprocessing
        sob_new = sob*np.exp([spline(gamma) for spline in splines])
        
        return np.log((ns/n_events*(sob_new - 1)) + 1)
    
    def minimize_ts(self, events: np.ndarray, event_model: models.EventModel,
                    test_ns: float,
                    test_gamma: float,
                    gamma_bounds: List[float] = [-4, -1],
                    test_signal_time_profile:Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                    test_background_time_profile:Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                    **kwargs) -> Dict[str, float]:
        """Function info...
        More function info...
        Args:
            events:
            event_model: An object containing data and preprocessed parameters.
            test_ns:
            test_gamma:
            test_signal_time_profile: Background time profile to calculate_TS.
            test_background_time_profile: The time profile of signals for calculate_TS.
        Returns:
            
        """
        ns = test_ns
        gamma = test_gamma
        output = {'ts': 0, 'ns': ns, 'gamma': gamma}
        preprocessing = self._preprocess_ts(events, event_model,test_signal_time_profile = test_signal_time_profile,test_background_time_profile = test_background_time_profile)
        if preprocessing is None: return output
        
        # Check: ns cannot be larger than n_events
        n_events = len(events)
        if n_events <= ns:
            ns = n_events - 0.00001
            
        drop = n_events - np.sum(preprocessing[1] != 0)
        drop_index = preprocessing[1] != 0
        preprocessing = (preprocessing[0][drop_index],preprocessing[1][drop_index])
        def get_ts(args):
            ns = args[0]
            gamma = args[1]
            llhs = self.evaluate_ts(events[drop_index], event_model, ns, gamma, n_events, preprocessing=preprocessing)
            return -2*(np.sum(llhs) + drop*np.log(1-ns/n_events))
            
        with np.errstate(divide='ignore', invalid='ignore'):
            # Set the seed values, which tell the minimizer
            # where to start, and the bounds. First do the
            # shape parameters.
            x0 = [test_ns, test_gamma]
            bounds = [[0, n_events], gamma_bounds] # gamma [min, max]
            if 'method' not in kwargs: kwargs['method'] = 'L-BFGS-B'
                
            result = scipy.optimize.minimize(get_ts, x0=x0, bounds=bounds, **kwargs)

            # Store the results in the output array
                
            output['ts'] = -1*result.fun
            output['ns'] = result.x[0]
            output['gamma'] = result.x[1]
        
        return output
    
    def produce_trial(self,
                      event_model: models.EventModel,
                      flux_norm: Optional[float] = 1e-9, # Python 3.9 bug... pylint: disable=unsubscriptable-object
                      reduced_sim: Optional[np.ndarray] = None, # Python 3.9 bug... pylint: disable=unsubscriptable-object
                      gamma: float = -2,
                      nsignal: Optional[int] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                      sampling_width: Optional[float] = None, # Python 3.9 bug... pylint: disable=unsubscriptable-object
                      random_seed: Optional[int] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                      signal_time_profile:Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                      background_time_profile: Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                      disable_time_filter: Optional[bool] = False,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                      verbose: bool = False) -> np.ndarray:
        """Produces a single trial of background+signal events based on input parameters.
        
        More function info...
        
        Args:
            event_model: An object containing data and preprocessed parameters.
            reduced_sim: Reweighted and pruned simulated events near the source declination.
            flux_norm: A flux normaliization to adjust weights.
            gamma: A spectral index to adjust weights.
            sampling_width: The bandwidth around the source declination to cut events.
            random_seed: A seed value for the numpy RNG.
            signal_time_profile: The time profile of the injected signal.
            background_time_profile: Background time profile to do the injection.
            disable_time_filter:Cut out events that is not in grl
            verbose: A flag to print progress.
            
        Returns:
            An array of combined signal and background events.
        """
        if random_seed is not None: np.random.seed(random_seed)
        if background_time_profile is not None:
            if not issubclass(type(background_time_profile) ,time_profiles.GenericProfile):
                background_time_profile = time_profiles.UniformProfile(background_time_profile[0],
                                                                       background_time_profile[1])
        if signal_time_profile is not None:                                                            
            if not issubclass(type(signal_time_profile) ,time_profiles.GenericProfile):
                signal_time_profile = time_profiles.UniformProfile(signal_time_profile[0],
                                                                   signal_time_profile[1]) 
        if reduced_sim is None: 
            reduced_sim = self.injector.reduced_sim(
                event_model=event_model,
                flux_norm=flux_norm,
                gamma=gamma,
                sampling_width=sampling_width)
        background = self.injector.inject_background_events(event_model,background_time_profile=background_time_profile)
        if nsignal is not None:
            signal = self.injector.inject_nsignal_events(reduced_sim,nsignal,signal_time_profile=signal_time_profile)
            
        else:
            if flux_norm > 0: signal = self.injector.inject_signal_events(reduced_sim,signal_time_profile=signal_time_profile)
            else: signal = np.empty(0, dtype=background.dtype)
        
        if verbose:
            print(f'number of background events: {len(background)}')
            print(f'number of signal events: {len(signal)}')
            
        # Because we want to return the entire event and not just the
        # number of events, we need to do some numpy magic. Specifically,
        # we need to remove the fields in the simulated events that are
        # not present in the data events. These include the true direction,
        # energy, and 'oneweight'.
        signal = rf.drop_fields(signal, [n for n in signal.dtype.names \
                                         if not n in background.dtype.names])
        
            
        # Combine the signal background events and time-sort them.
        
        if signal_time_profile is None and self.injector.signal_time_profile is None:
            signal['time'] = time_profiles.UniformProfile(event_model.grl['start'][0],event_model.grl['stop'][-1]).random(len(signal))
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
            during_uptime = np.any(after_start & before_stop, axis = 1)
            events = events[during_uptime]
        return events
    
    def produce_and_minimize(self,
                             event_model: models.EventModel,
                             n_trials: int,
                             test_ns: float = 1,
                             test_gamma: float = -2,
                             random_seed: Optional[int] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                             flux_norm: float = 0,
                             gamma: float = -2,
                             nsignal:Optional[int] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                             sampling_width: Optional[float] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                             signal_time_profile:Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                             background_time_profile: Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                             background_window: Optional[float] = 14,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                             test_signal_time_profile:Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                             test_background_time_profile:Optional[time_profiles.GenericProfile] = None,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                             disable_time_filter: Optional[bool] = False,# Python 3.9 bug... pylint: disable=unsubscriptable-object
                             verbose: bool = False,
    ) -> np.ndarray:
        """Produces n trials and calculate a test statistic for each trial.
        
        More function info...
        
        Args:
            event_model: An object containing data and preprocessed parameters.
            n_trials: The number of times to repeat the trial + evaluate_ts process.
            test_ns: A guess for the number of signal events.
            test_gamma: A guess for best fit spectral index of the signal.
            random_seed: A seed value for the numpy RNG.
            flux_norm: A flux normaliization to adjust weights.
            gamma: A guess for best fit spectral index of the signal.
            sampling_width: The bandwidth around the source declination to cut events.
            signal_time_profile: The time profile of the injected signal.
            background_time_profile: Background time profile to do the injection.
            background_window: background time window used to estimated the background rate
            test_signal_time_profile: Background time profile to calculate_TS.
            test_background_time_profile: The time profile of signals for calculate_TS.
            disable_time_filter:Cut out events that is not in grl
            verbose: A flag to print progress.
        Returns:
            An array of test statistic values and their best-fit parameters for each trial.
        """
        if random_seed: np.random.seed(random_seed)
        if background_time_profile is not None:
            if not issubclass(type(background_time_profile) ,time_profiles.GenericProfile):
                background_time_profile = time_profiles.UniformProfile(background_time_profile[0],
                                                                       background_time_profile[1])
            if test_background_time_profile is None:
                test_background_time_profile = background_time_profile
            elif not issubclass(type(test_background_time_profile),time_profiles.GenericProfile):
                test_background_time_profile = time_profiles.UniformProfile(test_background_time_profile[0],
                                                                            test_background_time_profile[1])
                                                                       
        if signal_time_profile is not None:                                                            
            if not issubclass(type(signal_time_profile) ,time_profiles.GenericProfile):
                signal_time_profile = time_profiles.UniformProfile(signal_time_profile[0],
                                                                   signal_time_profile[1])
            if test_signal_time_profile is None:
                test_signal_time_profile = signal_time_profile
            elif not issubclass(type(test_signal_time_profile) ,time_profiles.GenericProfile):
                test_signal_time_profile = time_profiles.UniformProfile(test_signal_time_profile[0],
                                                                        test_signal_time_profile[1])
        # Cut down the sim. We're going to be using the same
        # source and weights each time, so this stops us from
        # needing to recalculate over and over again.
        # Also set the signal time profile for injector
        self.injector.set_signal_profile(signal_time_profile)
        self.injector.set_background_profile(event_model,background_time_profile,background_window)
        reduced_sim = self.injector.reduced_sim(event_model,
                                                flux_norm=flux_norm,
                                                gamma=gamma,
                                                sampling_width=sampling_width)
        
        
        # Build a place to store information for the trial
        dtype = np.dtype([
            ('ts', np.float32),
            ('ntot', np.int),
            ('ninj', np.int),
            ('ns', np.float32),
            ('gamma', np.float32),
        ])
        fit_info = np.empty(n_trials, dtype=dtype)

        # We're going to cache the signal weights, which will
        # speed up our signal generation significantly.
        signal_weights = None
        
        if verbose:
            print(f'Running Trials (N={flux_norm:3.2e}, gamma={gamma:2.1f})', end='')
            prop_complete = 0
            
        for i in range(n_trials):
            # Produce the trial events
            trial = self.produce_trial(event_model,
                                       flux_norm,
                                       reduced_sim, 
                                       nsignal = nsignal,
                                       random_seed=random_seed,
                                       disable_time_filter=disable_time_filter,
                                       verbose = verbose)
            
            # And get the weights
            bestfit = self.minimize_ts(trial, 
                                       event_model, 
                                       test_ns, 
                                       test_gamma,
                                       test_signal_time_profile = test_signal_time_profile,
                                       test_background_time_profile = test_background_time_profile)

            fit_info['ts'][i] = bestfit['ts']
            fit_info['ntot'][i] = len(trial)
            fit_info['ninj'][i] = (trial['run'] > 200000).sum()
            fit_info['ns'][i] = bestfit['ns']
            fit_info['gamma'][i] = bestfit['gamma']
            
            if verbose and i/n_trials > prop_complete:
                prop_complete += .05
                print('.', end='')
        if verbose: print('done')

        return fit_info
    