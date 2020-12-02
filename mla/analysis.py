__author__ = 'John Evans'
__copyright__ = ''
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

"""
Docstring
"""

from typing import Dict, List, Optional, Union, Tuple

import scipy
import numpy as np
from mla import models
from mla import injectors
from mla import spectral
import abc
import numpy.lib.recfunctions as rf

class Analysis:
    """Abstract class defining the structure of analysis classes.

    More class info...
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None: pass

    @abc.abstractmethod
    def evaluate_ts(self, events: np.ndarray, *args, **kwargs) -> np.array: pass
    
    @abc.abstractmethod
    def minimize_ts(self, events: np.ndarray, *args, **kwargs) -> Dict[str, float]: pass
    
    @abc.abstractmethod
    def produce_trial(self, *args, **kwargs) -> np.ndarray: pass
    
    @abc.abstractmethod
    def produce_and_minimize(self, n_trials: int, *args, **kwargs) -> np.ndarray: pass


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
        if injector is not None: self.injector = injector
        else: self.injector = injectors.PsInjector(source)
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
                     recalculate:Optional[bool] = True)-> float:
        """Calculate the signal pdf"""
        if self.N == 0:
            return 0,0
        if data is not None:
                self.set_data(data)
        if recalculate:
            if data is not None:
                self.data = data
            if len(self.data) == 0: return 0
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
            self.energy_SoB = event_model.get_energy_sob(self.data)
        if ns is None:
            ns = event_model._spectrum(event_model._reduced_sim_truedec['trueE'])\
                 *event_model._reduced_sim_truedec['ow']\
                 *self.injector.signal_time_profile.effective_exposure()*3600*24
            ns = ns.sum()
            
        self.ts =( ns/self.N * (self.energy_SoB*self.Spatial_SoB*self.Time_SOB - 1))+1
        ts_value = 2*(np.sum(np.log(self.ts))+self.drop*np.log(1-ns/self.N))
        return ts_value,ns
    
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

class PsAnalysis(Analysis):
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
        if injector is not None: self.injector = injector
        else: self.injector = injectors.PsInjector(source)
    
    def _preprocess_ts(self, events: np.ndarray, event_model: models.EventModel
    ) -> Optional[Tuple[List[scipy.interpolate.UnivariateSpline], np.array]]:
        """Function info...

        More function info...

        Args:
            events:
            event_model: An object containing data and preprocessed parameters.

        Returns:
            None if there are no events, otherwise the pre-processed signal over background.
        """
        if len(events) == 0: return
            
        S = self.injector.signal_spatial_pdf(events)
        B = self.injector.background_spatial_pdf(events, event_model)
        splines = event_model.get_log_sob_gamma_splines(events)
        
        return splines, S/B
    
    def evaluate_ts(self, events: np.ndarray, event_model: models.EventModel, ns: float, gamma: float,
                    preprocessing: Optional[Tuple[List[scipy.interpolate.UnivariateSpline], np.array]] = None
    ) -> Optional[np.array]:
        """Function info...

        More function info...

        Args:
            events:
            event_model: An object containing data and preprocessed parameters.
            ns:
            gamma:

        Returns:
            
        """
        if preprocessing is None:
            preprocessing = self._preprocess_ts(events, event_model) 
            if preprocessing is None: return
        
        splines, sob = preprocessing
        sob *= np.exp([spline(gamma) for spline in splines])
        
        return np.log((ns/n_events*(sob - 1)) + 1)
    
    def minimize_ts(self, events: np.ndarray, event_model: models.EventModel,
                    test_ns: float,
                    test_gamma: float,
                    gamma_bounds: List[float] = [-4, -1],
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
        output = {'ts': np.nan, 'ns': ns, 'gamma': gamma}
        preprocessing = self._preprocess_ts(events, event_model)
        if preprocessing is None: return output
        
        # Check: ns cannot be larger than n_events
        n_events = len(events)
        if n_events <= ns:
            ns = n_events - 0.00001
            
        drop = n_events - np.sum(preprocessing[1] != 0)
        
        def get_ts(args):
            ns = args[0]
            gamma = args[1]
            llhs = self.evaluate_ts(events, ns, gamma, preprocessing=preprocessing)
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
                      flux_norm: float, 
                      reduced_sim: Optional[np.ndarray] = None, 
                      gamma: float = -2, 
                      sampling_width: Optional[float] = None, 
                      random_seed: Optional[int] = None,
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
            verbose: A flag to print progress.
            
        Returns:
            An array of combined signal and background events.
        """
        if random_seed is not None: np.random.seed(random_seed)
        
        if reduced_sim is None: 
            reduced_sim = self.injector.reduced_sim(
                flux_norm=flux_norm,
                gamma=gamma,
                sampling_width=sampling_width)

        background = self.injector.inject_background_events()
        if flux_norm > 0: signal = self.injector.inject_signal_events(reduced_sim)
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
        events = np.concatenate([background, signal])
        sorting_indices = np.argsort(events['time'])
        events = events[sorting_indices]

        # We need to check to ensure that every event is contained within
        # a good run. If the event happened when we had deadtime (when we
        # were not taking data), then we need to remove it.
        after_start = np.less(event_model.grl['start'].reshape((1, len(event_model.grl))),
                              events['time'].reshape(len(events), 1))
        before_stop = np.less(events['time'].reshape(len(events), 1),
                              event_model.grl['stop'].reshape((1, len(event_model.grl))))
        during_uptime = np.any(after_start & before_stop, axis = 1)

        return events[during_uptime]
    
    def produce_and_minimize(self,
                             event_model: models.EventModel,
                             n_trials: int,
                             test_ns: float = 1,
                             test_gamma: float = -2,
                             random_seed: Optional[int] = None,
                             flux_norm: float = 0,
                             gamma: float = -2,
                             sampling_width: Optional[float] = None,
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
            verbose: A flag to print progress.
            
        Returns:
            An array of test statistic values and their best-fit parameters for each trial.
        """
        if random_seed: np.random.seed(random_seed)

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

        # We're going to cache the signal weights, which will
        # speed up our signal generation significantly.
        signal_weights = None
        
        if verbose:
            print(f'Running Trials (N={flux_norm:3.2e}, gamma={gamma:2.1f})', end='')
            prop_complete = 0
            
        for i in range(n_trials):
            # Produce the trial events
            trial = self.produce_trial(event_model, flux_norm, reduced_sim, random_seed=random_seed)

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
        if verbose: print('done')

        return fit_info
    
class PsStackingAnalysis(Analysis):
    """Class info...

    More class info...

    Attributes:
        ps_analyses(List[PsAnalysis]):
    """
    def __init__(self, event_model: models.EventModel, sources: List[Dict[str, float]]) -> None:
        """Function info...

        More function info...

        Args:
            event_model:
            sources:
        """
        self.ps_analyses = [PsAnalysis(event_model, source) for source in sources]
    
    def evaluate_ts(self, events: np.ndarray, **fixed_args) -> np.array:
        return NotImplementedError
    
    def minimize_ts(self, events: np.ndarray, **test_args) -> Dict[str, float]:
        return NotImplementedError
    
    def produce_trial(self, **trial_args) -> np.ndarray:
        return NotImplementedError
    
    def produce_and_minimize(self, n_trials: int, **trial_and_test_args) -> np.ndarray:
        return NotImplementedError