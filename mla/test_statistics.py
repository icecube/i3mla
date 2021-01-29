"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.optimize

from . import core
from . import models
from . import injectors
from . import spectral
from . import time_profiles

TsPreprocess = Tuple[List[scipy.interpolate.UnivariateSpline], np.array]
TsTimePreprocess = Tuple[List[scipy.interpolate.UnivariateSpline], np.array, np.array]
TsThreeMLPreprocess = Tuple[np.array, np.array, np.array, int]
Minimizer = Callable[
    [Callable, np.ndarray, Union[Sequence, scipy.optimize.Bounds]],  # Python 3.9 bug... pylint: disable=unsubscriptable-object
    scipy.optimize.OptimizeResult,
]


class PsTestStatistic:
    """Docstring"""

    def __init__(self) -> None:
        """Docstring"""

    @staticmethod
    def preprocess_ts(event_model: models.EventModel,
                      injector: injectors.PsInjector, source: core.Source,
                      events: np.ndarray,**kwargs
    ) -> TsPreprocess:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            source:
            events: An array of events to calculate the test-statistic for.
            event_model: An object containing data and preprocessed parameters.

        Returns:
            A list of log(sob) vs gamma splines for each event and the pre-
            processed signal over background and the timing signal over background.

        Raises:
            ValueError: There must be at least one event.
        """
        if len(events) == 0:
            #raise ValueError('len(events) must be > 0.')
            return 0,0

        sig = injector.signal_spatial_pdf(source, events)
        bkgr = injector.background_spatial_pdf(events, event_model)
        splines = event_model.get_log_sob_gamma_splines(events)

        return splines, sig / bkgr

    @staticmethod
    def calculate_ts(events: np.ndarray, preprocessing: TsPreprocess,
                     n_signal: float, gamma: float,
                     n_events: Optional[float] = None) -> np.array:  # Python 3.9 bug... pylint: disable=unsubscriptable-object
        """Evaluates the test-statistic for the given events and parameters.

        Calculates the test-statistic using a given event model, n_signal, and
        gamma. This function does not attempt to fit n_signal or gamma.

        Args:
            events: An array of events to calculate the test statistic for.
            preprocessing:
            n_signal: A guess for the number of signal events.
            gamma: A guess for the spectral index.
            n_events:

        Returns:
            The overall test-statistic value for the given events and
            parameters.
        """
        if n_events is None:
            n_events = len(events)
        splines, sob = preprocessing
        sob_new = sob * np.exp([spline(gamma) for spline in splines])
        return np.log((n_signal / n_events * (sob_new - 1)) + 1)

    @classmethod
    def minimize_ts(cls, events: np.ndarray, preprocessing: TsPreprocess,  # pylint: disable=too-many-arguments
                    test_ns: float, test_gamma: float,
                    gamma_bounds: Tuple[float] = (-4, -1),
                    minimizer: Optional[Minimizer] = None) -> Dict[str, float]:  # Python 3.9 bug... pylint: disable=unsubscriptable-object
        """Calculates the params that minimize the ts for the given events.

        Accepts guess values for fitting the n_signal and spectral index, and
        bounds on the spectral index. Uses scipy.optimize.minimize() to fit.
        The default method is 'L-BFGS-B', but can be overwritten by passing
        kwargs to this fuction.

        Args:
            events: A sample array of events to find the best fit for.
            test_ns: An initial guess for the number of signal events
                (n_signal).
            test_gamma: An initial guess for the spectral index (gamma).

        Returns:
            A dictionary containing the minimized overall test-statistic, the
            best-fit n_signal, and the best fit gamma.
        """
        if minimizer is None:
            def minimizer(func, x_0, bounds):
                return scipy.optimize.minimize(func, x0=x_0, bounds=bounds,
                                               method='L-BFGS-B')

        output = {'ts': 0, 'n_signal': test_ns, 'gamma': test_gamma}
        if len(events) == 0:
            return output

        # Check: n_signal cannot be larger than n_events
        n_events = len(events)
        if n_events <= test_ns:
            test_ns = n_events - 0.00001

        # Drop events with zero spatial or time llh
        # The contribution of those llh will be accounts in drop*np.log(1-n_signal/n_events)
        drop = n_events - np.sum(preprocessing[1] != 0)
        drop_index = preprocessing[1] != 0
        preprocessing = (np.array(preprocessing[0])[drop_index],
                         preprocessing[1][drop_index])

        def get_ts(args):
            n_signal = args[0]
            gamma = args[1]
            llhs = cls.calculate_ts(events[drop_index], preprocessing, n_signal,
                                    gamma, n_events=n_events)
            return -2 * (np.sum(llhs) + drop * np.log(1 - n_signal / n_events))

        with np.errstate(divide='ignore', invalid='ignore'):
            # Set the seed values, which tell the minimizer
            # where to start, and the bounds. First do the
            # shape parameters.
            x_0 = [test_ns, test_gamma]
            bounds = [(0, n_events-0.0001), gamma_bounds]  # gamma [min, max]
            result = minimizer(get_ts, x0=x_0, bounds=bounds)

            # Store the results in the output array
            output['ts'] = -1 * result.fun
            output['n_signal'] = result.x[0]
            output['gamma'] = result.x[1]

        return output


class PsTimeDependentTestStatistic(PsTestStatistic):
    """Docstring"""

    def __init__(self) -> None:
        """Docstring"""
        super().__init__()
        
    @staticmethod
    def preprocess_ts(event_model: models.EventModel,
                      injector: injectors.TimeDependentPsInjector, source: core.Source,
                      events: np.ndarray,
                      signal_time_profile: time_profiles.GenericProfile,
                      background_time_profile: time_profiles.GenericProfile,
                      **kwargs
    ) -> TsTimePreprocess:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            events: An array of events to calculate the test-statistic for.
            source:
            signal_time_profile: Signal time profile for ts calculation
            background_time_profile: Background time profile for ts calculation            
            event_model: An object containing data and preprocessed parameters.

            
        Returns:
            A list of log(sob) vs gamma splines for each event and the pre-
            processed signal over background.

        Raises:
            ValueError: There must be at least one event.
        """
        if len(events) == 0:
            #raise ValueError('len(events) must be > 0.')
            return 0,0,0

        sig = injector.signal_spatial_pdf(source, events)
        bkgr = injector.background_spatial_pdf(event_model, events)
        splines = event_model.get_log_sob_gamma_splines(events)
        sig_time = signal_time_profile.pdf(events['time'])
        bkg_time = background_time_profile.pdf(events['time'])
        time_ratio = sig_time/bkg_time
        if np.logical_not(np.all(np.isfinite(time_ratio))):
            print("Warning,events outside background time profile")
        
        
        return splines, sig / bkgr, time_ratio

    @staticmethod
    def calculate_ts(events: np.ndarray, preprocessing: TsTimePreprocess,
                     n_signal: float, gamma: float,
                     n_events: Optional[float] = None) -> np.array:  # Python 3.9 bug... pylint: disable=unsubscriptable-object
        """Evaluates the test-statistic for the given events and parameters.

        Calculates the test-statistic using a given event model, n_signal, and
        gamma. This function does not attempt to fit n_signal or gamma.

        Args:
            events: An array of events to calculate the test statistic for.
            preprocessing:
            n_signal: A guess for the number of signal events.
            gamma: A guess for the spectral index.
            n_events:

        Returns:
            The overall test-statistic value for the given events and
            parameters.
        """
        if n_events is None:
            n_events = len(events)
        splines, sob , time_sob = preprocessing
        sob_new = sob * np.exp([spline(gamma) for spline in splines])
        return np.log((n_signal / n_events * (sob_new*time_sob - 1)) + 1)

    @classmethod
    def minimize_ts(cls, events: np.ndarray, preprocessing: TsTimePreprocess,  # pylint: disable=too-many-arguments
                    test_ns: float = 1, test_gamma: float = -2,
                    gamma_bounds: Tuple[float] = (-4, -1),
                    minimizer: Optional[Minimizer] = None,
                    **kwargs) -> Dict[str, float]:  # Python 3.9 bug... pylint: disable=unsubscriptable-object
        """Calculates the params that minimize the ts for the given events.

        Accepts guess values for fitting the n_signal and spectral index, and
        bounds on the spectral index. Uses scipy.optimize.minimize() to fit.
        The default method is 'L-BFGS-B', but can be overwritten by passing
        kwargs to this fuction.

        Args:
            events: A sample array of events to find the best fit for.
            test_ns: An initial guess for the number of signal events
                (n_signal).
            test_gamma: An initial guess for the spectral index (gamma).
        Returns:
            A dictionary containing the minimized overall test-statistic, the
            best-fit n_signal, and the best fit gamma.
        """
        if minimizer is None:
            def minimizer(func, x_0, bounds):
                return scipy.optimize.minimize(func, x0=x_0, bounds=bounds,
                                               method='L-BFGS-B')

        output = {'ts': 0, 'n_signal': test_ns, 'gamma': test_gamma}
        if len(events) == 0:
            return output

        # Check: n_signal cannot be larger than n_events
        n_events = len(events)
        if n_events <= test_ns:
            test_ns = n_events - 0.00001

        # Drop events with zero spatial or time llh
        # The contribution of those llh will be accounts in drop*np.log(1-n_signal/n_events)
        drop = n_events - np.sum(preprocessing[1] != 0)
        drop_index = preprocessing[1] != 0
        preprocessing = (np.array(preprocessing[0])[drop_index],
                         preprocessing[1][drop_index],
                         preprocessing[2][drop_index])

        def get_ts(args):
            n_signal = args[0]
            gamma = args[1]
            llhs = cls.calculate_ts(events[drop_index], preprocessing, n_signal,
                                    gamma, n_events=n_events)
            return -2 * (np.sum(llhs) + drop * np.log(1 - n_signal / n_events))

        with np.errstate(divide='ignore', invalid='ignore'):
            # Set the seed values, which tell the minimizer
            # where to start, and the bounds. First do the
            # shape parameters.
            x_0 = [test_ns, test_gamma]
            bounds = [(0, n_events-0.0001), gamma_bounds]  # gamma [min, max]
            result = minimizer(get_ts, x_0=x_0, bounds=bounds)

            # Store the results in the output array
            output['ts'] = -1 * result.fun
            output['n_signal'] = result.x[0]
            output['gamma'] = result.x[1]

        return output
        
class PsThreeMLTestStatistic(PsTestStatistic):
    """Docstring"""

    def __init__(self) -> None:
        """Docstring"""
        super().__init__()
        
    @staticmethod
    def preprocess_ts(event_model: models.EventModel,
                      injector: injectors.TimeDependentPsInjector, source: core.Source,
                      events: np.ndarray,
                      signal_time_profile: time_profiles.GenericProfile,
                      background_time_profile: time_profiles.GenericProfile,
    ) -> TsThreeMLPreprocess:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            source:
            events: An array of events to calculate the test-statistic for.
            signal_time_profile: Signal time profile for ts calculation
            background_time_profile: Background time profile for ts calculation            
            event_model: An object containing data and preprocessed parameters.

        Returns:
            A list of log(sob) vs gamma splines for each event and the pre-
            processed signal over background.

        Raises:
            ValueError: There must be at least one event.
        """
        if len(events) == 0:
            #raise ValueError('len(events) must be > 0.')
            return 0,0,0
            


        
        sig = injector.signal_spatial_pdf(source, events)
        bkgr = injector.background_spatial_pdf(events, event_model)
        sob_spatial = sig / bkgr
        drop = n_events - np.sum(sob_spatial != 0)
        drop_index = sob_spatial != 0
        sig_time = signal_time_profile.pdf(events['time'][drop_index])
        bkg_time = background_time_profile.pdf(events['time'][drop_index])
        time_ratio = sig_time/bkg_time
        if np.any(np.isfinite(time_ratio)):
            print("Warning,events outside background time profile")
            
        return sob_spatial[drop_index], time_ratio, drop_index, len(events)

    @staticmethod
    def calculate_ts(events: np.ndarray, preprocessing: TsThreeMLPreprocess,
                     event_model: models.EventModel,
                     n_signal: float) -> np.array:  # Python 3.9 bug... pylint: disable=unsubscriptable-object
        """Evaluates the test-statistic for the given events and parameters.

        Calculates the test-statistic using a given event model, n_signal, and
        gamma. This function does not attempt to fit n_signal or gamma.

        Args:
            events: An array of events to calculate the test statistic for.
            preprocessing:
            n_signal: A guess for the number of signal events.
            n_events:

        Returns:
            The overall test-statistic value for the given events and
            parameters.
        """

        sob , time_sob, drop_index, n_events = preprocessing  
        energysob = event_model.get_energy_sob(events[drop_index])
        sob_new = sob * energysob
        return np.log((n_signal / n_events * (sob_new*time_sob - 1)) + 1)

    @classmethod
    def minimize_ts(cls, events: np.ndarray, preprocessing: TsPreprocess,  # pylint: disable=too-many-arguments
                    test_ns: float, test_gamma: float,
                    gamma_bounds: Tuple[float] = (-4, -1),
                    minimizer: Optional[Minimizer] = None) -> Dict[str, float]:  # Python 3.9 bug... pylint: disable=unsubscriptable-object
        """Calculates the params that minimize the ts for the given events.

        Accepts guess values for fitting the n_signal and spectral index, and
        bounds on the spectral index. Uses scipy.optimize.minimize() to fit.
        The default method is 'L-BFGS-B', but can be overwritten by passing
        kwargs to this fuction.

        Args:
            events: A sample array of events to find the best fit for.
            test_ns: An initial guess for the number of signal events
                (n_signal).
            test_gamma: An initial guess for the spectral index (gamma).

        Returns:
            A dictionary containing the minimized overall test-statistic, the
            best-fit n_signal, and the best fit gamma.
        """
        if minimizer is None:
            def minimizer(func, x_0, bounds):
                return scipy.optimize.minimize(func, x0=x_0, bounds=bounds,
                                               method='L-BFGS-B')

        output = {'ts': 0, 'n_signal': test_ns, 'gamma': test_gamma}
        if len(events) == 0:
            return output

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
            llhs = cls.calculate_ts(events[drop_index], preprocessing, n_signal,
                                    gamma, n_events=n_events)
            return -2 * (np.sum(llhs) + drop * np.log(1 - n_signal / n_events))

        with np.errstate(divide='ignore', invalid='ignore'):
            # Set the seed values, which tell the minimizer
            # where to start, and the bounds. First do the
            # shape parameters.
            x_0 = [test_ns, test_gamma]
            bounds = [(0, n_events-0.0001), gamma_bounds]  # gamma [min, max]
            result = minimizer(get_ts, x0=x_0, bounds=bounds)

            # Store the results in the output array
            output['ts'] = -1 * result.fun
            output['n_signal'] = result.x[0]
            output['gamma'] = result.x[1]

        return output
