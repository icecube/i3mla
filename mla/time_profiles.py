"""
The classes in this file are example time profiles that can be used in the
analysis classes. There is also GenericProfile, an abstract parent class to
inherit from to create other time profiles.
"""

__author__ = 'John Evans'
__copyright__ = ''
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, Dict, List, Tuple, Union

import abc
import numpy as np
import scipy.stats

class GenericProfile:
    """A generic base class to standardize the methods for the time profiles.

    While I'm only currently using scipy-based
    probability distributions, you can write your own if you
    want. Just be sure to define these methods and ensure that
    the PDF is normalized!

    Attributes:
        default_params (Dict[str, float]): A dictionary of fitting parameters
            for this time profile.
        param_dtype (List[Tuple[str, str]]): The numpy dytpe for the fitting
            parameters.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self) -> None:
        """Docstring"""

    @abc.abstractmethod
    def pdf(self, times: np.array) -> np.array:
        """Docstring"""

    @abc.abstractmethod
    def logpdf(self, times: np.array) -> np.array:
        """Docstring"""

    @abc.abstractmethod
    def random(self, size: int) -> np.array:
        """Docstring"""

    @abc.abstractmethod
    def get_range(self) -> List[float]:
        """Docstring"""

    @abc.abstractmethod
    def x0(self, times: np.array) -> Tuple: # I think this is the best name... pylint: disable=invalid-name
        """Docstring"""

    @abc.abstractmethod
    def bounds(self, time_profile: 'GenericProfile') -> List[List[float]]:
        """Docstring"""

    @property
    @abc.abstractmethod
    def default_params(self) -> Dict[str, float]:
        """Docstring"""

    @property
    @abc.abstractmethod
    def param_dtype(self) -> List[Tuple[str, str]]:
        """Docstring"""

class GaussProfile(GenericProfile):
    """Time profile class for a gaussian distribution.

    Use this to produce gaussian-distributed times for your source.

    Attributes:
        mean (float):
        sigma (float):
        scipy_dist(scipy.stats.rv_continuous):
        default_params (Dict[str, float]): A dictionary of fitting parameters
            for this time profile.
        param_dtype (List[Tuple[str, str]]): The numpy dytpe for the fitting
            parameters.
    """

    def __init__(self, mean: float, sigma: float,
                 name: str = 'gauss_tp') -> None:
        """Constructs the object.

        More function info...

        Args:
            mean: The center form the distribution.
            sigma: The width for the distribution.
            name: prefix for printing parameters.
        """
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.scipy_dist = scipy.stats.norm(mean, sigma)
        self.norm = 1.0/np.sqrt(2*np.pi*sigma**2)
        self._default_params = {'_'.join([name, 'mean']):mean,
                                '_'.join([name, 'sigma']):sigma}
        self._param_dtype = [('_'.join([name, 'mean']), np.float32),
                             ('_'.join([name, 'sigma']), np.float32)]

    def pdf(self, times: np.array) -> np.array:
        """Calculates the probability for each time.

        More function info...

        Args:
            times: A numpy list of times to evaluate.

        Returns:

        """
        return self.scipy_dist.pdf(times)

    def logpdf(self, times: np.array) -> np.array:
        """Calculates the log(probability) for each time.

        More function info...

        Args:
            times: A numpy list of times to evaluate.

        Returns:

        """
        return self.scipy_dist.logpdf(times)

    def random(self, size: int = 1) -> np.array:
        """Returns random values following the gaussian distribution.

        More function info...

        Args:
            size: The number of random values to return.

        Returns:

        """
        return self.scipy_dist.rvs(size=size)

    def get_range(self) -> List[float]:
        """Returns the min/max values for the distribution."""
        return [None, None]

    def x0(self, times: np.array) -> Tuple[float, float]:
        """Short function info...

        More function info...

        Args:
            times: A numpy list of times to evaluate.

        Returns:

        """
        x0_mean = np.average(times)
        x0_sigma = np.std(times)
        return x0_mean, x0_sigma

    def effective_exposure(self) -> float: 
        """Return the effective exposure of the gaussian
        
        Args:
            
        
        Returns:
            effctive exposure
        """
        return self.norm
        
    def bounds(self, time_profile: GenericProfile) -> List[List[float]]:
        """Short function info...

        More function info...

        Args:
            time_profile:

        Returns:

        """

        if None in time_profile.get_range():
            return [time_profile.get_range(), [0, None]]

        diff = time_profile.get_range()[1] - time_profile.get_range()[0]

        return [time_profile.get_range(), [0, diff]]

    @property
    def default_params(self) -> Dict[str, float]:
        return self._default_params

    @property
    def param_dtype(self) -> List[Tuple[str, str]]:
        return self._param_dtype

class UniformProfile(GenericProfile):
    """Time profile class for a uniform distribution.

    Use this for background or if you want to assume a steady signal from
    your source.

    Attributes:
        window (List[float]):
        default_params (Dict[str, float]): A dictionary of fitting parameters
            for this time profile.
        param_dtype (List[Tuple[str, str]]): The numpy dytpe for the fitting
            parameters.
    """

    def __init__(self, start: float, end: float,
                 name: str = 'uniform_tp') -> None:
        """Constructs the object.

        More function info...

        Args:
            start: lower bound for the uniform distribution.
            end: upper bound for the uniform distribution.
            name: prefix for parameters.
        """
        super().__init__()
        self._window = [start, end]
        self.norm = end-start
        self._default_params = {'_'.join([name, 'start']):self._window[0],
                                '_'.join([name, 'end']):self._window[1]}
        self._param_dtype = [('_'.join([name, 'start']), np.float32),
                             ('_'.join([name, 'end']), np.float32)]

    def pdf(self, times: np.array) -> np.array:
        """Calculates the probability for each time.

        More function info...

        Args:
            times: A numpy list of times to evaluate.

        Returns:

        """
        output = np.zeros_like(times)
        output[(times>=self._window[0]) &\
               (times<self._window[1])] = 1/(self._window[1] - self._window[0])
        return output

    def logpdf(self, times: np.array) -> np.array:
        """Calculates the log(probability) for each time.

        More function info...

        Args:
            times: A numpy list of times to evaluate.

        Returns:

        """
        return np.log(self.pdf(times))

    def random(self, size: int = 1) -> np.array:
        """Returns random values following the uniform distribution.

        More function info...

        Args:
            size: The number of random values to return.

        Returns:

        """
        return np.random.uniform(*self._window, size)

    def get_range(self) -> List[float]:
        """Returns the min/max values for the distribution."""
        return self._window

    def x0(self, times: np.array) -> Tuple[float, float]:
        """Short function info...

        More function info...

        Args:
            times:

        Returns:

        """
        x0_start = np.min(times)
        x0_end = np.max(times)
        return x0_start, x0_end

    def bounds(self, time_profile: GenericProfile) -> List[List[float]]:
        """Given some other profile, returns allowable ranges for parameters.

        More function info...

        Args:
            time_profile:
        """
        return [time_profile.get_range(), time_profile.get_range()]

    def effective_exposure(self) -> float: 
        """Return the effective exposure 
        
        Args:
            
        
        Returns:
            effctive exposure
        """
        return self.norm

    @property
    def default_params(self) -> Dict[str, float]:
        return self._default_params

    @property
    def param_dtype(self) -> List[Tuple[str, str]]:
        return self._param_dtype

    @property
    def window(self) -> Tuple[float]:
        """Docstring"""
        return tuple(self._window)

class CustomProfile(GenericProfile):
    """Time profile class for a custom binned distribution.

    More class info...

    Attributes:
        window (List[float]): The range of distribution function.
        pdf (Callable[[np.array, Tuple[float, float]], float]): The distribution
            function.
        default_params (Dict[str, float]): A dictionary of fitting parameters
            for this time profile.
        param_dtype (List[Tuple[str, str]]): The numpy dytpe for the fitting
            parameters.
        dist (scipy.stats.rv_histogram): The histogrammed version of the
            distribution function.
    """

    def __init__(self, pdf: Callable[[np.array, Tuple[float, float]], float],
                 time_window: Tuple[float], bins: Union[List[float], int] = 100, # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
                 name: str = 'custom_tp') -> None:
        """Constructs the object.

        More function info...

        Args:
            pdf: The distribution function (takes times and time window).
            time_window: lower and upper bound for the distribution.
            bins: Either a list of specific bin edges to use (values should be
                between 0 and 1), or an integer giving the number of linear
                spaced bins to use.
            name: prefix for parameters.
        """
        super().__init__()
        self._window = time_window
        self._default_params = {'_'.join([name, 'start']):self._window[0],
                                '_'.join([name, 'end']):self._window[1]}
        self._param_dtype = [('_'.join([name, 'start']), np.float32),
                             ('_'.join([name, 'end']), np.float32)]
        self.dist = self.build_rv(pdf, bins)

    def build_rv(self, pdf: Callable[[np.array, Tuple[float, float]], float],
                 bins: Union[List[float], int]) -> scipy.stats.rv_histogram: # Python 3.9 pylint bug... pylint: disable=unsubscriptable-object
        """Function info...

        More function info...

        Args:
            pdf: The normalized distribution function (takes times and time
                window).
            bins: Either a list of specific bin edges to use (values should be
                between 0 and 1), or an integer giving the number of linear
                spaced bins to use.

        Returns:
            The scipy histogram distribution based on the bin edges and the
            distribution function.
        """
        if isinstance(bins, int):
            bin_edges = np.linspace(*self._window, bins)
        else:
            span = self._window[1] - self._window[0]
            bin_edges = span * np.array(bins)

        bin_widths = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + bin_widths
        hist = pdf(bin_centers, tuple(self._window))
        area_under_hist = np.sum(hist * bin_widths)
        hist *= 1/area_under_hist
        self.norm = 1/np.max(hist)
        hist *= bin_widths
        
        return scipy.stats.rv_histogram((hist, bin_edges))

    def pdf(self, times: np.array) -> np.array:
        """Calculates the probability for each time.

        More function info...

        Args:
            times: A numpy list of times to evaluate.

        Returns:

        """
        return self.dist.pdf(times)

    def logpdf(self, times: np.array) -> np.array:
        """Calculates the log(probability) for each time.

        More function info...

        Args:
            times: A numpy list of times to evaluate.

        Returns:

        """
        return self.dist.logpdf(times)

    def random(self, size: int = 1) -> np.array:
        """Returns random values following the uniform distribution.

        More function info...

        Args:
            size: The number of random values to return.

        Returns:

        """
        return self.dist.rvs(size=size)

    def get_range(self) -> List[float]:
        """Returns the min/max values for the distribution."""
        return self._window

    def x0(self, times: np.array) -> Tuple[float, float]:
        """Short function info...

        More function info...

        Args:
            times:

        Returns:

        """
        x0_start = np.min(times)
        x0_end = np.max(times)
        return x0_start, x0_end

    def bounds(self, time_profile: GenericProfile) -> List[List[float]]:
        """Given some other profile, returns allowable ranges for parameters.

        More function info...

        Args:
            time_profile:
        """
        return [time_profile.get_range(), time_profile.get_range()]
        
    def effective_exposure(self) -> float: 
        """Return the effective exposure 
        
        Args:
            
        
        Returns:
            effctive exposure
        """
        return self.norm

    @property
    def default_params(self) -> Dict[str, float]:
        return self._default_params

    @property
    def param_dtype(self) -> List[Tuple[str, str]]:
        return self._param_dtype

    @property
    def window(self) -> Tuple[float]:
        """Docstring"""
        return tuple(self._window)
