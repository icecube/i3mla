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

from typing import Dict, List, Tuple

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
        default_params (Dict): A dictionary of fitting parameters for this time profile.
        param_dtype (List[Tuple[str, str]]): The numpy dytpe for the fitting parameters. 
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self) -> None: pass

    @abc.abstractmethod
    def pdf(self, times: np.array) -> np.array: pass

    @abc.abstractmethod
    def logpdf(self, times: np.array) -> np.array: pass

    @abc.abstractmethod
    def random(self, size: int) -> np.array: pass

    @abc.abstractmethod
    def get_range(self) -> List[float]: pass
    
    @abc.abstractmethod
    def x0(self, times: np.array) -> Tuple: pass
    
    @abc.abstractmethod
    def bounds(self, time_profile: 'GenericProfile') -> List[List[float]]: pass

    @property
    @abc.abstractmethod
    def default_params(self) -> Dict: pass
    
    @property
    @abc.abstractmethod
    def param_dtype(self) -> List[Tuple[str, str]]: pass
    
class GaussProfile(GenericProfile):
    """Time profile class for a gaussian distribution. 
    
    Use this to produce gaussian-distributed times for your source.
    
    Attributes:
        mean (float): 
        sigma (float): 
        scipy_dist(scipy.stats.rv_continuous): 
        default_params (Dict[float, float]): A dictionary of fitting parameters for this time profile.
        param_dtype (List[Tuple[str, str]]): The numpy dytpe for the fitting parameters.
    """

    def __init__(self, mean: float, sigma: float, name: str = 'gauss_tp') -> None:
        """Constructs the object.
        
        More function info...

        Args:
            mean: The center form the distribution.
            sigma: The width for the distribution.
            name: prefix for printing parameters.
        """
        self.mean = mean
        self.sigma = sigma
        self.scipy_dist = scipy.stats.norm(mean, sigma)
        self._default_params = {'_'.join([name, 'mean']):mean, '_'.join([name, 'sigma']):sigma}
        self._param_dtype = [('_'.join([name, 'mean']), np.float32),('_'.join([name, 'sigma']), np.float32)]
        return

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
    def default_params(self) -> Dict[float, float]:
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
        default_params (Dict[float, float]): A dictionary of fitting parameters for this time profile.
        param_dtype (List[Tuple[str, str]]): The numpy dytpe for the fitting parameters.
    """

    def __init__(self, start: float, end: float, name: str = 'uniform_tp') -> None:
        """Constructs the object.
        
        More function info...

        Args:
            start: lower bound for the uniform distribution.
            end: upper bound for the uniform distribution.
            name: prefix for parameters.
        """
        self._window = [start, end]
        self._default_params = {'_'.join([name, 'start']):self._window[0], '_'.join([name, 'end']):self._window[1]}
        self._param_dtype = [('_'.join([name, 'start']), np.float32),('_'.join([name, 'end']), np.float32)]
        return

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
    
    @property
    def default_params(self) -> Dict[float, float]:
        return self._default_params
    
    @property
    def param_dtype(self) -> List[Tuple[str, str]]:
        return self._param_dtype
    
    @property
    def window(self) -> Tuple[float]:
        return tuple(self._window)