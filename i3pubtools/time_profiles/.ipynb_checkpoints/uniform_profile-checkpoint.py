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

import numpy as np
import scipy
from i3pubtools.time_profiles import generic_profile

class UniformProfile(generic_profile.GenericProfile):
    """Time profile class for a uniform distribution. 
    
    Use this for background or if you want to assume a steady signal from
    your source.
    
    Attributes:
        window (List[float]): 
        norm (float): 
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
        self._norm = 1.0/(self._window[1]-self._window[0])
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
               (times<self._window[1])] = self._norm
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

    def effective_exposure(self) -> float:
        """Calculates the weight associated with each event time."""
        return 1.0/self._norm

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
        
    def bounds(self, time_profile: generic_profile.GenericProfile) -> List[List[float]]:
        """Given some other profile, returns allowable ranges for parameters.
        
        More function info...
        
        Args:
            time_profile (generic_profile.GenericProfile):
        """
        return [time_profile.get_range(), time_profile.get_range()]
    
    @property
    def default_params(self) -> Dict[float, float]:
        return self._default_params
    
    @property
    def param_dtype(self) -> List[Tuple[str, str]]:
        return self._param_dtype
    
    @property
    def window(self) -> List[float]:
        return tuple(self._window)
    
    @window.setter
    def window(self, new_window: List[float]) -> None:
        """Sets start and end times for the profile and computes normalization.
        
        More function info...
        
        Args:
            new_window (list): new start and end times
        """
        assert(new_window[1] > new_window[0])
        self._window = new_window
        self._norm = 1.0/(self._window[1]-self._window[0])
        
    @property
    def norm(self) -> float:
        return self._norm
        
