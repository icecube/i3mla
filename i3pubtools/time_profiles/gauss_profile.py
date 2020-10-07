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

class GaussProfile(generic_profile.GenericProfile):
    """Time profile class for a gaussian distribution. 
    
    Use this to produce gaussian-distributed times for your source.
    
    Attributes:
        mean (float): 
        sigma (float): 
        scipy_dist(scipy.stats.rv_continuous): 
        norm (float): 
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
        self.norm = 1.0/np.sqrt(2*np.pi*sigma**2)
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

    def effective_exposure(self) -> float:
        """Calculates the weight associated with each event time."""
        return 1.0/self.norm

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
        
        
    def bounds(self, time_profile: generic_profile.GenericProfile) -> List[List[float]]:
        """Short function info...
        
        More function info...
        
        Args:
            time_profile: 
            
        Returns:
            
        """
        return [time_profile.get_range(), [0, time_profile.effective_exposure()]]
    
    @property
    def default_params(self) -> Dict[float, float]:
        return self._default_params
    
    @property
    def param_dtype(self) -> List[Tuple[str, str]]:
        return self._param_dtype
