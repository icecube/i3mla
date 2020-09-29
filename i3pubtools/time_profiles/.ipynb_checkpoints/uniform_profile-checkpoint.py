__author__ = 'John Evans'
__copyright__ = ''
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'jevans96@umd.edu'
__status__ = 'Development'

"""
Docstring
"""

import numpy as np
import scipy
from i3pubtools.time_profiles import generic_profile

class UniformProfile(generic_profile.GenericProfile):
    """Time profile class for a uniform distribution. Use this
    for background or if you want to assume a steady signal from
    your source.
    """

    def __init__(self, start, end, name='uniform_tp'):
        """Constructs the object.

        Args:
            start (float): lower bound for the uniform distribution
            end (float): upper bound for the uniform distribution
            name (string, optional): prefix for parameters
        """
        self._window = [start, end]
        self._norm = 1.0/(self._window[1]-self._window[0])
        self._default_params = {'_'.join([name, 'start']):self._window[0], '_'.join([name, 'end']):self._window[1]}
        self._param_dtype = [('_'.join([name, 'start']), np.float32),('_'.join([name, 'end']), np.float32)]
        return

    def pdf(self, times):
        """Calculates the probability for each time.

        Args:
            times (np.array): A numpy list of times to evaluate
        """
        output = np.zeros_like(times)
        output[(times>=self._window[0]) &\
               (times<self._window[1])] = self._norm
        return output

    def logpdf(self, times):
        """Calculates the log(probability) for each time.

        Args:
            times (np.array): A numpy list of times to evaluate
        """
        return np.log(self.pdf(times))

    def random(self, n=1):
        """Returns random values following the uniform distribution.

        Args:
            n (int, optional): The number of random values to return
        """
        return np.random.uniform(*self._window, n)

    def effective_exposure(self):
        """Calculates the weight associated with each event time."""
        return 1.0/self._norm

    def get_range(self):
        """Returns the min/max values for the distribution."""
        return self._window
    
    def x0(self, times):
        """
        
        Args:
            times (np.array):
        """
        x0_start = np.min(times)
        x0_end = np.max(times)
        return x0_start, x0_end
        
    def bounds(self, time_profile):
        """Given some other profile, returns allowable ranges for parameters.
        
        Args:
            time_profile (generic_profile.GenericProfile):
        """
        return [time_profile.get_range(), time_profile.get_range()]
    
    @property
    def default_params(self):
        return self._default_params
    
    @property
    def param_dtype(self):
        return self._param_dtype
    
    @property
    def window(self):
        return self._window
    
    @window.setter
    def window(self, new_window):
        """Sets start and end times for the profile and computes normalization.
        
        Args:
            new_window (list): new start and end times
        """
        assert(new_window[1] > new_window[0])
        self._window = new_window
        self._norm = 1.0/(self._window[1]-self._window[0])
        
    @property
    def norm(self):
        return self._norm
        
