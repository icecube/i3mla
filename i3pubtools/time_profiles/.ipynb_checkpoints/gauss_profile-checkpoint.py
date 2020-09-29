import numpy as np
import scipy
from i3pubtools.time_profiles import generic_profile

class GaussProfile(generic_profile.GenericProfile):
    '''Time profile class for a gaussian distribution. Use this
    to produce gaussian-distributed times for your source.
    '''

    def __init__(self, mean, sigma, name = 'gauss_tp'):
        '''Constructs the object.

        Args:
            mean (float): The center form the distribution
            sigma (float): The width for the distribution
            name (string, optional): prefix for printing parameters
        '''
        self.mean = mean
        self.sigma = sigma
        self.scipy_dist = scipy.stats.norm(mean, sigma)
        self.norm = 1.0/np.sqrt(2*np.pi*sigma**2)
        self._default_params = {'_'.join([name, 'mean']):mean, '_'.join([name, 'sigma']):sigma}
        self._param_dtype = [('_'.join([name, 'mean']), np.float32),('_'.join([name, 'sigma']), np.float32)]
        return

    def pdf(self, times):
        '''Calculates the probability for each time.

        Args:
            times (np.array): A numpy list of times to evaluate
        '''
        return self.scipy_dist.pdf(times)

    def logpdf(self, times):
        '''Calculates the log(probability) for each time.

        Args:
            times (np.array): A numpy list of times to evaluate
        '''
        return self.scipy_dist.logpdf(times)

    def random(self, n=1):
        '''Returns random values following the gaussian distribution.

        Args:
            n (int, optional): The number of random values to return
        '''
        return self.scipy_dist.rvs(size=n)

    def effective_exposure(self):
        '''Calculates the weight associated with each event time.'''
        return 1.0/self.norm

    def get_range(self):
        '''Returns the min/max values for the distribution.'''
        return [-np.inf, np.inf]
    
    def x0(self, times):
        '''
        
        Args:
            times (np.array):
        '''
        x0_mean = np.average(times)
        x0_sigma = np.std(times)
        return x0_mean, x0_sigma
        
        
    def bounds(self, time_profile):
        '''
        
        Args:
            time_profile(generic_profile.GenericProfile):
        '''
        return [time_profile.get_range(), [0, time_profile.effective_exposure()]]
    
    @property
    def default_params(self):
        return self._default_params
    
    @property
    def param_dtype(self):
        return self._param_dtype
