import numpy as np
import scipy
from .generic_profile import generic_profile

class gauss_profile(generic_profile):
    '''Time profile class for a gaussian distribution. Use this
    to produce gaussian-distributed times for your source.
    '''

    def __init__(self, mean, sigma):
        '''Constructor for the class.

        args:
            mean: The center form the distribution
            sigma: The width for the distribution
        '''
        self.mean = mean
        self.sigma = sigma
        self.scipy_dist = scipy.stats.norm(mean, sigma)
        self.norm = 1.0/np.sqrt(2*np.pi*sigma**2)
        return

    def pdf(self, times):
        '''Calculates the probability for each time

        args:
            times: A numpy list of times to evaluate
        '''
        return self.scipy_dist.pdf(times)

    def logpdf(self, times):
        '''Calculates the log(probability) for each time

        args:
            times: A numpy list of times to evaluate
        '''
        return self.scipy_dist.logpdf(times)

    def random(self, n=1):
        '''Return random values following the gaussian distribution

        args:
            n: The number of random values to return
        '''
        return self.scipy_dist.rvs(size=n)

    def effective_exposure(self):
        '''Calculate the weight associated with each event time'''
        return 1.0/self.norm

    def get_range(self):
        '''Return the min/max values for the distribution'''
        return [-np.inf, np.inf]
