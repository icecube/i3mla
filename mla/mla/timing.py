'''Timing Profile'''

from __future__ import print_function, division
import numpy as np
import scipy
import abc

class generic_profile(object):
    r""" A generic base class to standardize the methods for the
    time profiles."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self,): pass
    
    @abc.abstractmethod
    def pdf(self, times): pass

    @abc.abstractmethod
    def logpdf(self, times): pass
    
    @abc.abstractmethod
    def random(self, n): pass

    @abc.abstractmethod
    def effective_exposure(self, times): pass
    
    @abc.abstractmethod
    def get_range(self): pass
    

class uniform_profile(generic_profile):
    r"""Time profile class for a uniform distribution."""
    def __init__(self, start_time, end_time):
        r""" Constructor for the class."""
        assert(end_time > start_time)
        self.start_time = start_time
        self.end_time = end_time
        self.norm = 1.0/(end_time-start_time)
        return
    
    def pdf(self, times):
        r""" Calculates the probability for each time."""
        output = np.zeros_like(times)
        output[(times>=self.start_time) &\
               (times<self.end_time)] = self.norm
        return output
    
    def logpdf(self, times):
        r""" Calculates the log(probability) for each time."""
        return np.log(self.pdf(times))
    
    def random(self, n=1): 
        r""" Return random values following the uniform distribution
        
        args:
            n: The number of random values to return
        """
        return np.random.uniform(self.start_time,
                                 self.end_time,
                                 n)
    
    def effective_exposure(self): 
        r""" Calculate the weight associated with each
            event time. 
        """
        return 1.0/self.norm
    
    def get_range(self): 
        r""" Return the min/max values for the distribution 
        """
        return [self.start_time, self.end_time]
    

class gauss_profile(generic_profile):
    r"""Time profile class for a gaussian distribution. Use this
    to produce gaussian-distributed times for your source.
    """
    def __init__(self, mean, sigma):
        r""" Constructor for the class.

        args:
            mean: The center form the distribution
            sigma: The width for the distribution
        """
        self.mean = mean
        self.sigma = sigma
        self.scipy_dist = scipy.stats.norm(mean, sigma)
        self.norm = 1.0/np.sqrt(2*np.pi*sigma**2)
        return
    
    def pdf(self, times):
        r""" Calculates the probability for each time
            
        args:
            times: A numpy list of times to evaluate
        """
        return self.scipy_dist.pdf(times)
    
    def logpdf(self, times):
        r""" Calculates the log(probability) for each time
            
        args:
            times: A numpy list of times to evaluate
        """
        return self.scipy_dist.logpdf(times)
        
    def random(self, n=1): 
        r""" Return random values following the gaussian distribution
        
        args:
            n: The number of random values to return
        """
        return self.scipy_dist.rvs(size=n)
    
    def effective_exposure(self): 
        r""" Calculate the weight associated with each
            event time. 
        """
        return 1.0/self.norm
    
    def get_range(self): 
        r""" Return the min/max values for the distribution 
        """
        return [-np.inf, np.inf]