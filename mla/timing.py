'''Timing Profile'''

from __future__ import print_function, division
import numpy as np
import scipy
import abc
import scipy.stats

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
    
    @abc.abstractmethod
    def change_offset(self, offset): pass
    

class uniform_profile(generic_profile):
    r"""Time profile class for a uniform distribution."""
    def __init__(self, start_time, end_time):
        r""" Constructor for the class.
        args:
        start_time: Float
        The start time of the profile
        
        end_time: Float
        The end time of the profile
        """
        assert(end_time > start_time)
        self.start_time = start_time
        self.end_time = end_time
        self.norm = 1.0/(end_time-start_time)
        self.offset = 0
        return
    
    def pdf(self, times):
        r""" Calculates the probability for each time.
        
        args:
        times: Float or array.
        The evaluation time.
        """
        output = np.zeros_like(times)
        output[(times-self.offset>=self.start_time) &\
               (times-self.offset<self.end_time)] = self.norm
        return output
    
    def logpdf(self, times):
        r""" Calculates the log(probability) for each time.
        args:
        times: Float or array 
        The evaluation time.
        """
        
        return np.log(self.pdf(times))
    
    def random(self, n=1): 
        r""" Return random values following the uniform distribution
        args:
        n: The number of random values to return   
        """
        return np.random.uniform(self.start_time+self.offset,
                                 self.end_time+self.offset,
                                 n)
    
    def effective_exposure(self): 
        r""" Calculate the weight associated with each
            event time. 
        
        returns:
        (Float)The time range of the uniform profile.
        """
        return 1.0/self.norm
    
    def get_range(self): 
        r""" Return the min/max values for the distribution 
        """
        return [self.start_time+self.offset, self.end_time+self.offset]
    
    def change_offset(self, offset):
        r'''Change the offset
        args:
        offset: Float
        New offset
        '''
        self.offset = offset
        return

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
        self.offset = 0
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
        
        returns:
        N samples drawn from pdf.
        """
        return self.scipy_dist.rvs(size=n)
    
    def effective_exposure(self): 
        r""" Calculate the weight associated with each
            event time. 
            
        return:
        The effective exposure(normalization)
        """
        return 1.0/self.norm
    
    def get_range(self): 
        r""" Return the min/max values for the distribution 
        """
        return [-np.inf, np.inf]
        
    def change_offset(self, offset):
        r"""Change the offset of gaussian(add the offset to mean
        
        args:
        The New offset
        """
        self.offset = offset
        self.scipy_dist = scipy.stats.norm(mean + self.offset, sigma)
        return
        
class custom_profile(generic_profile):
    r"""Time profile class for a user-defined pdf. Notice that the pdf have to be normalized. The effective_exposure is chosen such that the peak of the profile is 1.
    """
    def __init__(self, time_pdf , range , grid = None , size = 10000 ):
        r""" Constructor for the class.

        args:
        time_pdf : the normalized pdf
        range : the time range of the pdf(have to be finite)
        grid(optional) : the time points that will be evaluated(overide size args)
        size(optional) : the number of time points that will be evaluated with linear spacing
        """
        self.pdf = time_pdf
        self.range = range 
        if grid is None:
            self.grid = np.linspace(range[0],range[1], size) #grid size 
        else :
            self.grid = grid
        self.dist = self.build_rv()
        self.offset = 0
        return
    
    def build_rv(self):
        r""" build the distribution using scipy
        return:
        The scipy distribution object.
        """
        hist = self.pdf(self.grid[:-1])
        peak = self.grid[np.argmax(hist)]
        self.norm = 1/peak
        return scipy.stats.rv_histogram((hist,self.grid))
    
    def pdf(self, times):
        r""" Calculates the probability for each time
            
        args:
        times: A numpy list of times to evaluate
        """
        return self.dist.pdf(times-self.offset)
    
    def logpdf(self, times):
        r""" Calculates the log(probability) for each time
            
        args:
            times: A numpy list of times to evaluate
        """
        return self.dist.logpdf(times-self.offset)
        
    def random(self, n=1): 
        r""" Return random values following the custom distribution
        
        args:
            n: The number of random values to return
        """
        return self.dist.rvs(size=n) + self.offset
    
    def effective_exposure(self): 
        r""" Calculate the weight associated with each
            event time. 
        """
        return 1.0/self.norm
    
    def get_range(self): 
        r""" Return the min/max values for the distribution 
        """
        return [self.range[0], self.range[1]]
    
    def change_offset(self, offset):
        r"""Change the offset of gaussian(add the offset to mean"""
        self.offset = offset
        self.range = [ self.range[0] + offset, self.range[1] + offset ]
        return
        
        
