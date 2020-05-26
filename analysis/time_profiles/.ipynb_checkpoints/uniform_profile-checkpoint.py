import numpy as np
import scipy
from .generic_profile import generic_profile

class uniform_profile(generic_profile):
    '''Time profile class for a uniform distribution. Use this
    for background or if you want to assume a steady signal from
    your source.
    '''

    def __init__(self, start_time, end_time):
        '''Constructor for the class.

        args:
            start_time, end_time: The bounds for the uniform
                            distribution.
        '''
        assert(end_time > start_time)
        self.start_time = start_time
        self.end_time = end_time
        self.norm = 1.0/(end_time-start_time)
        return

    def pdf(self, times):
        '''Calculates the probability for each time

        args:
            times: A numpy list of times to evaluate
        '''
        output = np.zeros_like(times)
        output[(times>=self.start_time) &\
               (times<self.end_time)] = self.norm
        return output

    def logpdf(self, times):
        '''Calculates the log(probability) for each time

        args:
            times: A numpy list of times to evaluate
        '''
        return np.log(self.pdf(times))

    def random(self, n=1):
        '''Return random values following the uniform distribution

        args:
            n: The number of random values to return
        '''
        return np.random.uniform(self.start_time,
                                 self.end_time,
                                 n)

    def effective_exposure(self):
        '''Calculate the weight associated with each event time'''
        return 1.0/self.norm

    def get_range(self):
        '''Return the min/max values for the distribution'''
        return [self.start_time, self.end_time]
