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

import abc

class GenericProfile:
    """A generic base class to standardize the methods for the
    time profiles. While I'm only currently using scipy-based
    probability distributions, you can write your own if you
    want. Just be sure to define these methods and ensure that
    the PDF is normalized!
    """

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
    def x0(self, times): pass
    
    @abc.abstractmethod
    def bounds(self, time_profile): pass

    @property
    @abc.abstractmethod
    def default_params(self): pass
    
    @property
    @abc.abstractmethod
    def param_dtype(self): pass