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
    def effective_exposure(self, times: np.array) -> float: pass

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