"""
The classes in this file are example time profiles that can be used in the
analysis classes. There is also GenericProfile, an abstract parent class to
inherit from to create other time profiles.
"""


__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'


import numpy as np
import abc
from typing import Callable, Dict, List, Optional, Tuple, Union


class BaseSpectrum(object):
    """A generic base class to standardize the methods for the Spectrum.

    Any callable function will work.

    Attributes:
    """

    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def __init__(self)-> None:
        """Initializes the Spectrum.
        """
        pass

    @abc.abstractmethod
    def __call__(self, E:Union[np.ndarray,float], **kwargs) -> np.ndarray:
        """return the differential flux at given energy(s).

        Args:
            E: An array of Energy

        Returns:
            A numpy array of differential flux.
        """
        pass
    
    @abc.abstractmethod
    def __str__(self) -> None:
        """String representation
        """
        return "Base spectrum"

      
      
class PowerLaw(BaseSpectrum):
    """Spectrum class for PowerLaw.

    Use this to produce PowerLaw spectrum.

    Attributes:
        E0 (float): pivot energy
        A (float): Flux Norm
        gamma(float): Spectral index
        Ecut(float): Cut-off energy
    """
    
    def __init__(self, E0:float, A:float, gamma:float, Ecut:float = None) -> None:
        """ Constructor of PowerLaw object.
        
        Args:
            E0: Normalize Energy
            A: Flux Normalization
            gamma: Spectral index
            Ecut(optional): Cut-off energy
        """
        self.E0 = E0
        self.A = A
        self.gamma = gamma
        self.Ecut = Ecut
        return
    
    def __call__(self, E:Union[np.ndarray,float], **kwargs) -> np.ndarray:
        """Evaluate spectrum at energy E according to

                 dN/dE = A (E / E0)^gamma

        where A has units of events / (GeV cm^2 s). We treat
        the 'events' in the numerator as implicit and say the
        units are [GeV^-1 cm^-2 s^-1]. Specifying Ecut provides
        an optional spectral cutoff according to

                 dN/dE = A (E / E0)^gamma * exp( -E/Ecut )

        Args:
            E : Evaluation energy [GeV]
        
        Returns:
            np.ndarray of differential flux
        """
        A = kwargs.pop("A", self.A)
        E0 = kwargs.pop("E0", self.E0)
        Ecut = kwargs.pop("Ecut", self.Ecut)
        gamma = kwargs.pop("gamma", self.gamma)

        flux = A * (E / E0)**(gamma)

        # apply optional exponential cutoff
        if Ecut is not None:
            flux *= np.exp(-E / self.Ecut)

        return flux
        
    def __str__(self) -> None:
        """String representation
        """
        return "PowerLaw"
    
    
class CustomSpectrum(BaseSpectrum):
    '''Custom spectrum using callable object
    '''
    def __init__(self,spectrum):
        """Constructor of CustomSpectrum object.
        
        Constructor
        
        Args:
            spectrum: Any callable object
        
        """
        self.spectrum = spectrum
        return
        
    def __call__(self, E:Union[np.ndarray,float])-> np.ndarray:
        """Evaluate spectrum at energy E
        
        Constructor
        
        Args:
            E : Evaluation energy 
        
        Returns:
            np.ndarray of differential flux 
        """
        return self.spectrum(E)
        
    def __str__(self):
        r"""String representation of class"""
        return "CustomSpectrum"