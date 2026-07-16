"""
The classes in this file are example time profiles that can be used in the
analysis classes. There is also GenericProfile, an abstract parent class to
inherit from to create other time profiles.
"""


__author__ = 'John Evans and Jason Fan'
__copyright__ = 'Copyright 2024'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '1.4.1'
__maintainer__ = 'Jason Fan'
__email__ = 'klfan@terpmail.umd.edu'
__status__ = 'Development'


from typing import Optional, Union

import abc
import numpy as np


class BaseSpectrum:
    """A generic base class to standardize the methods for the Spectrum.

    Any callable function will work.

    Attributes:
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self) -> None:
        """Initializes the Spectrum.
        """

    @abc.abstractmethod
    def __call__(self, energy: Union[np.ndarray, float],
                 **kwargs) -> np.ndarray:
        """return the differential flux at given energy(s).

        Args:
            energy: An array of energy

        Returns:
            A numpy array of differential flux.
        """

    @abc.abstractmethod
    def __str__(self) -> None:
        """String representation
        """
        return 'Base spectrum'


class PowerLaw(BaseSpectrum):
    """Spectrum class for PowerLaw.

    Use this to produce PowerLaw spectrum.

    Attributes:
        E0 (float): pivot energy
        A (float): Flux Norm
        gamma(float): Spectral index
        Ecut(float): Cut-off energy
    """

    def __init__(self, energy_0: float, flux_norm: float, gamma: float,
                 energy_cut: Optional[float] = None) -> None:
        """ Constructor of PowerLaw object.

        Args:
            energy_0: Normalize Energy
            flux_norm: Flux Normalization
            gamma: Spectral index
            energy_cut: Cut-off energy
        """

        super().__init__()
        self.energy_0 = energy_0
        self.flux_norm = flux_norm
        self.gamma = gamma
        self.energy_cut = energy_cut

    def __call__(self, energy: Union[np.ndarray, float],
                 **kwargs) -> np.ndarray:
        """Evaluate spectrum at energy E according to

                 dN/dE = A (E / E0)^gamma

        where A has units of events / (GeV cm^2 s). We treat
        the 'events' in the numerator as implicit and say the
        units are [GeV^-1 cm^-2 s^-1]. Specifying Ecut provides
        an optional spectral cutoff according to

                 dN/dE = A (E / E0)^gamma * exp( -E/Ecut )

        Args:
            energy : Evaluation energy [GeV]

        Returns:
            np.ndarray of differential flux
        """
        flux_norm = kwargs.pop('flux_norm', self.flux_norm)
        energy_0 = kwargs.pop('energy_0', self.energy_0)
        energy_cut = kwargs.pop('energy_cut', self.energy_cut)
        gamma = kwargs.pop('gamma', self.gamma)

        flux = flux_norm * (energy / energy_0)**(gamma)

        # apply optional exponential cutoff
        if energy_cut is not None:
            flux *= np.exp(-energy / self.energy_cut)

        return flux

    def __str__(self) -> None:
        """String representation
        """
        return 'PowerLaw'


class CustomSpectrum(BaseSpectrum):
    '''Custom spectrum using callable object
    '''
    def __init__(self, spectrum):
        """Constructor of CustomSpectrum object.

        Constructor

        Args:
            spectrum: Any callable object

        """

        super().__init__()
        self.spectrum = spectrum

    def __call__(self, energy: Union[np.ndarray, float],
                 **kwargs) -> np.ndarray:
        """Evaluate spectrum at energy E

        Constructor

        Args:
            energy : Evaluation energy

        Returns:
            np.ndarray of differential flux
        """
        return self.spectrum(energy)

    def __str__(self):
        r"""String representation of class"""
        return 'CustomSpectrum'
