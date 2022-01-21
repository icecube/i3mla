"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Tuple

import abc
import copy
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import numpy.lib.recfunctions as rf
from scipy.interpolate import UnivariateSpline as Spline

from .. import configurable
from .. import data_handlers
from . import spectral


@dataclass
class ThreeMLDataHandler(NuSourcesDataHandler):
    """Docstring"""
    spectrum:spectral.BaseSpectrum

    _spectrum: np.spectral.BaseSpectrum = field(
        init=False, repr=False, default=spectral.PowerLaw(1e3, 1e-14, -2))

    def build_signal_energy_histogram(
        self, reduce_sim:np.ndarray , bins: np.ndarray) -> np.ndarray:
        """Docstring"""
        return np.histogram2d(
            reduce_sim['sindec'],
            reduce_sim['logE'],
            bins=bins,
            weights = reduce_sim['ow'] * self._spectrum(reduce_sim['trueE']),
            density=True,
        )[0]

    @property
    def spectrum(self) -> spectral.BaseSpectrum:
        """Docstring"""
        return self._spectrum
    
    @spectrum.setter
    def spectrum(self, spectrum:spectral.BaseSpectrum) -> None:
        """Docstring"""
        self._spectrum = spectrum