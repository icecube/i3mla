"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

import dataclasses

import numpy as np

from .. import configurable
from .. import sob_terms
from .. import params
from . import spectral
from . import data_handlers
 
@dataclasses.dataclass
class ThreeMLPSEnergyTerm(sob_terms.SoBTerm):
    """Docstring"""
    _energysobhist: np.ndarray
    _sin_dec_idx: np.ndarray,
    _log_energy_idx: np.ndarray
    
    @property
    def params(self) -> params.Params:
        """Docstring"""
        return self._params

    @params.setter
    def params(self, params: params.Params) -> None:
        """Docstring"""
        self._params = params

    @property
    def sob(self) -> np.ndarray:
        """Docstring"""
        return self._energysobhist[_sin_dec_idx, _log_energy_idx]


@dataclasses.dataclass
class ThreeMLPSEnergyTermFactory(sob_terms.SoBTermFactory):
    """Docstring"""
    data_handler: data_handlers.ThreeMLDataHandler
    _reduced_reco_sim: np.ndarray = dataclasses.field(init=False, repr=False)
    _sin_dec_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _log_energy_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _energysobhist: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Docstring"""
        self._sin_dec_bins = np.linspace(-1, 1, 1 + self.config['sin_dec_bins'])
        self._log_energy_bins = np.linspace(
            *self.config['log_energy_bounds'], 1 + self.config['log_energy_bins'])
        self._reduced_reco_sim = self.data_handler.cut_reconstructed_sim(
            config['PointSource']['dec'], *self.config['reco_sampling_width'])
        self._energysobhist = self.data_handler._init_sob_map()
    
    def __call__(self, params: Params, events: np.ndarray) -> SoBTerm:
        """Docstring"""
        sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1], events['sindec'])
        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1], events['logE'])

        # Get the bin that each event belongs to
        try:
            sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1],
                                          events['sindec']) - 1
        except ValueError:
            sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1],
                                          np.sin(events['dec'])) - 1

        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1],
                                         events['logE']) - 1

        return ThreeMLPSEnergyTerm(
            _params = params,
            _sob = np.empty(1),
            _sin_dec_idx = sin_dec_idx,
            _log_energy_idx = log_energy_idx,
            _energysobhist = self._energysobhist,
        )


    