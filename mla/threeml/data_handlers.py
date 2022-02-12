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
import numpy.lib.recfunctions as rf

from .. import data_handlers
from . import spectral


@dataclasses.dataclass
class ThreeMLDataHandler(data_handlers.NuSourcesDataHandler):
    """Docstring"""
    injection_spectrum: spectral.BaseSpectrum = spectral.PowerLaw(
        1e3, 1e-14, -2)
    _reduced_reco_sim: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Docstring"""
        self._reduced_reco_sim = self.cut_reconstructed_sim(
            self.dec_cut_location, self.config['reco_sampling_width'])

    def build_signal_energy_histogram(
        self,
        spectrum: spectral.BaseSpectrum,
        bins: np.ndarray
    ) -> np.ndarray:
        """Docstring"""
        return np.histogram2d(
            self.reduced_reco_sim['sindec'],
            self.reduced_reco_sim['logE'],
            bins=bins,
            weights=self.reduced_reco_sim['ow'] * spectrum(
                self.reduced_reco_sim['trueE']
            ),
            density=True,
        )[0]

    def cut_reconstructed_sim(
        self,
        dec: float,
        sampling_width: float
    ) -> np.ndarray:
        """Docstring"""
        dec_dist = np.abs(
            dec - self._full_sim['dec'])
        close = dec_dist < sampling_width
        return self._full_sim[close].copy()

    @property
    def reduced_reco_sim(self) -> np.ndarray:
        """Docstring"""
        return self._reduced_reco_sim

    @reduced_reco_sim.setter
    def reduced_reco_sim(
        self,
        reduced_reco_sim: np.ndarray
    ) -> None:
        """Docstring"""
        self._reduced_reco_sim = reduced_reco_sim.copy()

    @injection_spectrum.setter
    def injection_spectrum(self, injection_spectrum: spectral.BaseSpectrum):
        """Docstring"""
        self.injection_spectrum = injection_spectrum
        if 'weight' not in self._full_sim.dtype.names:
            self._full_sim = rf.append_fields(
                self._full_sim, 'weight',
                np.zeros(len(self._full_sim)),
                dtypes=np.float32
            )

        self._full_sim['weight'] = self._full_sim['ow'] * (
            self.injection_spectrum(self._full_sim['trueE'])
        )

        self._cut_sim_dec()

    @property
    def sim(self) -> np.ndarray:
        """Docstring"""
        return self._sim

    @sim.setter
    def sim(self, sim: np.ndarray) -> None:
        """Docstring"""
        self._full_sim = sim.copy()

        if 'sindec' not in self._full_sim.dtype.names:
            self._full_sim = rf.append_fields(
                self._full_sim,
                'sindec',
                np.sin(self._full_sim['dec']),
                usemask=False,
            )

    @classmethod
    def generate_config(cls):
        """Docstring"""
        config = super().generate_config()
        config['reco_sampling_width'] = np.deg2rad(3)
        return config
