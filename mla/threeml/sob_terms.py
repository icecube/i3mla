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
from scipy.interpolate import UnivariateSpline as Spline

from .. import sob_terms
from .. import sources
from .. import params as par
from . import spectral
from . import data_handlers


@dataclasses.dataclass
class ThreeMLPSEnergyTerm(sob_terms.SoBTerm):
    """Docstring"""
    _energysobhist: np.ndarray
    _sin_dec_idx: np.ndarray
    _log_energy_idx: np.ndarray

    def update_sob_hist(
        self,
        factory: sob_terms.SoBTermFactory,
        spectrum: spectral.BaseSpectrum
    ) -> None:
        """Docstring"""
        self._energysobhist = factory.update_spectrum(spectrum)

    @property
    def params(self) -> par.Params:
        """Docstring"""
        return self._params

    @params.setter
    def params(self, params: par.Params) -> None:
        """Docstring"""
        self._params = params

    @property
    def sob(self) -> np.ndarray:
        """Docstring"""
        return self._energysobhist[self._sin_dec_idx, self._log_energy_idx]


@dataclasses.dataclass
class ThreeMLPSEnergyTermFactory(sob_terms.SoBTermFactory):
    """Docstring"""
    data_handler: data_handlers.ThreeMLDataHandler
    source: sources.PointSource
    _spectrum: spectral.BaseSpectrum = dataclasses.field(init=False, repr=False)
    _bg_sob: np.ndarray = dataclasses.field(init=False, repr=False)
    _reduced_reco_sim: np.ndarray = dataclasses.field(init=False, repr=False)
    _sin_dec_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _log_energy_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _bins: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Docstring"""
        self._init_bg_sob_map()
        self.spectrum = self.config['assumed_spectrum']
        self._sin_dec_bins = np.linspace(-1, 1, 1 + self.config['sin_dec_bins'])
        self._log_energy_bins = np.linspace(
            *self.config['log_energy_bounds'], 1 + self.config['log_energy_bins'])
        self._reduced_reco_sim = self.data_handler.cut_reconstructed_sim(
            self.source.location[1], *self.config['reco_sampling_width'])
        self._bins = np.array([self._sin_dec_bins, self._log_energy_bins])

    def __call__(
        self,
        params: par.Params,
        events: np.ndarray
    ) -> sob_terms.SoBTerm:
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
            _params=params,
            _sob=np.empty(1),
            _sin_dec_idx=sin_dec_idx,
            _log_energy_idx=log_energy_idx,
            _energysobhist=self.cal_sob_map(),
        )

    def _init_bg_sob_map(self) -> np.ndarray:
        """Docstring"""
        bg_h = self.data_handler.build_background_sindec_logenergy_histogram(
            self._bins
        )
        # Normalize histogram by dec band
        bg_h /= np.sum(bg_h, axis=1)[:, None]
        self._bg_sob = bg_h

    def update_source(
        self,
        source: sources.PointSource
    ) -> None:
        """Docstring"""
        self.source = source
        self._reduced_reco_sim = self.data_handler.cut_reconstructed_sim(
            self.source.location[1], *self.config['reco_sampling_width'])

    def update_spectrum(
        self,
        spectrum: spectral.BaseSpectrum
    ) -> np.ndarray:
        """Docstring"""
        self.spectrum = spectrum
        return self.cal_sob_map()

    def cal_sob_map(
        self
    ) -> np.ndarray:
        """Creates sob histogram for a given spectrum.

        Returns:
            An array of signal-over-background values binned in sin(dec) and
            log(energy) for a given gamma.
        """
        sig_h = self.data_handler.build_signal_energy_histogram(
            self._reduced_reco_sim,
            self.spectrum,
            self._bins
        )
        bin_centers = self._bins[:-1] + np.diff(self._bins) / 2
        # Normalize histogram by dec band
        sig_h /= np.sum(sig_h, axis=1)[:, None]

        # div-0 okay here
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = sig_h / self._bg_sob

        for i in range(ratio.shape[0]):
            # Pick out the values we want to use.
            # We explicitly want to avoid NaNs and infinities
            good = np.isfinite(ratio[i]) & (ratio[i] > 0)
            good_bins, good_vals = bin_centers[good], ratio[i][good]

            # Do a linear interpolation across the energy range
            spline = Spline(
                good_bins,
                good_vals,
                k=self.config['energy_spline_k'],
                s=self.config['energy_spline_s'],
                ext=self.config['energy_spline_ext'],
            )

            # And store the interpolated values
            ratio[i] = spline(bin_centers)
        return ratio

    def calculate_drop_mask(self, events: np.ndarray) -> np.ndarray:
        """Docstring"""
        return np.ones(len(events), dtype=bool)

    @property
    def spectrum(self) -> spectral.BaseSpectrum:
        """Docstring"""
        return self._spectrum

    @spectrum.setter
    def spectrum(self, spectrum: spectral.BaseSpectrum) -> None:
        """Docstring"""
        self._spectrum = spectrum

    @classmethod
    def generate_config(cls):
        """Docstring"""
        config = super().generate_config()
        config['assumed_spectrum'] = spectral.PowerLaw(1e3, 1e-14, -2)
        config['reco_sampling_width'] = np.deg2rad(3)
        return config
