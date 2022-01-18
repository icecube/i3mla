"""
The classes in this file do preprocessing on data and monte carlo to be used
to do a point source analysis.
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List, Optional, Tuple, Union

from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar

import numpy as np
from scipy.interpolate import UnivariateSpline as Spline


@dataclass
class EnergyModelBase:
    """Docstring"""
    config: dict

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        return {}
    

@dataclass
class SplineMapEnergyModel(EnergyModelBase):
    """Docstring"""
    data: InitVar[np.ndarray]
    sim: InitVar[np.ndarray]
    _spline_map: List[List[Spline]] = field(init=False)

    @classmethod
    def generate_config(cls):
        """Docstring"""
        config = super().generate_config()
        config['sin_dec_bins'] = 50
        config['log_energy_bins'] = 50
        config['log_energy_bounds'] = (1, 8)
        config['gamma_bins'] = 50
        config['gamma_bounds'] = (-4.25, -0.5)
        config['sob_spline_k'] = 3
        config['sob_spline_s'] = 0
        config['sob_spline_ext'] = 'raise'
        config['energy_spline_k'] = 1
        config['energy_spline_s'] = 0
        config['energy_spline_ext'] = 3
        return config


    def __post_init__(self, data: np.ndarray, sim: np.ndarray) -> None:
        """Docstring"""
        if isinstance(self.config['sin_dec_bins'], int):
            self.config['sin_dec_bins'] = np.linspace(
                    -1, 1, 1 + self.config['sin_dec_bins'])

        if isinstance(self.config['log_energy_bins'], int):
            self.config['log_energy_bins'] = np.linspace(
                    *self.config['log_energy_bounds'], 1 + self.config['log_energy_bins'])

        if isinstance(self.config['gamma_bins'], int):
            gamma_bins = np.linspace(
                    *self.config['gamma_bounds'], 1 + self.config['gamma_bins'])

        self._log_sob_gamma_splines = self._init_log_sob_gamma_splines()

    def _init_sob_map(self, gamma: float) -> np.array:
        """Creates sob histogram for a given spectral index (gamma).

        Args:
            gamma: The gamma value to use to weight the signal.

        Returns:
            An array of signal-over-background values binned in sin(dec) and
            log(energy) for a given gamma.
        """
        bins = np.array([self.config['sin_dec_bins'], self.config['log_energy_bins']])
        bin_centers = bins[1, :-1] + np.diff(bins[1]) / 2

        # background
        bg_h, _, _ = np.histogram2d(
            self.data['sindec'],
            self.data['logE'],
            bins=bins,
            density=True,
        )

        # signal
        sig_w = self.sim['ow'] * self.sim['trueE']**gamma
        sig_h, _, _ = np.histogram2d(
            self.sim['sindec'],
            self.sim['logE'],
            bins=bins,
            weights=sig_w,
            density=True,
        )

        # Normalize histograms by dec band
        bg_h /= np.sum(bg_h, axis=1)[:, None]
        sig_h /= np.sum(sig_h, axis=1)[:, None]

        # div-0 okay here
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = sig_h / bg_h

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

    def _init_log_sob_gamma_splines(self) -> List[List[Spline]]:
        """Builds a 3D hist of sob vs. sin(dec), log(energy), and gamma, then
            returns splines of sob vs. gamma.

        Returns: A Nested spline list of shape (sin_dec_bins, log_energy_bins).
        """
        sob_maps = np.array([
            self._init_sob_map(gamma) for gamma in self.config['gamma_bins']])

        transposed_log_sob_maps = np.log(sob_maps.transpose(1, 2, 0))

        splines = [[
            Spline(
                self.config['gamma_bins'],
                log_ratios,
                k=self.config['sob_spline_k'],
                s=self.config['sob_spline_s'],
                ext=self.config['sob_spline_ext'],
            )
            for log_ratios in dec_bin
        ] for dec_bin in transposed_log_sob_maps]

        return splines

    def build_event_map(self, events: np.ndarray) -> dict:
        """Docstring"""
        # Get the bin that each event belongs to
        sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1],
                                      events['sindec'])

        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1],
                                         events['logE'])

        spline_idxs, event_spline_idxs = np.unique(
            [sin_dec_idx - 1, log_energy_idx - 1],
            return_inverse=True,
            axis=1
        )

        splines = [
            self._log_sob_gamma_splines[i][j]
            for i, j in spline_idxs.T
        ]

        return {
            'splines': splines,
            'event_spline_idxs': np.array(event_spline_idxs, dtype=int),
        }

    @staticmethod
    def evaluate_sob(
        gamma: float,
        splines: List[Spline],
        event_spline_idxs: np.ndarray,
    ) -> np.array:
        """Docstring"""
        spline_evals = np.exp([spline(gamma) for spline in splines])
        return spline_evals[event_spline_idxs]
