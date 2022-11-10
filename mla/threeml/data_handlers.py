"""Docstring"""

__author__ = "John Evans"
__copyright__ = "Copyright 2021 John Evans"
__credits__ = ["John Evans", "Jason Fan", "Michael Larson"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "John Evans"
__email__ = "john.evans@icecube.wisc.edu"
__status__ = "Development"

import dataclasses

import numpy as np
import numpy.lib.recfunctions as rf

from .. import data_handlers
from . import spectral


@dataclasses.dataclass
class ThreeMLDataHandler(data_handlers.NuSourcesDataHandler):
    """
    Inheritance class from NuSourcesDataHandler.
    For time independent 3ML analysis.

    Additional init argument:
        injection_spectrum: spectral.BaseSpectrum
    """

    injection_spectrum: spectral.BaseSpectrum
    _injection_spectrum: spectral.BaseSpectrum = dataclasses.field(
        init=False, repr=False, default=spectral.PowerLaw(1e3, 1e-14, -2)
    )
    _reduced_reco_sim: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Docstring"""
        self._reduced_reco_sim = self.cut_reconstructed_sim(
            self.config["dec_cut_location"], self.config["reco_sampling_width"]
        )

    def build_signal_energy_histogram(
        self, spectrum: spectral.BaseSpectrum, bins: np.ndarray, scale: float
    ) -> np.ndarray:
        """
        Building the signal energy histogram.
        Only used when using MC instead of IRF to build signal energy histogram.

        Args:
            spectrum: signal spectrum
            bins: 2d bins in sindec and logE
        """
        return np.histogram2d(
            self.reduced_reco_sim["sindec"],
            self.reduced_reco_sim["logE"],
            bins=bins,
            weights=self.reduced_reco_sim["ow"]
            * spectrum(self.reduced_reco_sim["trueE"] * scale),
            density=True,
        )[0]

    def cut_reconstructed_sim(self, dec: float, sampling_width: float) -> np.ndarray:
        """
        Cutting the MC based on reconstructed dec.
        Only use when using MC instead of IRF to build signal energy histogram.

        Args:
            dec: declination of the source
            sampling_width: size of the sampling band in reconstruction dec.
        """
        dec_dist = np.abs(dec - self._full_sim["dec"])
        close = dec_dist < sampling_width
        return self._full_sim[close].copy()

    @property
    def reduced_reco_sim(self) -> np.ndarray:
        """
        Return the reduced sim based on reconstructed dec.
        This is the return output of cut_reconstructed_sim.
        """
        return self._reduced_reco_sim

    @reduced_reco_sim.setter
    def reduced_reco_sim(self, reduced_reco_sim: np.ndarray) -> None:
        """
        setting the reduced sim based on reconstructed dec directly.

        Args:
            reduced_reco_sim: reduced sim based on reconstructed dec
        """
        self._reduced_reco_sim = reduced_reco_sim.copy()

    @property
    def injection_spectrum(self) -> spectral.BaseSpectrum:
        """
        Getting the injection spectrum
        """
        return self._injection_spectrum

    @injection_spectrum.setter
    def injection_spectrum(self, inject_spectrum: spectral.BaseSpectrum) -> None:
        """
        Setting the injection spectrum

        Args:
            inject_spectrum: spectrum used for injection
        """
        if isinstance(inject_spectrum, property):
            # initial value not specified, use default
            inject_spectrum = ThreeMLDataHandler._injection_spectrum
        self._injection_spectrum = inject_spectrum
        if "weight" not in self._full_sim.dtype.names:
            self._full_sim = rf.append_fields(
                self._full_sim,
                "weight",
                np.zeros(len(self._full_sim)),
                dtypes=np.float32,
            )

        self._full_sim["weight"] = self._full_sim["ow"] * (
            inject_spectrum(self._full_sim["trueE"])
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

        if "sindec" not in self._full_sim.dtype.names:
            self._full_sim = rf.append_fields(
                self._full_sim,
                "sindec",
                np.sin(self._full_sim["dec"]),
                usemask=False,
            )
        if "weight" not in self._full_sim.dtype.names:
            self._full_sim = rf.append_fields(
                self._full_sim,
                "weight",
                np.zeros(len(self._full_sim)),
                dtypes=np.float32,
            )

        self._cut_sim_dec()

    @classmethod
    def generate_config(cls):
        """Docstring"""
        config = super().generate_config()
        config["reco_sampling_width"] = np.deg2rad(5)
        return config


@dataclasses.dataclass
class ThreeMLTimeDepDataHandler(
    data_handlers.TimeDependentNuSourcesDataHandler, ThreeMLDataHandler
):
    """Docstring"""

    @classmethod
    def generate_config(cls):
        """Docstring"""
        config = super().generate_config()
        return config
