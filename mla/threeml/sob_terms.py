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
from scipy.interpolate import UnivariateSpline as Spline
from .. import sob_terms
from .. import sources
from .. import params as par
from . import spectral
from . import data_handlers

PSTrackv4_sin_dec_bin = np.unique(
    np.concatenate(
        [
            np.linspace(-1, -0.93, 4 + 1),
            np.linspace(-0.93, -0.3, 10 + 1),
            np.linspace(-0.3, 0.05, 9 + 1),
            np.linspace(0.05, 1, 18 + 1),
        ]
    )
)

PSTrackv4_log_energy_bins = np.arange(1, 9.5 + 0.01, 0.125)


@dataclasses.dataclass
class ThreeMLPSEnergyTerm(sob_terms.SoBTerm):
    """
    Energy term for 3ML. Constructs only from 3ML energy term factory
    """

    _energysobhist: np.ndarray
    _sin_dec_idx: np.ndarray
    _log_energy_idx: np.ndarray

    def update_sob_hist(self, factory: sob_terms.SoBTermFactory) -> None:
        """
        Updating the signal-over-background energy histogram.

        Args:
            factory: energy term factory
        """
        self._energysobhist = factory.cal_sob_map()

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
class ThreeMLBaseEnergyTermFactory(sob_terms.SoBTermFactory):
    """Docstring"""

    pass


@dataclasses.dataclass
class ThreeMLPSEnergyTermFactory(ThreeMLBaseEnergyTermFactory):
    """
    This is the class for using MC directly to build the Energy terms.
    We sugguest using the IRF for Energy term factory due to speed.

    Args:
        data_handler: 3ML data handler
        source: 3ML source object
        spectrum: signal spectrum
    """

    data_handler: data_handlers.ThreeMLDataHandler
    source: sources.PointSource
    spectrum: spectral.BaseSpectrum
    _source: sources.PointSource = dataclasses.field(init=False, repr=False)
    _spectrum: spectral.BaseSpectrum = dataclasses.field(
        init=False, repr=False, default=spectral.PowerLaw(1e3, 1e-14, -2)
    )
    _bg_sob: np.ndarray = dataclasses.field(init=False, repr=False)
    _sin_dec_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _log_energy_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _ow_hist: np.ndarray = dataclasses.field(init=False, repr=False)
    _ow_ebin: np.ndarray = dataclasses.field(init=False, repr=False)
    _unit_scale: float = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Docstring"""
        if self.config["list_sin_dec_bins"] is None:
            self._sin_dec_bins = np.linspace(-1, 1, 1 + self.config["sin_dec_bins"])
        else:
            self._sin_dec_bins = self.config["list_sin_dec_bins"]
        if self.config["list_log_energy_bins"] is None:
            self._log_energy_bins = np.linspace(
                *self.config["log_energy_bounds"], 1 + self.config["log_energy_bins"]
            )
        else:
            self._log_energy_bins = self.config["list_log_energy_bins"]
        self.data_handler.reduced_reco_sim = self.data_handler.cut_reconstructed_sim(
            self.source.location[1],
            self.data_handler.config["reco_sampling_width"],
        )
        self._unit_scale = self.config["Energy_convesion(ToGeV)"]
        self._bins = np.array([self._sin_dec_bins, self._log_energy_bins])
        self._init_bg_sob_map()
        self._build_ow_hist()

    def __call__(self, params: par.Params, events: np.ndarray) -> sob_terms.SoBTerm:
        """Docstring"""
        # Get the bin that each event belongs to
        sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1], events["sindec"]) - 1

        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1], events["logE"]) - 1

        return ThreeMLPSEnergyTerm(
            name=self.config["name"],
            _params=params,
            _sob=np.empty(1),
            _sin_dec_idx=sin_dec_idx,
            _log_energy_idx=log_energy_idx,
            _energysobhist=self.cal_sob_map(),
        )

    def _build_ow_hist(self) -> np.ndarray:
        """Docstring"""
        self._ow_hist, self._ow_ebin = np.histogram(
            np.log10(self.data_handler.sim["trueE"]),
            bins=200,
            weights=self.data_handler.sim["ow"],
        )
        self._ow_ebin = 10**self._ow_ebin[:-1] * self._unit_scale

    def get_ns(self) -> float:
        """Docstring"""
        return (self.spectrum(self._ow_ebin) * self._ow_hist).sum() * self._unit_scale

    def _init_bg_sob_map(self) -> np.ndarray:
        """Docstring"""
        if self.config["mc_bkgweight"] is None:
            bg_h = self.data_handler.build_background_sindec_logenergy_histogram(
                self._bins
            )
        else:
            bg_h = self.data_handler.build_mcbackground_sindec_logenergy_histogram(
                self._bins, self.config["mc_bkgweight"]
            )
            print("using mc background")
        # Normalize histogram by dec band
        bg_h /= np.sum(bg_h, axis=1)[:, None]
        if self.config["backgroundSOBoption"] == 1:
            bg_h[bg_h <= 0] = np.min(bg_h[bg_h > 0])
        elif self.config["backgroundSOBoption"] == 0:
            pass
        self._bg_sob = bg_h

    @property
    def source(self) -> sources.PointSource:
        """Docstring"""
        return self._source

    @source.setter
    def source(self, source: sources.PointSource) -> None:
        """Docstring"""
        self._source = source
        self.data_handler.reduced_reco_sim = self.data_handler.cut_reconstructed_sim(
            self.source.location[1],
            self.data_handler.config["reco_sampling_width"],
        )

    def cal_sob_map(self) -> np.ndarray:
        """Creates sob histogram for a given spectrum.
        Returns:
            An array of signal-over-background values binned in sin(dec) and
            log(energy) for a given gamma.
        """
        sig_h = self.data_handler.build_signal_energy_histogram(
            self.spectrum, self._bins, self._unit_scale
        )
        bin_centers = self._log_energy_bins[:-1] + np.diff(self._log_energy_bins) / 2
        # Normalize histogram by dec band
        sig_h /= np.sum(sig_h, axis=1)[:, None]

        # div-0 okay here
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = sig_h / self._bg_sob

        for i in range(ratio.shape[0]):
            # Pick out the values we want to use.
            # We explicitly want to avoid NaNs and infinities
            good = np.isfinite(ratio[i]) & (ratio[i] > 0)
            good_bins, good_vals = bin_centers[good], ratio[i][good]
            if len(good_bins) > 1:
                # Do a linear interpolation across the energy range
                spline = Spline(
                    good_bins, good_vals, **self.config["energy_spline_kwargs"]
                )

                # And store the interpolated values
                ratio[i] = spline(bin_centers)
            elif len(good_bins) == 1:
                ratio[i] = good_vals
            else:
                ratio[i] = 0
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
        if isinstance(spectrum, property):
            # initial value not specified, use default
            spectrum = ThreeMLPSEnergyTermFactory._spectrum
        self._spectrum = spectrum

    @classmethod
    def generate_config(cls):
        """Docstring"""
        config = super().generate_config()
        config["sin_dec_bins"] = 68
        config["log_energy_bins"] = 42
        config["log_energy_bounds"] = (1, 8)
        config["energy_spline_kwargs"] = {
            "k": 1,
            "s": 0,
            "ext": 3,
        }
        config["backgroundSOBoption"] = 0
        config["mc_bkgweight"] = None
        config["list_sin_dec_bins"] = PSTrackv4_sin_dec_bin
        config["list_log_energy_bins"] = PSTrackv4_log_energy_bins
        return config


@dataclasses.dataclass
class ThreeMLPSIRFEnergyTermFactory(ThreeMLPSEnergyTermFactory):
    """Docstring"""

    data_handler: data_handlers.ThreeMLDataHandler
    source: sources.PointSource
    spectrum: spectral.BaseSpectrum
    _source: sources.PointSource = dataclasses.field(init=False, repr=False)
    _spectrum: spectral.BaseSpectrum = dataclasses.field(
        init=False, repr=False, default=spectral.PowerLaw(1e3, 1e-14, -2)
    )
    _bg_sob: np.ndarray = dataclasses.field(init=False, repr=False)
    _sin_dec_bins: np.ndarray = dataclasses.field(
        init=False, repr=False, default=PSTrackv4_sin_dec_bin
    )
    _log_energy_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _trueebin: np.ndarray = dataclasses.field(init=False, repr=False)
    _irf: np.ndarray = dataclasses.field(init=False, repr=False)
    _sindec_bounds: np.ndarray = dataclasses.field(init=False, repr=False)
    _ntrueebin: int = dataclasses.field(init=False, repr=False)
    _unit_scale: float = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Docstring"""
        if self.config["list_sin_dec_bins"] is None:
            self._sin_dec_bins = np.linspace(-1, 1, 1 + self.config["sin_dec_bins"])
        else:
            self._sin_dec_bins = self.config["list_sin_dec_bins"]
        if self.config["list_log_energy_bins"] is None:
            self._log_energy_bins = np.linspace(
                *self.config["log_energy_bounds"], 1 + self.config["log_energy_bins"]
            )
        else:
            self._log_energy_bins = self.config["list_log_energy_bins"]
        lower_sindec = np.maximum(
            np.sin(
                self.source.location[1]
                - self.data_handler.config["reco_sampling_width"]
            ),
            -0.99,
        )
        upper_sindec = np.minimum(
            np.sin(
                self.source.location[1]
                + self.data_handler.config["reco_sampling_width"]
            ),
            1,
        )
        lower_sindec_index = np.searchsorted(self._sin_dec_bins, lower_sindec) - 1
        uppper_sindec_index = np.searchsorted(self._sin_dec_bins, upper_sindec)
        self._sindec_bounds = np.array([lower_sindec_index, uppper_sindec_index])
        self._bins = np.array([self._sin_dec_bins, self._log_energy_bins])
        self._truelogebin = self.config["list_truelogebin"]
        self._unit_scale = self.config["Energy_convesion(ToGeV)"]
        self._init_bg_sob_map()
        self._build_ow_hist()
        self._init_irf()

    def __call__(self, params: par.Params, events: np.ndarray) -> sob_terms.SoBTerm:
        """Docstring"""
        # Get the bin that each event belongs to
        sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1], events["sindec"]) - 1

        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1], events["logE"]) - 1

        return ThreeMLPSEnergyTerm(
            name=self.config["name"],
            _params=params,
            _sob=np.empty(1),
            _sin_dec_idx=sin_dec_idx,
            _log_energy_idx=log_energy_idx,
            _energysobhist=self.cal_sob_map(),
        )

    def _init_bg_sob_map(self) -> None:
        """Docstring"""
        if self.config["mc_bkgweight"] is None:
            bg_h = self.data_handler.build_background_sindec_logenergy_histogram(
                self._bins
            )
        else:
            bg_h = self.data_handler.build_mcbackground_sindec_logenergy_histogram(
                self._bins, self.config["mc_bkgweight"]
            )
            print("using mc background")
        # Normalize histogram by dec band
        bg_h /= np.sum(bg_h, axis=1)[:, None]
        if self.config["backgroundSOBoption"] == 1:
            bg_h[bg_h <= 0] = np.min(bg_h[bg_h > 0])
        elif self.config["backgroundSOBoption"] == 0:
            pass
        self._bg_sob = bg_h

    def _init_irf(self) -> None:
        """Docstring"""
        self._irf = np.zeros(
            (
                len(self._sin_dec_bins) - 1,
                len(self._log_energy_bins) - 1,
                len(self._truelogebin) - 1,
            )
        )
        self._trueebin = 10 ** (self._truelogebin[:-1])
        sindec_idx = (
            np.digitize(np.sin(self.data_handler.full_sim["dec"]), self._sin_dec_bins)
            - 1
        )

        for i in range(len(self._sin_dec_bins) - 1):
            events_dec = self.data_handler.full_sim[(sindec_idx == i)]
            loge_idx = np.digitize(events_dec["logE"], self._log_energy_bins) - 1

            for j in range(len(self._log_energy_bins) - 1):
                events = events_dec[(loge_idx == j)]

                # Don't bother if we don't find events.
                if events["ow"].sum() == 0:
                    continue

                # True bins are in log(trueE) to ensure they're well spaced.
                self._irf[i, j], _ = np.histogram(
                    np.log10(events["trueE"]),
                    bins=self._truelogebin,
                    weights=events["ow"],
                )

                # Have to pick an "energy" to assign to the bin. That's complicated, since
                # you'd (in principle) want the flux-weighted average energy, but we don't
                # have a flux function here. Instead, try just using the minimum energy of
                # the bin? Should be fine for small enough bins.
                # self._trueebin[i,j] = np.exp(bins[:-1] + (bins[1] - bins[0]))
                # emean[i,j] = np.average(events['trueE'], weights=events['ow'])

    def build_sig_h(self, spectrum: spectral.BaseSpectrum) -> np.ndarray:
        """Docstring"""
        sig = np.zeros(self._bg_sob.shape)
        flux = spectrum(self._trueebin * self._unit_scale)  # converting unit
        sig[self._sindec_bounds[0]:self._sindec_bounds[1], :] = np.dot(
            self._irf[self._sindec_bounds[0]:self._sindec_bounds[1], :, :], flux
        )
        sig /= np.sum(sig, axis=1)[:, None]
        return sig

    @property
    def source(self) -> sources.PointSource:
        """Docstring"""
        return self._source

    @source.setter
    def source(self, source: sources.PointSource) -> None:
        """Docstring"""
        self._source = source
        lower_sindec = np.maximum(
            np.sin(
                self.source.location[1]
                - self.data_handler.config["reco_sampling_width"]
            ),
            -0.99,
        )
        upper_sindec = np.minimum(
            np.sin(
                self.source.location[1]
                + self.data_handler.config["reco_sampling_width"]
            ),
            1,
        )
        lower_sindec_index = np.searchsorted(self._sin_dec_bins, lower_sindec) - 1
        uppper_sindec_index = np.searchsorted(self._sin_dec_bins, upper_sindec)
        self._sindec_bounds = np.array([lower_sindec_index, uppper_sindec_index])

    def cal_sob_map(self) -> np.ndarray:
        """Creates sob histogram for a given spectrum.

        Returns:
            An array of signal-over-background values binned in sin(dec) and
            log(energy) for a given gamma.
        """
        sig_h = self.build_sig_h(self.spectrum)

        bin_spline = self._log_energy_bins[:-1]
        # Normalize histogram by dec band

        # div-0 okay here
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = sig_h / self._bg_sob

        for i in range(ratio.shape[0]):
            # Pick out the values we want to use.
            # We explicitly want to avoid NaNs and infinities
            good = np.isfinite(ratio[i]) & (ratio[i] > 0)
            good_bins, good_vals = bin_spline[good], ratio[i][good]
            if len(good_bins) > 1:
                # Do a linear interpolation across the energy range
                spline = Spline(
                    good_bins, good_vals, **self.config["energy_spline_kwargs"]
                )

                # And store the interpolated values
                ratio[i] = spline(bin_spline)
            elif len(good_bins) == 1:
                ratio[i] = good_vals
            else:
                ratio[i] = 0

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
        if isinstance(spectrum, property):
            # initial value not specified, use default
            spectrum = ThreeMLPSEnergyTermFactory._spectrum
        self._spectrum = spectrum

    @classmethod
    def generate_config(cls):
        """Docstring"""
        config = super().generate_config()
        config["sin_dec_bins"] = 68
        config["log_energy_bins"] = 42
        config["log_energy_bounds"] = (1, 8)
        config["energy_spline_kwargs"] = {
            "k": 1,
            "s": 0,
            "ext": 3,
        }
        config["backgroundSOBoption"] = 0
        config["mc_bkgweight"] = None
        config["list_sin_dec_bins"] = PSTrackv4_sin_dec_bin
        config["list_log_energy_bins"] = PSTrackv4_log_energy_bins
        config["list_truelogebin"] = np.arange(
            2, 9.01 + 0.01, 0.01
        )
        config["Energy_convesion(ToGeV)"] = 1e6  # GeV to keV
        return config
