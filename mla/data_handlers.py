"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import ClassVar, Optional, Tuple

import abc
import copy
from dataclasses import InitVar, dataclass
from dataclasses import field

import numpy as np
from scipy.interpolate import UnivariateSpline as Spline

from .configurable import Configurable
from .time_profiles import GenericProfile
from .events import Events, SimEvents


@dataclass(kw_only=True)
class Injector(metaclass=abc.ABCMeta):
    """Docstring"""
    n_background: float

    @abc.abstractmethod
    def sample_background(self, n: int, rng: np.random.Generator) -> Events:
        """Docstring"""

    @abc.abstractmethod
    def sample_signal(self, n: int, rng: np.random.Generator) -> SimEvents:
        """Docstring"""

    @abc.abstractmethod
    def calculate_n_signal(self, time_integrated_flux: float) -> float:
        """Docstring"""

    @abc.abstractmethod
    def evaluate_background_sindec_pdf(self, events: Events) -> np.ndarray:
        """Docstring"""


@dataclass(kw_only=True)
class DataHandler(metaclass=abc.ABCMeta):
    """Docstring"""
    n_background: float = field(init=False, repr=False)

    @abc.abstractmethod
    def build_background_sindec_logenergy_histogram(self, bins: np.ndarray) -> np.ndarray:
        """Docstring"""

    @abc.abstractmethod
    def build_signal_sindec_logenergy_histogram(
            self, gamma: float, bins: np.ndarray) -> np.ndarray:
        """Docstring"""

    @property
    @abc.abstractmethod
    def dec_cut_loc(self) -> float:
        """Docstring"""

    @dec_cut_loc.setter
    @abc.abstractmethod
    def dec_cut_loc(self, new_dec_cut_loc: Optional[float]) -> None:
        """Docstring"""

    @abc.abstractmethod
    def build_injector(self) -> Injector:
        """Docstring"""


@dataclass(kw_only=True)
class NuSourcesInjector(Injector):
    """Docstring"""
    sim: SimEvents
    data: Events
    dec_spline: Spline

    def sample_background(self, n: int, rng: np.random.Generator) -> Events:
        """Docstring"""
        return self.data.sample(n, rng)

    def sample_signal(self, n: int, rng: np.random.Generator) -> SimEvents:
        """Docstring"""
        return self.sim.sample(n, rng)

    def calculate_n_signal(self, time_integrated_flux: float) -> float:
        """Docstring"""
        return self.sim.weight.sum() * time_integrated_flux

    def evaluate_background_sindec_pdf(self, events: Events) -> np.ndarray:
        """Calculates the background probability of events based on their dec.

        Args:
            events: An array of events including their declination.

        Returns:
            The value for the background space pdf for the given events decs.
        """
        return (1 / (2 * np.pi)) * self.dec_spline(events.sinDec)


@dataclass(kw_only=True)
class NuSourcesDataHandler(DataHandler, Configurable):
    """
        norm_energy: Normalization Energy (GeV)
        assumed_gamma: Assumed Gamma
        dec_cut_loc: Declination Cut Location (rad)
        dec_band: Declination Bandwidth (rad)
        sin_dec_bins: sin(Declination) Bins
        dec_spline_kwargs: Declination Spline Keyword Arguments
    """
    sim: InitVar[SimEvents]
    data: InitVar[Events]
    grl: InitVar[np.ndarray]

    norm_energy: float = 100e3
    assumed_gamma: float = -2
    dec_cut_loc: Optional[float] = None
    dec_band: Optional[float] = None
    sin_dec_bins: int = 30
    dec_spline_kwargs: dict = field(
        default_factory=lambda: {'bbox': [-1, 1], 's': 1.5e-5, 'ext': 3})

    _sim: SimEvents = field(init=False, repr=False)
    _full_sim: SimEvents = field(init=False, repr=False)
    _data: Events = field(init=False, repr=False)
    _grl: np.ndarray = field(init=False, repr=False)
    n_background: float = field(init=False, repr=False)
    grl_rates: np.ndarray = field(init=False, repr=False)
    dec_spline: Spline = field(init=False, repr=False)
    livetime: float = field(init=False, repr=False)
    _sin_dec_bins: np.ndarray = field(init=False, repr=False)

    def __post_init__(self, sim: SimEvents, data: Events, grl: np.ndarray) -> None:
        self.sim = sim
        self.data_grl = (data, grl)

    def build_injector(self) -> NuSourcesInjector:
        return NuSourcesInjector(
            n_background=self.n_background,
            sim=self._sim,
            data=self._data,
            dec_spline=self.dec_spline,
        )

    def build_background_sindec_logenergy_histogram(self, bins: np.ndarray) -> np.ndarray:
        """Docstring"""
        return np.histogram2d(
            self._data.sinDec,
            self._data.logE,
            bins=bins,
            density=True,
        )[0]

    def build_mcbackground_sindec_logenergy_histogram(
        self,
        bins: np.ndarray,
        mcbkgname: str,
    ) -> np.ndarray:
        """Docstring"""
        return np.histogram2d(
            self._full_sim.sinDec,
            self._full_sim.logE,
            bins=bins,
            weights=self._full_sim[mcbkgname],
            density=True,
        )[0]

    def build_signal_sindec_logenergy_histogram(
            self, gamma: float, bins: np.ndarray) -> np.ndarray:
        """Docstring"""
        return np.histogram2d(
            self._full_sim.sinDec,
            self._full_sim.logE,
            bins=bins,
            weights=self._full_sim.ow * (self._full_sim.trueE)**gamma,
            density=True,
        )[0]

    @property
    def sim(self) -> SimEvents:
        """Docstring"""
        return self._sim

    @property
    def full_sim(self) -> SimEvents:
        """Docstring"""
        return self._full_sim

    @sim.setter
    def sim(self, sim: SimEvents) -> None:
        """Docstring"""
        self._full_sim = sim.copy()

        self._full_sim.weight = self._full_sim.ow * (
            self._full_sim.trueE / self.norm_energy)**self.assumed_gamma

        self._cut_sim_dec()

    def _cut_sim_dec(self) -> None:
        """Docstring"""
        if self.dec_band is None or self.dec_cut_loc is None:
            self._sim = self._full_sim
            return

        sindec_dist = np.abs(self.dec_cut_loc - self._full_sim.trueDec)
        close = sindec_dist < self.dec_band
        self._sim = self._full_sim.from_idx(close)

        self._sim.ow /= 2 * np.pi * (np.min([np.sin(
            self.dec_cut_loc + self.dec_band
        ), 1]) - np.max([np.sin(
            self.dec_cut_loc - self.dec_band
        ), -1]))
        self._sim.weight /= 2 * np.pi * (np.min([np.sin(
            self.dec_cut_loc + self.dec_band
        ), 1]) - np.max([np.sin(
            self.dec_cut_loc - self.dec_band
        ), -1]))

    @property
    def data_grl(self) -> Tuple[Events, np.ndarray]:
        """Docstring"""
        return self._data, self._grl

    @data_grl.setter
    def data_grl(self, data_grl: Tuple[Events, np.ndarray]) -> None:
        """Docstring"""
        self._sin_dec_bins = np.linspace(-1, 1, 1 + self.sin_dec_bins)
        self._data = data_grl[0].copy()
        self._grl = data_grl[1].copy()

        min_mjd = np.min(self._data.time)
        max_mjd = np.max(self._data.time)
        self._grl = self._grl[
            (self._grl['start'] < max_mjd) & (self._grl['stop'] > min_mjd)]

        self.livetime = self._grl['livetime'].sum()
        self.n_background = self._grl['events'].sum()
        self.grl_rates = self._grl['events'] / self._grl['livetime']

        hist, bins = np.histogram(
            self._data.sinDec, bins=self._sin_dec_bins, density=True)
        bin_centers = bins[:-1] + np.diff(bins) / 2

        self._dec_spline = Spline(bin_centers, hist, **self.dec_spline_kwargs)


def _contained_run_mask(
    start: float,
    stop: float,
    grl: np.ndarray,
    return_stop_contained: bool = True,
) -> np.ndarray:
    """Docstring"""

    fully_contained = (grl['start'] >= start) & (grl['stop'] < stop)
    start_contained = (grl['start'] < start) & (grl['stop'] > start)

    if not return_stop_contained:
        return fully_contained | start_contained

    stop_contained = (grl['start'] < stop) & (grl['stop'] > stop)

    return fully_contained | start_contained | stop_contained


def _contained_livetime(
    start: float,
    stop: float,
    contained_runs: np.ndarray,
) -> float:
    """Docstring"""
    runs_before_start = contained_runs[contained_runs['start'] < start]
    runs_after_stop = contained_runs[contained_runs['stop'] > stop]
    contained_livetime = contained_runs['livetime'].sum()

    if len(runs_before_start) == 1:
        contained_livetime -= start - runs_before_start['start'][0]

    if len(runs_after_stop) == 1:
        contained_livetime -= runs_after_stop['stop'][0] - stop

    return contained_livetime


@dataclass(kw_only=True)
class TimeDependentNuSourcesInjector(NuSourcesInjector):
    """Docstring"""
    grl: np.ndarray
    background_time_profile: GenericProfile
    signal_time_profile: GenericProfile

    def sample_background(self, n: int, rng: np.random.Generator) -> Events:
        """Docstring"""
        events = super().sample_background(n, rng)
        return self._randomize_times(events, self.background_time_profile)

    def sample_signal(self, n: int, rng: np.random.Generator) -> SimEvents:
        """Docstring"""
        events = super().sample_signal(n, rng)
        events = self._randomize_times(events, self.signal_time_profile)
        print(events.time)
        return events

    def _randomize_times(
        self,
        events: Events,
        time_profile: GenericProfile,
    ) -> Events:
        grl_start_cdf = time_profile.cdf(self.grl['start'])
        grl_stop_cdf = time_profile.cdf(self.grl['stop'])
        valid = np.logical_and(grl_start_cdf < 1, grl_stop_cdf > 0)
        rates = grl_stop_cdf[valid] - grl_start_cdf[valid]

        if not np.any(valid):
            return events

        runs = np.random.choice(
            self.grl[valid],
            size=len(events),
            replace=True,
            p=rates / rates.sum(),
        )

        events.time = time_profile.inverse_transform_sample(
            runs['start'], runs['stop'])
        events.sort('time')
        return events

    def contained_livetime(self, start: float, stop: float) -> float:
        """Docstring"""
        contained_runs = self.grl[_contained_run_mask(start, stop, self.grl)]
        return _contained_livetime(start, stop, contained_runs)


@dataclass(kw_only=True)
class TimeDependentNuSourcesDataHandler(NuSourcesDataHandler):
    """
        outside_time_prof: Outside Time Profile (days)
    """
    background_time_profile: InitVar[GenericProfile]
    signal_time_profile: InitVar[GenericProfile]

    outside_time_prof: Optional[float] = None

    _background_time_profile: GenericProfile = field(init=False, repr=False)
    _signal_time_profile: GenericProfile = field(init=False, repr=False)

    def __post_init__(
        self,
        sim: SimEvents,
        data: Events,
        grl: np.ndarray,
        background_time_profile: GenericProfile,
        signal_time_profile: GenericProfile,
    ) -> None:
        super().__post_init__(sim, data, grl)
        self.background_time_profile = background_time_profile
        self.signal_time_profile = signal_time_profile

    def build_injector(self) -> TimeDependentNuSourcesInjector:
        return TimeDependentNuSourcesInjector(
            n_background=self.n_background,
            sim=self._sim,
            data=self._data,
            grl=self._grl,
            dec_spline=self._dec_spline,
            background_time_profile=self._background_time_profile,
            signal_time_profile=self._signal_time_profile,
        )

    @property
    def background_time_profile(self) -> GenericProfile:
        """Docstring"""
        return self._background_time_profile

    @background_time_profile.setter
    def background_time_profile(self, profile: GenericProfile) -> None:
        """Docstring"""
        # Find the runs contianed in the background time window
        start, stop = profile.range
        return_stop_contained = True

        if self.outside_time_prof is not None:
            stop = start
            start -= self.outside_time_prof
            return_stop_contained = False

        background_run_mask = _contained_run_mask(
            start,
            stop,
            self._grl,
            return_stop_contained=return_stop_contained,
        )

        if not np.any(background_run_mask):
            print('ERROR: No runs found in GRL for calculation of '
                  'background rates!')
            raise RuntimeError

        background_grl = self._grl[background_run_mask]
        self.n_background = background_grl['events'].sum()
        self.n_background /= background_grl['livetime'].sum()
        self.n_background *= _contained_livetime(*profile.range, background_grl)
        self._background_time_profile = copy.deepcopy(profile)

    @property
    def signal_time_profile(self) -> GenericProfile:
        """Docstring"""
        return self._signal_time_profile

    @signal_time_profile.setter
    def signal_time_profile(self, profile: GenericProfile) -> None:
        """Docstring"""
        self._signal_time_profile = copy.deepcopy(profile)
