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
    _n_background: float

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

    @property
    @abc.abstractmethod
    def n_background(self) -> float:
        """Docstring"""


@dataclass(kw_only=True)
class DataHandler(metaclass=abc.ABCMeta):
    """Docstring"""
    _n_background: float = field(init=False, repr=False)

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
    _sim: SimEvents
    _data: Events
    _dec_spline: Spline

    def sample_background(self, n: int, rng: np.random.Generator) -> Events:
        """Docstring"""
        return self._data.sample(n, rng)

    def sample_signal(self, n: int, rng: np.random.Generator) -> SimEvents:
        """Docstring"""
        return self._sim.sample(n, rng)

    def calculate_n_signal(self, time_integrated_flux: float) -> float:
        """Docstring"""
        return self._sim.weight.sum() * time_integrated_flux

    def evaluate_background_sindec_pdf(self, events: Events) -> np.ndarray:
        """Calculates the background probability of events based on their dec.

        Args:
            events: An array of events including their declination.

        Returns:
            The value for the background space pdf for the given events decs.
        """
        return (1 / (2 * np.pi)) * self._dec_spline(events.sinDec)

    @property
    def n_background(self) -> float:
        """Docstring"""
        return self._n_background


@dataclass(kw_only=True)
class NuSourcesDataHandler(DataHandler, Configurable):
    """Docstring"""
    sim: InitVar[SimEvents]
    data_grl: InitVar[Tuple[Events, np.ndarray]]

    _config_map: ClassVar[dict] = {
        '_norm_energy': ('Normalization Energy (GeV)', 100e3),
        '_assumed_gamma': ('Assumed Gamma', -2),
        '_dec_cut_loc': ('Declination Cut Location (rad)', None),
        '_dec_band': ('Declination Bandwidth (rad)', None),
        '_sin_dec_bins_config': ('sin(Declination) Bins', 30),
        '_dec_spline_kwargs': ('Declination Spline Keyword Arguments', {
            'bbox': [-1, 1],
            's': 1.5e-5,
            'ext': 3,
        })
    }

    _norm_energy: float = 100e3
    _assumed_gamma: float = -2
    _dec_cut_loc: Optional[float] = None
    _dec_band: Optional[float] = None
    _sin_dec_bins_config: int = 30
    _dec_spline_kwargs: dict = field(
        default_factory=lambda: {'bbox': [-1, 1], 's': 1.5e-5, 'ext': 3})

    _sim: SimEvents = field(init=False, repr=False)
    _full_sim: SimEvents = field(init=False, repr=False)
    _data: Events = field(init=False, repr=False)
    _grl: np.ndarray = field(init=False, repr=False)
    _n_background: float = field(init=False, repr=False)
    _grl_rates: np.ndarray = field(init=False, repr=False)
    _dec_spline: Spline = field(init=False, repr=False)
    _livetime: float = field(init=False, repr=False)
    _sin_dec_bins: np.ndarray = field(init=False, repr=False)

    def __post_init__(self, sim: SimEvents, data_grl: Tuple[Events, np.ndarray]) -> None:
        self.sim = sim
        self.data_grl = data_grl

    def build_injector(self) -> NuSourcesInjector:
        return NuSourcesInjector(
            _n_background=self._n_background,
            _sim=self._sim,
            _data=self._data,
            _dec_spline=self._dec_spline,
        )

    @classmethod
    def from_config(
            cls,
            config: dict,
            sim: SimEvents,
            data_grl: Tuple[Events, np.ndarray],
    ) -> 'NuSourcesDataHandler':
        return cls(sim=sim, data_grl=data_grl, **cls._map_kwargs(config))

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
            self.full_sim.sinDec,
            self.full_sim.logE,
            bins=bins,
            weights=self.full_sim.ow * self.full_sim.trueE**gamma,
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
            self._full_sim.trueE / self._norm_energy)**self._assumed_gamma

        self._cut_sim_dec()

    def _cut_sim_dec(self) -> None:
        """Docstring"""
        if self._dec_band is None or self._dec_cut_loc is None:
            self._sim = self._full_sim
            return

        sindec_dist = np.abs(self._dec_cut_loc - self._full_sim.trueDec)
        close = sindec_dist < self._dec_band
        self._sim = self._full_sim.from_idx(close)

        self._sim.ow /= 2 * np.pi * (np.min([np.sin(
            self._dec_cut_loc + self._dec_band
        ), 1]) - np.max([np.sin(
            self._dec_cut_loc - self._dec_band
        ), -1]))
        self._sim.weight /= 2 * np.pi * (np.min([np.sin(
            self._dec_cut_loc + self._dec_band
        ), 1]) - np.max([np.sin(
            self._dec_cut_loc - self._dec_band
        ), -1]))

    @property
    def data_grl(self) -> Tuple[Events, np.ndarray]:
        """Docstring"""
        return self._data, self._grl

    @data_grl.setter
    def data_grl(self, data_grl: Tuple[Events, np.ndarray]) -> None:
        """Docstring"""
        self._sin_dec_bins = np.linspace(-1, 1, 1 + self._sin_dec_bins_config)
        self._data = data_grl[0].copy()
        self._grl = data_grl[1].copy()

        min_mjd = np.min(self._data.time)
        max_mjd = np.max(self._data.time)
        self._grl = self._grl[
            (self._grl['start'] < max_mjd) & (self._grl['stop'] > min_mjd)]

        self._livetime = self._grl['livetime'].sum()
        self._n_background = self._grl['events'].sum()
        self._grl_rates = self._grl['events'] / self._grl['livetime']

        hist, bins = np.histogram(
            self._data.sinDec, bins=self._sin_dec_bins, density=True)
        bin_centers = bins[:-1] + np.diff(bins) / 2

        self._dec_spline = Spline(bin_centers, hist, **self._dec_spline_kwargs)

    @property
    def livetime(self) -> float:
        """Docstring"""
        return self._livetime

    @property
    def dec_cut_loc(self) -> Optional[float]:
        return self._dec_cut_loc

    @dec_cut_loc.setter
    def dec_cut_loc(self, new_dec_cut_loc: Optional[float]) -> None:
        self._dec_cut_loc = new_dec_cut_loc
        self._cut_sim_dec()


@dataclass(kw_only=True)
class TimeDependentNuSourcesInjector(NuSourcesInjector):
    """Docstring"""
    _grl: np.ndarray
    _background_time_profile: GenericProfile
    _signal_time_profile: GenericProfile

    def sample_background(self, n: int, rng: np.random.Generator) -> Events:
        """Docstring"""
        events = super().sample_background(n, rng)
        return self._randomize_times(events, self._background_time_profile)

    def sample_signal(self, n: int, rng: np.random.Generator) -> SimEvents:
        """Docstring"""
        events = super().sample_signal(n, rng)
        return self._randomize_times(events, self._signal_time_profile)

    def _randomize_times(
        self,
        events: Events,
        time_profile: GenericProfile,
    ) -> Events:
        grl_start_cdf = time_profile.cdf(self._grl['start'])
        grl_stop_cdf = time_profile.cdf(self._grl['stop'])
        valid = np.logical_and(grl_start_cdf < 1, grl_stop_cdf > 0)
        rates = grl_stop_cdf[valid] - grl_start_cdf[valid]

        if not np.any(valid):
            return events

        runs = np.random.choice(
            self._grl[valid],
            size=len(events),
            replace=True,
            p=rates / rates.sum(),
        )

        events.time = time_profile.inverse_transform_sample(
            runs['start'], runs['stop'])
        events.sort('time')
        return events


@dataclass(kw_only=True)
class TimeDependentNuSourcesDataHandler(NuSourcesDataHandler):
    """Docstring"""
    background_time_profile: InitVar[GenericProfile]
    signal_time_profile: InitVar[GenericProfile]

    _config_map: ClassVar[dict] = {
        **NuSourcesDataHandler._config_map,
        '_outside_time_prof': ('Outside Time Profile (days)', None),
    }

    _outside_time_prof: Optional[float] = None

    _background_time_profile: GenericProfile = field(init=False, repr=False)
    _signal_time_profile: GenericProfile = field(init=False, repr=False)

    def __post_init__(
        self,
        sim: SimEvents,
        data_grl: Tuple[Events, np.ndarray],
        background_time_profile: GenericProfile,
        signal_time_profile: GenericProfile,
    ) -> None:
        super().__post_init__(sim, data_grl)
        self.background_time_profile = background_time_profile
        self.signal_time_profile = signal_time_profile

    def build_injector(self) -> TimeDependentNuSourcesInjector:
        return TimeDependentNuSourcesInjector(
            _n_background=self._n_background,
            _sim=self._sim,
            _data=self._data,
            _grl=self._grl,
            _dec_spline=self._dec_spline,
            _background_time_profile=self._background_time_profile,
            _signal_time_profile=self._signal_time_profile,
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        sim: SimEvents,
        data_grl: Tuple[Events, np.ndarray],
        background_time_profile: GenericProfile,
        signal_time_profile: GenericProfile,
    ) -> 'TimeDependentNuSourcesDataHandler':
        return cls(
            sim=sim,
            data_grl=data_grl,
            background_time_profile=background_time_profile,
            signal_time_profile=signal_time_profile,
            **cls._map_kwargs(config),
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

        if self._outside_time_prof is not None:
            stop = start
            start -= self._outside_time_prof
            return_stop_contained = False

        background_run_mask = self._contained_run_mask(
            start,
            stop,
            return_stop_contained=return_stop_contained,
        )

        if not np.any(background_run_mask):
            print('ERROR: No runs found in GRL for calculation of '
                  'background rates!')
            raise RuntimeError

        background_grl = self._grl[background_run_mask]
        self._n_background = background_grl['events'].sum()
        self._n_background /= background_grl['livetime'].sum()
        self._n_background *= self._contained_livetime(*profile.range, background_grl)
        self._background_time_profile = copy.deepcopy(profile)

    @property
    def signal_time_profile(self) -> GenericProfile:
        """Docstring"""
        return self._signal_time_profile

    @signal_time_profile.setter
    def signal_time_profile(self, profile: GenericProfile) -> None:
        """Docstring"""
        self._signal_time_profile = copy.deepcopy(profile)

    def _contained_run_mask(
        self,
        start: float,
        stop: float,
        return_stop_contained: bool = True,
    ) -> np.ndarray:
        """Docstring"""

        fully_contained = (
            self._grl['start'] >= start
        ) & (self._grl['stop'] < stop)

        start_contained = (
            self._grl['start'] < start
        ) & (self._grl['stop'] > start)

        if not return_stop_contained:
            return fully_contained | start_contained

        stop_contained = (
            self._grl['start'] < stop
        ) & (self._grl['stop'] > stop)

        return fully_contained | start_contained | stop_contained

    def contained_livetime(self, start: float, stop: float) -> float:
        """Docstring"""
        contained_runs = self._grl[self._contained_run_mask(start, stop)]
        return self._contained_livetime(start, stop, contained_runs)

    @staticmethod
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
