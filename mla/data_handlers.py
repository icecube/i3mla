"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import TYPE_CHECKING

import abc
import copy
from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar

import numpy as np
import numpy.lib.recfunctions as rf
from scipy.interpolate import UnivariateSpline as Spline

if TYPE_CHECKING:
    from .time_profiles import GenericProfile
else:
    GenericProfile = object


@dataclass
class DataHandler:
    """Docstring"""
    config: dict

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sample_background(self, n: int) -> np.ndarray:
        """Docstring"""

    @abc.abstractmethod
    def sample_signal(self, n: int) -> np.ndarray:
        """Docstring"""

    @abc.abstractmethod
    def calculate_ns(self, time_integrated_flux: float) -> float:
        """Docstring"""

    @abc.abstractmethod
    def evaluate_background_sindec_pdf(self, events: np.ndarray) -> np.ndarray:
        """Docstring"""

    @abc.abstractmethod
    def build_background_sindec_logenergy_histogram(self, bins: np.ndarray) -> np.ndarray:
        """Docstring"""

    @abc.abstractmethod
    def build_signal_sindec_logenergy_histogram(
            self, gamma: float, bins: np.ndarray) -> np.ndarray:
        """Docstring"""

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        return {}


@dataclass
class NuSourcesDataHandler(DataHandler):
    """Docstring"""
    sim: InitVar[np.ndarray]
    data: InitVar[np.ndarray]
    grl: InitVar[np.ndarray]

    _sim: np.ndarray = field(init=False, repr=False)
    _data: np.ndarray = field(init=False, repr=False)
    _grl: np.ndarray = field(init=False, repr=False)
    _n_background: int = field(init=False, repr=False)
    _grl_rates: np.ndarray = field(init=False, repr=False)
    _dec_spline: Spline = field(init=False, repr=False)
    _livetime: float = field(init=False, repr=False)

    def __post_init__(self, sim:np.ndarray, data:np.ndarray, grl: np.ndarray) -> None:
        """Docstring"""
        if isinstance(self.config['sin_dec_bins'], int):
            self.config['sin_dec_bins'] = np.linspace(
                -1, 1, 1 + self.config['sin_dec_bins'])

        self.sim = sim
        self.set_data_grl(data, grl)

    def set_data_grl(self, data: np.ndarray, grl: np.ndarray) -> None:
        """Docstring"""
        self._data = data.copy()
        self._grl = grl.copy()
        if 'sindec' not in self._data.dtype.names:
            self._data = rf.append_fields(
                self._data,
                'sindec',
                np.sin(self._data['dec']),
                usemask=False,
            )

        min_mjd = np.min(self.data['time'])
        max_mjd = np.max(self.data['time'])
        self._grl = self.grl[(self.grl['start'] < max_mjd) & (self.grl['stop'] > min_mjd)]

        self._livetime = self.grl['livetime'].sum()
        self._n_background = self.grl['events'].sum()
        self._grl_rates = self.grl['events'] / self.grl['livetime']

        hist, bins = np.histogram(
            self.data['sindec'], bins=self.config['sin_dec_bins'], density=True)
        bin_centers = bins[:-1] + np.diff(bins) / 2

        self._dec_spline = Spline(
            bin_centers,
            hist,
            bbox=self.config['dec_spline_bbox'],
            s=self.config['dec_spline_s'],
            ext=self.config['dec_spline_ext'],
        )

    def sample_background(self, n: int) -> np.ndarray:
        """Docstring"""
        return np.random.choice(self._data, n).copy()

    def sample_signal(self, n: int):
        """Docstring"""
        return np.random.choice(
            self.sim,
            n,
            p=self.sim['weight'] / self.sim['weight'].sum(),
            replace=False,
        ).copy()

    def calculate_ns(self, time_integrated_flux: float) -> float:
        """Docstring"""
        return self.sim['weight'].sum() * time_integrated_flux

    def evaluate_background_sindec_pdf(self, events: np.ndarray) -> np.ndarray:
        """Calculates the background probability of events based on their dec.

        Args:
            events: An array of events including their declination.

        Returns:
            The value for the background space pdf for the given events decs.
        """
        return (1 / (2 * np.pi)) * self._dec_spline(events['sindec'])

    def build_background_sindec_logenergy_histogram(self, bins: np.ndarray) -> np.ndarray:
        """Docstring"""
        return np.histogram2d(
            self.data['sindec'],
            self.data['logE'],
            bins=bins,
            density=True,
        )[0]

    def build_signal_sindec_logenergy_histogram(
        self, gamma: float, bins: np.ndarray) -> np.ndarray:
        """Docstring"""
        return np.histogram2d(
            self.sim['sindec'],
            self.sim['logE'],
            bins=bins,
            weights=self.sim['ow'] * self.sim['trueE']**gamma,
            density=True,
        )[0]

    @property
    def sim(self) -> np.ndarray:
        """Docstring"""
        return self._sim

    @property
    def data(self) -> np.ndarray:
        """Docstring"""
        return self._data

    @property
    def grl(self) -> np.ndarray:
        """Docstring"""
        return self._grl

    @property
    def livetime(self) -> float:
        """Docstring"""
        return self._livetime

    @property
    def n_background(self) -> float:
        """Docstring"""
        return self._n_background

    @sim.setter
    def sim(self, sim: np.ndarray) -> None:
        """Docstring"""
        self._sim = sim.copy()
        if 'sindec' not in self._sim.dtype.names:
            self._sim = rf.append_fields(
                self._sim,
                'sindec',
                np.sin(self._sim['dec']),
                usemask=False,
            )

        if 'weight' not in self._sim.dtype.names:
            self._sim = rf.append_fields(
                self._sim, 'weight',
                np.zeros(len(self._sim)),
                dtypes=np.float32
            )

        self._sim['weight'] = self._sim['ow'] * (
            self.sim['trueE'] / self.config['normalization_energy (GeV)']
        )**self.config['gamma']

        if self.config['dec_bandwidth'] is not None:
            sindec_dist = np.abs(self.source.dec - self.sim['truedec'])
            close = sindec_dist < self.config['dec_bandwidth']
            self.sim = self.sim[close]

            self._sim['ow'] /= 2 * np.pi * (np.min(
                [np.sin(self.source.dec + self.config['dec_bandwidth']), 1]
            ) - np.max([np.sin(self.source.dec - self.config['dec_bandwidth']), -1]))

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['dec_bandwidth'] = np.deg2rad(3)
        config['sin_dec_bins'] = 500
        config['dec_spline_bbox'] = [-1, 1]
        config['dec_spline_s'] = 1.5e-5
        config['dec_spline_ext'] = 3
        return config

    def inject_events(self) -> np.ndarray:
        """Injects background events for a trial.

        Returns:
            An array of injected background events.
        """
        # Get the number of events we see from these runs
        n_background_observed = np.random.poisson(self._n_background)

        # How many events should we add in? This will now be based on the
        # total number of events actually observed during these runs
        background = np.random.choice(self.data, n_background_observed).copy()

        # Randomize the background RA
        background['ra'] = np.random.uniform(0, 2 * np.pi, len(background))

        return background


@dataclass
class TimeDependentNuSourcesDataHandler(NuSourcesDataHandler):
    """Docstring"""
    background_time_profile: InitVar[GenericProfile]
    signal_time_profile: InitVar[GenericProfile]

    _background_time_profile: GenericProfile = field(init=False, repr=False)
    _signal_time_profile: GenericProfile = field(init=False, repr=False)

    def __post_init__(
        self,
        sim: np.ndarray,
        data: np.ndarray,
        grl: np.ndarray,
        background_time_profile,
        signal_time_profile,
    ) -> None:
        """Docstring"""
        super().__post_init__(sim, data, grl)

        self._background_time_profile = copy.deepcopy(background_time_profile)
        self._signal_time_profile = copy.deepcopy(signal_time_profile)

        # Find the runs contianed in the background time window
        start, stop = self._background_time_profile.range
        return_stop_contained = True

        if self.config['outside_time_profile'] is not None:
            stop = start
            start -= self.config['outside_time_profile']
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

        background_grl = self.grl[background_run_mask]
        self._n_background = background_grl['events'].sum()
        self._n_background /= background_grl['livetime'].sum()
        self._n_background *= self._contained_livetime(
            *self.config['background_time_profile'].range,
            background_grl,
        )

    def _contained_run_mask(
        self,
        start: float,
        stop: float,
        return_stop_contained: bool = True,
    ) -> np.ndarray:
        """Docstring"""
        fully_contained = (
            self.grl['start'] >= start
        ) & (self.grl['stop'] < stop)

        start_contained = (
            self.grl['start'] < start
        ) & (self.grl['stop'] > start)

        if not return_stop_contained:
            return fully_contained | start_contained

        stop_contained = (
            self.grl['start'] < stop
        ) & (self.grl['stop'] > stop)

        return fully_contained | start_contained | stop_contained

    def contained_livetime(self, start: float, stop: float) -> float:
        """Docstring"""
        contained_runs = self.grl[self._contained_run_mask(start, stop)]
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

    def sample_background(self, n: int) -> np.ndarray:
        """Docstring"""
        events = super().sample_background(n)
        return self._randomize_times(events, self._background_time_profile)

    def sample_signal(self, n: int) -> np.ndarray:
        """Docstring"""
        events = super().sample_sinal(n)
        return self._randomize_times(events, self._signal_time_profile)

    def _radomize_times(
        self,
        events: np.ndarray,
        time_profile: GenericProfile,
    ) -> np.ndarray:
        grl_start_cdf = time_profile.cdf(self.grl['start'])
        grl_stop_cdf = time_profile.cdf(self.grl['stop'])
        valid = np.logical_and(grl_start_cdf < 1, grl_stop_cdf > 0)
        rates = grl_stop_cdf[valid] - grl_start_cdf[valid]

        runs = np.random.choice(
            self.grl[valid],
            size=len(events),
            replace=True,
            p=rates / rates.sum(),
        )

        events['time'] = time_profile.inverse_transform_sample(
            runs['start'], runs['stop'])

        return events

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['outside_time_profile (days)'] = None
        return config
