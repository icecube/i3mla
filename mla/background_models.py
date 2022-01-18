"""
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Optional, Union
from typing import TYPE_CHECKING

from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar

from . import time_profiles

import numpy as np
import numpy.lib.recfunctions as rf
from scipy.interpolate import UnivariateSpline as Spline

if TYPE_CHECKING:
    from .time_profiles import GenericProfile
else:
    GenericProfile = object


@dataclass
class BaseBackgroundModel:
    """Stores the events and pre-processed parameters used in analyses.

    Currently, this class uses internal data and monte-carlo datasets. This
    will be updated before the first release to use the upcoming public data
    release.

    Attributes:
        data (np.ndarray): Real neutrino event data.
        grl (np.ndarray): A list of runs/times when the detector was working
            properly.
    """
    data: InitVar[np.ndarray]
    grl: InitVar[np.ndarray]
    config: dict 

    _data: np.ndarray = field(init=False, repr=False)
    _grl: np.ndarray = field(init=False, repr=False)
    _n_background: int = field(init=False, repr=False)
    _grl_rates: np.ndarray = field(init=False, repr=False)
    _dec_spline: Spline = field(init=False, repr=False)
    _livetime: float = field(init=False, repr=False)

    def __post_init__(self, data:np.ndarray, grl: np.ndarray) -> None:
        """Initializes EventModel and calculates energy sob maps."""
        if 'sindec' not in self.data.dtype.names:
            self._data = rf.append_fields(
                data,
                'sindec',
                np.sin(self.data['dec']),
                usemask=False,
            )

        min_mjd = np.min(self.data['time'])
        max_mjd = np.max(self.data['time'])
        self._grl = grl[(grl['start'] < max_mjd) & (grl['stop'] > min_mjd)]

        if isinstance(self.config['sin_dec_bins'], int):
            sin_dec_bins = np.linspace(-1, 1, 1 + self.config['sin_dec_bins'])
        else:
            sin_dec_bins = self.config['sin_dec_bins']

        self._dec_spline = self._init_dec_spline(sin_dec_bins)

        self._livetime = self.grl['livetime'].sum()
        self._n_background = self.grl['events'].sum()
        self._grl_rates = self.grl['events'] / self.grl['livetime']

    def _init_dec_spline(self, sin_dec_bins: np.array) -> Spline:
        """Builds a histogram of neutrino flux vs. sin(dec) and splines it.

        The UnivariateSpline function call uses these default arguments:
        bbox=[-1.0, 1.0], s=1.5e-5, ext=1. To replace any of these defaults, or
        to pass any other args/kwargs to UnivariateSpline, just pass them to
        this function.

        Args:
            sin_dec_bins: A numpy array of bin edges to use to build the
                histogram to spline.

        Returns:
            A spline function of the neutrino flux vs. sin(dec) histogram.
        """
        # Our background PDF only depends on declination.
        # In order for us to capture the dec-dependent
        # behavior, we first take a look at the dec values
        # in the data. We can do this by histogramming them.
        hist, bins = np.histogram(self.data['sindec'], bins=sin_dec_bins, density=True)
        bin_centers = bins[:-1] + np.diff(bins) / 2

        # These values have a lot of "noise": they jump
        # up and down quite a lot. We could use fewer
        # bins, but that may hide some features that
        # we care about. We want something that captures
        # the right behavior, but is smooth and continuous.
        # The best way to do that is to use a "spline",
        # which will fit a continuous and differentiable
        # piecewise polynomial function to our data.
        # We can set a smoothing factor (s) to control
        # how smooth our spline is.

        return Spline(
            bin_centers,
            hist,
            bbox=self.config['dec_spline_bbox'],
            s=self.config['dec_spline_s'],
            ext=self.config['dec_spline_ext'],
        )

    def spatial_pdf(self, events: np.array) -> np.array:
        """Calculates the background probability of events based on their dec.

        Uses the background_dec_spline() function from the given event_model to
        get the probabilities.

        Args:
            events: An array of events including their declination.

        Returns:
            The value for the background space pdf for the given events decs.
        """
        return (1 / (2 * np.pi)) * self._dec_spline(events['sindec'])

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

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        return {
            'sin_dec_bins': 500,
            'dec_spline_bbox': [-1, 1],
            'dec_spline_s': 1.5e-5,
            'dec_spline_ext': 3,
        }


@dataclass
class TimeDependentMixin(BaseBackgroundModel):
    """Docstring"""

    def __post_init__(self, data: np.ndarray, grl: np.ndarray) -> None:
        """Initializes EventModel and calculates energy sob maps.

        Args:
            grl:

        Raises:
            RuntimeError:
        """
        super().__post_init__(data, grl)

        if self.config['time_profile'] is None:
            self.config['time_profile'] = time_profiles.UniformProfile(
                start=np.min(data['time']),
                length=np.max(data['time']) - np.min(data['time']),
            )

        # Find the runs contianed in the background time window
        start, stop = self.config['time_profile'].range
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
            *self.config['time_profile'].range,
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

    def scramble_times(
        self,
        times: np.ndarray,
        time_profile: Optional[GenericProfile] = None,
    ) -> np.ndarray:
        """Docstring"""
        grl_start_cdf = time_profile.cdf(self.grl['start'])
        grl_stop_cdf = time_profile.cdf(self.grl['stop'])
        valid = np.logical_and(grl_start_cdf < 1, grl_stop_cdf > 0)
        rates = grl_stop_cdf[valid] - grl_start_cdf[valid]

        runs = np.random.choice(
            self.grl[valid],
            size=len(times),
            replace=True,
            p=rates / rates.sum(),
        )

        return time_profile.inverse_transform_sample(runs['start'], runs['stop'])

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['time_profile'] = None
        config['outside_time_profile (days)'] = None
        return config


class TimeDependentDetectorModel(TimeDependentMixin, BaseDetectorModel):
    """Docstring"""
    pass
