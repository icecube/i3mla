"""
The classes in this file do preprocessing on data and monte carlo to be used
to do a point source analysis.
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List, Union, Optional, Tuple

import scipy
import numpy as np
import numpy.lib.recfunctions as rf
from scipy.interpolate import UnivariateSpline as Spline

from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar

from . import sources
from . import spectral
from . import time_profiles


def angular_distance(src_ra: float, src_dec: float, r_a: float,
                     dec: float) -> float:
    """Computes angular distance between source and location.

    Args:
        src_ra: The right ascension of the first point (radians).
        src_dec: The declination of the first point (radians).
        r_a: The right ascension of the second point (radians).
        dec: The declination of the second point (radians).

    Returns:
        The distance, in radians, between the two points.
    """
    sin_dec = np.sin(dec)

    cos_dec = np.sqrt(1. - sin_dec**2)

    cos_dist = (
        np.cos(src_ra - r_a) * np.cos(src_dec) * cos_dec
    ) + np.sin(src_dec) * sin_dec
    # handle possible floating precision errors
    cos_dist = np.clip(cos_dist, -1, 1)

    return np.arccos(cos_dist)


def cross_matrix(mat: np.array) -> np.array:
    """Calculate cross product matrix.
    A[ij] = x_i * y_j - y_i * x_j
    Args:
        mat: A 2D array to take the cross product of.
    Returns:
        The cross matrix.
    """
    skv = np.roll(np.roll(np.diag(mat.ravel()), 1, 1), -1, 0)
    return skv - skv.T


def rotate(ra1: float, dec1: float, ra2: float, dec2: float,
           ra3: float, dec3: float) -> Tuple[float, float]:
    """Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2).

    The rotation is performed on (ra3, dec3).

    Args:
        ra1: The right ascension of the point to be rotated from.
        dec1: The declination of the point to be rotated from.
        ra2: the right ascension of the point to be rotated onto.
        dec2: the declination of the point to be rotated onto.
        ra3: the right ascension of the point that will actually be rotated.
        dec3: the declination of the point that will actually be rotated.

    Returns:
        The rotated ra3 and dec3.

    Raises:
        IndexError: Arguments must all have the same dimension.
    """
    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    if not (
        len(ra1) == len(dec1) == len(ra2) == len(dec2) == len(ra3) == len(dec3)
    ):
        raise IndexError('Arguments must all have the same dimension.')

    cos_alpha = np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2) \
        + np.sin(dec1) * np.sin(dec2)

    # correct rounding errors
    cos_alpha[cos_alpha > 1] = 1
    cos_alpha[cos_alpha < -1] = -1

    alpha = np.arccos(cos_alpha)
    vec1 = np.vstack([np.cos(ra1) * np.cos(dec1),
                      np.sin(ra1) * np.cos(dec1),
                      np.sin(dec1)]).T
    vec2 = np.vstack([np.cos(ra2) * np.cos(dec2),
                      np.sin(ra2) * np.cos(dec2),
                      np.sin(dec2)]).T
    vec3 = np.vstack([np.cos(ra3) * np.cos(dec3),
                      np.sin(ra3) * np.cos(dec3),
                      np.sin(dec3)]).T
    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec**2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diagflat(np.ones(3))
    ntn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([cross_matrix(nv) for nv in nvec])

    r = np.array([(1. - np.cos(a)) * ntn_i + np.cos(a) * one + np.sin(a) * nx_i
                  for a, ntn_i, nx_i in zip(alpha, ntn, nx)])
    vec = np.array([np.dot(r_i, vec_i.T) for r_i, vec_i in zip(r, vec3)])

    r_a = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    r_a += np.where(r_a < 0., 2. * np.pi, 0.)

    return r_a, dec


@dataclass
class _EventModelBase:
    """Stores the events and pre-processed parameters used in analyses.

    Currently, this class uses internal data and monte-carlo datasets. This
    will be updated before the first release to use the upcoming public data
    release.

    Attributes:
        data (np.ndarray): Real neutrino event data.
        sim (np.ndarray): Simulated neutrino events.
        grl (np.ndarray): A list of runs/times when the detector was working
            properly.
        reduced_sim (np.ndarray):
        gamma (float):
    """
    source: InitVar[sources.Source]

    _data: np.ndarray
    _sim: np.ndarray

    grl: InitVar[np.ndarray]

    _gamma: float

    _grl: np.ndarray = field(init=False)
    _reduced_sim: np.ndarray = field(init=False)
    _background_dec_spline: Spline = field(init=False)


@dataclass
class _EventModelDefaultsBase:
    """Stores the events and pre-processed parameters used in analyses.

    Currently, this class uses internal data and monte-carlo datasets. This
    will be updated before the first release to use the upcoming public data
    release.

    Attributes:
        sampling_width:
        background_dec_spline: A spline fit of neutrino flux vs. sin(dec).
    """
    _sampling_width: Optional[float] = field(default=None)
    background_sin_dec_bins: InitVar[Union[np.array, int]] = field(default=500)

    @property
    def sampling_width(self) -> float:
        """Docstring"""
        return self._sampling_width


@dataclass
class EventModel(_EventModelDefaultsBase, _EventModelBase):
    """Stores the events and pre-processed parameters used in analyses.

    Currently, this class uses internal data and monte-carlo datasets. This
    will be updated before the first release to use the upcoming public data
    release.
    """
    def __post_init__(self, source: sources.Source, grl: np.ndarray,
                      background_sin_dec_bins: Union[np.array, int]) -> None:
        """Initializes EventModel and calculates energy sob maps.

        Args:
            source:
            grl:
            background_sin_dec_bins: If an int, then the number of bins
                spanning -1 -> 1, otherwise, a numpy array of bin edges.

        Raises:
            ValueError:
        """
        try:
            self._data = rf.append_fields(self._data, 'sindec',
                                          np.sin(self._data['dec']),
                                          usemask=False)
            # The full simulation set,this is for the overall normalization of
            # the Energy S/B ratio
        except ValueError:  # sindec already exist
            pass

        try:
            self._sim = rf.append_fields(self._sim, 'sindec',
                                         np.sin(self._sim['dec']),
                                         usemask=False)
            # The full simulation set,this is for the overall normalization of
            # the Energy S/B ratio
        except ValueError:  # sindec already exist
            pass

        min_mjd = np.min(self._data['time'])
        max_mjd = np.max(self._data['time'])
        self._grl = grl[(grl['start'] < max_mjd) & (grl['stop'] > min_mjd)]

        self._init_reduced_sim(source)

        if isinstance(background_sin_dec_bins, int):
            background_sin_dec_bins = np.linspace(-1, 1,
                                                  1 + background_sin_dec_bins)

        self._background_dec_spline = self._init_background_dec_spline(
            background_sin_dec_bins)

    def _init_background_dec_spline(self, sin_dec_bins: np.array, *args,
                                    **kwargs) -> Spline:
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
        hist, bins = np.histogram(self._data['sindec'], bins=sin_dec_bins,
                                  density=True)
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

        if 'bbox' not in kwargs:
            kwargs['bbox'] = [-1.0, 1.0]
        # if 's' not in kwargs:
        #    kwargs['s'] = 1.5e-5
        if 'ext' not in kwargs:
            kwargs['ext'] = 3

        return Spline(bin_centers, hist, *args, **kwargs)

    def _init_reduced_sim(self, source: sources.Source) -> None:
        """Gets a small simulation dataset to use for injecting signal.

        Prunes the simulation set to only events close to a given source and
        calculate the weight for each event. Adds the weights as a new column
        to the simulation set.

        Args:
            source:

        Returns:
            A reweighted simulation set around the source declination.
        """
        if self._sampling_width is not None:
            self._cut_sim_truedec(source)
        else:
            self._reduced_sim = self._sim.copy()
        self._reduced_sim = self._weight_reduced_sim(self._reduced_sim)
        self._randomize_sim_times()

    def _randomize_sim_times(self) -> None:
        """Docstring"""
        # Randomly assign times to the simulation events within the data time
        # range.
        min_time = np.min(self._data['time'])
        max_time = np.max(self._data['time'])
        self._reduced_sim['time'] = np.random.uniform(
            min_time, max_time, size=len(self._reduced_sim))

    def _cut_sim_truedec(self, source: sources.Source) -> None:
        """Select simulation events in a true dec band(for ns calculation)

        Args:
            source:
        """
        sindec_dist = np.abs(source.dec - self._sim['trueDec'])
        close = sindec_dist < self._sampling_width
        self._reduced_sim = self._sim[close].copy()

        omega = 2 * np.pi * (np.min(
            [np.sin(source.dec + self._sampling_width), 1]
        ) - np.max([np.sin(source.dec - self._sampling_width), -1]))
        self._reduced_sim['ow'] /= omega

    def _weight_reduced_sim(self, reduced_sim: np.ndarray) -> np.ndarray:
        """Docstring"""
        reduced_sim = rf.append_fields(
            reduced_sim, 'weight',
            np.zeros(len(reduced_sim)),
            dtypes=np.float32
        )

        # Assign the weights using the newly defined "time profile"
        # classes above. If you want to make this a more complicated
        # shape, talk to me and we can work it out.
        rescaled_energy = (reduced_sim['trueE'] / 100.e3)**self._gamma
        reduced_sim['weight'] = reduced_sim['ow'] * rescaled_energy
        return reduced_sim

    def signal_spatial_pdf(self, source: sources.Source,
                           events: np.ndarray) -> np.array:
        """Calculates the signal probability of events.

        Gives a gaussian probability based on their angular distance from the
        source object.

        Args:
            source:
            events: An array of events including their positional data.

        Returns:
            The value for the signal spatial pdf for the given events angular
            distances.
        """
        sigma = events['angErr']
        dist = angular_distance(events['ra'], events['dec'], source.r_asc,
                                source.dec)
        norm = 1 / (2 * np.pi * sigma**2)
        return norm * np.exp(-dist**2 / (2 * sigma**2))

    def background_spatial_pdf(self, events: np.array) -> np.array:
        """Calculates the background probability of events based on their dec.

        Uses the background_dec_spline() function from the given event_model to
        get the probabilities.

        Args:
            events: An array of events including their declination.
            event_model: Preprocessed data and simulation.

        Returns:
            The value for the background space pdf for the given events decs.
        """
        bg_densities = self._background_dec_spline(events['sindec'])
        return (1 / (2 * np.pi)) * bg_densities

    def inject_background_events(self) -> np.ndarray:
        """Injects background events for a trial.

        Args:
            event_model: Preprocessed data and simulation.

        Returns:
            An array of injected background events.
        """
        # Get the number of events we see from these runs
        n_background = self._grl['events'].sum()
        n_background_observed = np.random.poisson(n_background)

        # How many events should we add in? This will now be based on the
        # total number of events actually observed during these runs
        background = np.random.choice(self._data, n_background_observed).copy()

        # Randomize the background RA
        background['ra'] = np.random.uniform(0, 2 * np.pi, len(background))

        return background

    def inject_signal_events(
        self, source: sources.Source,
        flux_norm: float,
        n_signal_observed: Optional[int] = None,
    ) -> np.ndarray:
        """Injects signal events for a trial.

        Args:
            source:
            flux_norm:
            n_signal_observed:

        Returns:
            An array of injected signal events.
        """

        # Pick the signal events
        total = self._reduced_sim['weight'].sum()
        if n_signal_observed is None:
            n_signal_observed = scipy.stats.poisson.rvs(total * flux_norm)
        signal = np.random.choice(
            self._reduced_sim,
            n_signal_observed,
            p=self._reduced_sim['weight'] / total,
            replace=False).copy()

        if len(signal) > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source.r_asc, ones * source.dec,
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source.r_asc, ones * source.dec,
                signal['trueRa'], signal['trueDec'])

        return signal

    def grl_filter(self, events: np.ndarray) -> np.ndarray:
        """Docstring"""
        sorting_indices = np.argsort(events['time'])
        events = events[sorting_indices]

        # We need to check to ensure that every event is contained within
        # a good run. If the event happened when we had deadtime (when we
        # were not taking data), then we need to remove it.
        grl_start = self._grl['start'].reshape((1, len(self._grl)))
        grl_stop = self._grl['stop'].reshape((1, len(self._grl)))
        after_start = np.less(grl_start, events['time'].reshape(len(events), 1))
        before_stop = np.less(events['time'].reshape(len(events), 1), grl_stop)
        during_uptime = np.any(after_start & before_stop, axis=1)
        events = events[during_uptime]

        return events


@dataclass
class _TdEventModelBase(_EventModelBase):
    """Docstring"""
    _background_time_profile: time_profiles.GenericProfile
    _signal_time_profile: time_profiles.GenericProfile
    _n_background: int = field(init=False)


@dataclass
class _TdEventModelDefaultsBase(_EventModelDefaultsBase):
    """Docstring"""
    background_window: InitVar[float] = field(default=14)
    withinwindow: InitVar[bool] = field(default=False)


@dataclass
class TdEventModel(EventModel, _TdEventModelDefaultsBase, _TdEventModelBase):
    """Docstring"""
    def __post_init__(self, source: sources.Source, grl: np.ndarray,
                      background_sin_dec_bins: Union[np.array, int],
                      background_window: float, withinwindow: bool) -> None:
        """Initializes EventModel and calculates energy sob maps.

        Args:
            source:
            grl:
            background_sin_dec_bins: If an int, then the number of bins
                spanning -1 -> 1, otherwise, a numpy array of bin edges.
            background_window:
            withinwindow:

        Raises:
            RuntimeError:
        """
        super().__post_init__(source, grl, background_sin_dec_bins)

        # Find the run contian in the background time window
        start_time, end_time = self._background_time_profile.range
        if withinwindow:
            fully_contained = (self._grl['start'] >= start_time
            ) & (self._grl['stop'] < end_time)
            start_contained = (self._grl['start'] < start_time
            ) & (self._grl['stop'] > start_time)
            background_runs = (fully_contained | start_contained)
            if not np.any(background_runs):
                print('ERROR: No runs found in GRL for calculation of '
                      'background rates!')
                raise RuntimeError
            background_grl = self._grl[background_runs]
        else:
            fully_contained = (
                self._grl['start'] >= start_time - background_window
            ) & (self._grl['stop'] < start_time)
            start_contained = (
                self._grl['start'] < start_time - background_window
            ) & (self._grl['stop'] > start_time - background_window)
            background_runs = (fully_contained | start_contained)
            if not np.any(background_runs):
                print('ERROR: No runs found in GRL for calculation of '
                      'background rates!')
                raise RuntimeError
            background_grl = self._grl[background_runs]

        # Find the background rate
        n_background = background_grl['events'].sum()
        n_background /= background_grl['livetime'].sum()
        n_background *= self._background_time_profile.exposure
        self._n_background = n_background

    def inject_background_events(self) -> np.ndarray:
        """Injects background events with specific time profile for a trial.

        Returns:
            An array of injected background events.
        """
        n_background_observed = np.random.poisson(self._n_background)
        background = np.random.choice(self._data, n_background_observed).copy()
        background['ra'] = np.random.uniform(0, 2 * np.pi, len(background))
        background['time'] = self._background_time_profile.random(
            size=len(background))

        return background

    def inject_signal_events(
        self, source: sources.Source,
        flux_norm: float,
        n_signal_observed: Optional[int] = None,
    ) -> np.ndarray:
        """Function info...

        More function info...

        Args:
            source:
            flux_norm:
            n_signal_observed:

        Returns:
            An array of injected signal events.
        """
        # Pick the signal events
        weighttotal = self._reduced_sim['weight'].sum()
        normed_weight = self._reduced_sim['weight'] / weighttotal

        if n_signal_observed is None:
            n_signal_observed = scipy.stats.poisson.rvs(
                weighttotal * self._signal_time_profile.exposure * flux_norm)

        signal = np.random.choice(self._reduced_sim, n_signal_observed,
                                  p=normed_weight, replace=False).copy()

        if len(signal) > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source.r_asc, ones * source.dec,
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = rotate(
                signal['trueRa'], signal['trueDec'],
                ones * source.r_asc, ones * source.dec,
                signal['trueRa'], signal['trueDec'])

        # Randomize the signal time
        signal['time'] = self._signal_time_profile.random(size=len(signal))

        return signal


@dataclass
class _I3EventModelBase(_TdEventModelBase):
    """Docstring

    Attributes:
        sin_dec_bins (np.array): An array of sin(dec) bin edges for the energy
            maps.
        log_energy_bins (np.array): An array of log(energy) bin edges for the
            energy maps.
        log_sob_gamma_splines (List[List[scipy.interpolate.UnivariateSpline]]):
            A 2D list of spline fits of the log(signal-over-background) vs.
            gamma at a binned energy and sin(dec).
        """
    _sin_dec_bins: np.array = field(init=False)
    _log_energy_bins: np.array = field(init=False)
    _log_sob_gamma_splines: List[List[Spline]] = field(init=False)


@dataclass
class _I3EventModelDefaultsBase(_TdEventModelDefaultsBase):
    """Docstring"""
    signal_sin_dec_bins: InitVar[Union[np.array, int]] = field(default=50)
    log_energy_bins: InitVar[Union[np.array, int]] = field(default=50)
    gamma_bins: InitVar[Union[np.array, int]] = field(default=50)
    verbose: InitVar[bool] = field(default=False)


@dataclass
class I3EventModel(TdEventModel, _I3EventModelDefaultsBase, _I3EventModelBase):
    """Docstring"""
    def __post_init__(self, source: sources.Source, grl: np.ndarray,
                      background_sin_dec_bins: Union[np.array, int],
                      background_window: float, withinwindow: bool,
                      signal_sin_dec_bins: Union[np.array, int],
                      log_energy_bins: Union[np.array, int],
                      gamma_bins: Union[np.array, int],
                      verbose: bool) -> None:
        """Docstring"""
        super().__post_init__(source, grl, background_sin_dec_bins,
                              background_window, withinwindow)

        if isinstance(signal_sin_dec_bins, int):
            signal_sin_dec_bins = np.linspace(-1, 1, 1 + signal_sin_dec_bins)
        self._sin_dec_bins = signal_sin_dec_bins

        if isinstance(log_energy_bins, int):
            log_energy_bins = np.linspace(1, 8, 1 + log_energy_bins)
        self._log_energy_bins = log_energy_bins

        if isinstance(gamma_bins, int):
            gamma_bins = np.linspace(-4.25, -0.5, 1 + gamma_bins)

        self._log_sob_gamma_splines = self._init_log_sob_gamma_splines(
            gamma_bins, verbose=verbose)

    def _init_sob_map(self, gamma: float, *args, verbose: bool = False,
                      **kwargs) -> np.array:
        """Creates sob histogram for a given spectral index (gamma).

        The UnivariateSpline function call uses these default arguments:
        k=1, s=0, ext=3. To replace any of these defaults, or to pass any other
        args/kwargs to UnivariateSpline, just pass them to this function.
        Spline is used here to smooth over the energies to get the values for
        each bin.

        Args:
            gamma: The gamma value to use to weight the signal.
            verbose: A flag to print progress.

        Returns:
            An array of signal-over-background values binned in sin(dec) and
            log(energy) for a given gamma.
        """
        if verbose:
            print(f'Building map for gamma = {gamma}...', end='')
        bins = np.array([self._sin_dec_bins, self._log_energy_bins])
        bin_centers = bins[1, :-1] + np.diff(bins[1]) / 2

        # background
        bg_h, _, _ = np.histogram2d(self._data['sindec'], self._data['logE'],
                                    bins=bins, density=True)

        # signal
        sig_w = self._sim['ow'] * self._sim['trueE']**gamma
        sig_h, _, _ = np.histogram2d(self._sim['sindec'], self._sim['logE'],
                                     bins=bins, weights=sig_w, density=True)

        # Normalize histograms by dec band
        bg_h /= np.sum(bg_h, axis=1)[:, None]
        sig_h /= np.sum(sig_h, axis=1)[:, None]

        ratio = sig_h / bg_h

        if 'k' not in kwargs:
            kwargs['k'] = 1
        if 's' not in kwargs:
            kwargs['s'] = 0
        if 'ext' not in kwargs:
            kwargs['ext'] = 3

        for i in range(ratio.shape[0]):
            # Pick out the values we want to use.
            # We explicitly want to avoid NaNs and infinities
            good = np.isfinite(ratio[i]) & (ratio[i] > 0)
            good_bins, good_vals = bin_centers[good], ratio[i][good]

            # Do a linear interpolation across the energy range
            spline = Spline(good_bins, good_vals, *args, **kwargs)

            # And store the interpolated values
            ratio[i] = spline(bin_centers)
        if verbose:
            print('done')
        return ratio

    def _init_log_sob_gamma_splines(self, gamma_bins: np.array, *args,
                                    verbose: bool = False,
                                    **kwargs) -> List[List[Spline]]:
        """Builds a 3D hist of sob vs. sin(dec), log(energy), and gamma, then
            returns splines of sob vs. gamma.

        The UnivariateSpline function call uses these default arguments:
        k=3, s=0, ext='raise'. To replace any of these defaults, or to pass any
        other args/kwargs to UnivariateSpline, just pass them to this function.
        Spline is used here to smooth over the energies to get the values for
        each bin.

        Args:
            gamma_bins: The spectral indicies at which to build the histograms.
            verbose: A flag to print progress.

        Returns: A Nested spline list of shape (sin_dec_bins, log_energy_bins).
        """
        if verbose:
            print('Building signal-over-background maps...')
        sob_maps = np.array([self._init_sob_map(gamma, verbose=verbose)
                             for gamma in gamma_bins])
        if verbose:
            print('done.')

        if 'k' not in kwargs:
            kwargs['k'] = 3
        if 's' not in kwargs:
            kwargs['s'] = 0
        if 'ext' not in kwargs:
            kwargs['ext'] = 'raise'

        transposed_log_sob_maps = np.log(sob_maps.transpose(1, 2, 0))

        if verbose:
            print('Fitting log(signal-over-background vs. gamma splines)...',
                  end='')

        def spline(log_ratios):
            return Spline(gamma_bins, log_ratios, *args, **kwargs)

        splines = [[spline(log_ratios) for log_ratios in dec_bin]
                   for dec_bin in transposed_log_sob_maps]
        if verbose:
            print('done')

        return splines

    def log_sob_gamma_splines(self, events: np.ndarray) -> List[Spline]:
        """Gets the splines of sob vs. gamma required for each event.

        Args:
            events: An array of events including their positional data.

        Returns:
            A list of splines of sob vs gamma for each event.
        """
        # Get the bin that each event belongs to
        sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1],
                                      events['sindec'])
        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1],
                                         events['logE'])

        return [self._log_sob_gamma_splines[i - 1][j - 1]
                for i, j in zip(sin_dec_idx, log_energy_idx)]


@dataclass
class _ThreeMLEventModelBase(_TdEventModelBase):
    """Docstring"""
    _sin_dec_bins: np.array = field(init=False)
    _log_energy_bins: np.array = field(init=False)
    _edge_point: Tuple[float, float] = field(init=False)
    _background_sob_map: np.ndarray = field(init=False)
    _ratio: np.ndarray = field(init=False)
    _reduced_sim_reconstructed: np.ndarray = field(init=False)


@dataclass
class _ThreeMLEventModelDefaultsBase(_TdEventModelDefaultsBase):
    """Docstring"""
    signal_sin_dec_bins: InitVar[Union[np.array, int]] = field(default=50)
    log_energy_bins: InitVar[Union[np.array, int]] = field(default=50)
    _spectrum: spectral.BaseSpectrum = field(
        default=spectral.PowerLaw(1e3, 1e-14, -2))


@dataclass
class ThreeMLEventModel(
        TdEventModel, _ThreeMLEventModelDefaultsBase, _ThreeMLEventModelBase):
    """Docstring"""
    def __post_init__(self, source: sources.Source, grl: np.ndarray,
                      background_sin_dec_bins: Union[np.array, int],
                      background_window: float, withinwindow: bool,
                      signal_sin_dec_bins: Union[np.array, int],
                      log_energy_bins: Union[np.array, int]) -> None:
        """
        Args:
            source:
            grl:
            background_sin_dec_bins: If an int, then the number of bins
                spanning -1 -> 1, otherwise, a numpy array of bin edges.
            background_window:
            withinwindow:
        """
        super().__post_init__(source, grl, background_sin_dec_bins,
                              background_window, withinwindow)
        if isinstance(signal_sin_dec_bins, int):
            signal_sin_dec_bins = np.linspace(-1, 1, 1 + signal_sin_dec_bins)
        self._sin_dec_bins = signal_sin_dec_bins

        if isinstance(log_energy_bins, int):
            log_energy_bins = np.linspace(1, 8, 1 + log_energy_bins)

        self._log_energy_bins = log_energy_bins
        self._background_sob_map = self._init_background_sob_map()
        self._ratio = self._init_sob_ratio(source)

    def _init_background_sob_map(self) -> None:
        """Create the backgroub SOB map
        """
        # background
        bins = np.array([self._sin_dec_bins, self._log_energy_bins])
        bg_h, _, _ = np.histogram2d(self._data['sindec'], self._data['logE'],
                                    bins=bins, density=True)
        bg_h /= np.sum(bg_h, axis=1)[:, None]
        return bg_h

    def _init_sob_ratio(self, source: sources.Source, *args, **kwargs) -> None:
        """Create the SOB map with a spectrum
        """
        self._init_reduced_sim_reconstructed(source)
        bins = np.array([self._sin_dec_bins, self._log_energy_bins])
        bin_centers = bins[1, :-1] + np.diff(bins[1]) / 2
        sig_w = self._reduced_sim_reconstructed['ow'] * self._spectrum(
            self._reduced_sim_reconstructed['trueE'])
        sig_h, _, _ = np.histogram2d(self._reduced_sim_reconstructed['sindec'],
                                     self._reduced_sim_reconstructed['logE'],
                                     bins=bins, weights=sig_w, density=True)

        # Normalize histograms by dec band

        sig_h /= np.sum(sig_h, axis=1)[:, None]
        if 'k' not in kwargs:
            kwargs['k'] = 1
        if 's' not in kwargs:
            kwargs['s'] = 0
        if 'ext' not in kwargs:
            kwargs['ext'] = 3
        ratio = sig_h / self._background_sob_map
        for i in range(ratio.shape[0]):
            # Pick out the values we want to use.
            # We explicitly want to avoid NaNs and infinities
            values = ratio[i]
            good = np.isfinite(values) & (values > 0)
            x_good, y_good = bin_centers[good], values[good]

            # Do a linear interpolation across the energy range
            if len(x_good) > 1:
                spline = Spline(x_good, y_good, *args, **kwargs)
                ratio[i] = spline(bin_centers)
            elif len(x_good) == 1:
                ratio[i] = y_good
            else:
                ratio[i] = 0
        return ratio

    def _init_reduced_sim_reconstructed(self, source: sources.Source) -> None:
        """Gets a small simulation dataset to use for injecting signal.

        Prunes the simulation set to only events close to a given source and
        calculate the weight for each event. Adds the weights as a new column
        to the simulation set.

        Args:
            source:

        Returns:
            A reweighted simulation set around the source declination.
        """
        if self._sampling_width is not None:
            self._cut_sim_reconstructed(source)
        else:
            self._reduced_sim_reconstructed = self._sim

    def _cut_sim_reconstructed(self, source: sources.Source) -> np.ndarray:
        """Select simulation events in a reconstruction dec band

        Args:
            source:
        """
        self._edge_point = (np.searchsorted(
                            self._sin_dec_bins,
                            np.sin(source.dec - self.sampling_width)) - 1,
                            np.searchsorted(
                            self._sin_dec_bins,
                            np.sin(source.dec + self.sampling_width)) - 1)
        sindec_dist = np.abs(source.dec - self._sim['dec'])
        close = sindec_dist < self._sampling_width
        self._reduced_sim_reconstructed = self._sim[close].copy()

    def _weight_reduced_sim(self, reduced_sim: np.ndarray) -> np.ndarray:
        """Docstring"""
        reduced_sim = rf.append_fields(reduced_sim, 'weight',
                                       np.zeros(len(reduced_sim)),
                                       dtypes=np.float32)

        # Assign the weights using the newly defined "time profile"
        # classes above. If you want to make this a more complicated
        # shape, talk to me and we can work it out.
        reduced_sim['weight'] = reduced_sim['ow'] * self._spectrum(
            reduced_sim['trueE'])
        return reduced_sim

    def energy_sob(self, events: np.ndarray) -> np.ndarray:
        """Gets the sob vs. gamma required for each event and specific .

        More function info...

        Args:
            events: An array of events including their positional data.

        Returns:
            A list of splines of signal-over-background vs gamma for each event.
        """
        # Get the bin that each event belongs to
        try:
            sin_dec_idx = np.searchsorted(self._sin_dec_bins,
                                          events['sindec']) - 1
        except ValueError:
            sin_dec_idx = np.searchsorted(self._sin_dec_bins,
                                          events['sindec']) - 1
        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1],
                                         events['logE']) - 1
        sin_dec_idx[sin_dec_idx < self._edge_point[0]] = self._edge_point[0]
        # If events fall outside the sampling width, just gonna approxiamte the
        # weight using the nearest non-zero sinDec bin.
        sin_dec_idx[sin_dec_idx > self._edge_point[1]] = self._edge_point[1]
        return self._ratio[sin_dec_idx, log_energy_idx]
