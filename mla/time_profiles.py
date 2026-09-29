"""
The classes in this file are example time profiles that can be used in the
analysis classes. There is also GenericProfile, an abstract parent class to
inherit from to create other time profiles.
"""

__author__ = 'John Evans and Jason Fan'
__copyright__ = 'Copyright 2024'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '1.4.1'
__maintainer__ = 'Jason Fan'
__email__ = 'klfan@terpmail.umd.edu'
__status__ = 'Development'

from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import abc
import dataclasses

import numpy as np
import scipy.stats

from . import configurable
from .params import Params


@dataclasses.dataclass
class GenericProfile(configurable.Configurable):
    """A generic base class to standardize the methods for the time profiles.

    While I'm only currently using scipy-based
    probability distributions, you can write your own if you
    want. Just be sure to define these methods and ensure that
    the PDF is normalized!

    Attributes:
        exposure (float):
        range (Tuple[Optional[float], Optional[float]]): The range of allowed
            times for for events injected using this time profile.
        default_params (Dict[str, float]): A dictionary of fitting parameters
            for this time profile.
        param_dtype (List[Tuple[str, str]]): The numpy dytpe for the fitting
            parameters.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def pdf(self, times: np.ndarray) -> np.ndarray:
        """Get the probability amplitude given a time for this time profile.

        Args:
            times: An array of event times to get the probability amplitude.

        Returns:
            A numpy array of probability amplitudes at the given times.
        """

    @abc.abstractmethod
    def logpdf(self, times: np.ndarray) -> np.ndarray:
        """Get the log(probability) given a time for this time profile.

        Args:
            times: An array of times to get the log(probability) of.

        Returns:
            A numpy array of log(probability) at the given times.
        """

    @abc.abstractmethod
    def random(self, size: int) -> np.ndarray:
        """Get random times sampled from the pdf of this time profile.

        Args:
            size: The number of times to return.

        Returns:
            An array of times.
        """

    @abc.abstractmethod
    def x0(self, times: np.ndarray) -> Tuple:
        """Gets a tuple of initial guess to use when fitting parameters.

        The guesses are arrived at by simple approximations using the given
        times.

        Args:
            times: An array of times to use to approximate the fitting
                parameters of this time profile.

        Returns:
            A tuple of approximate parameters.
        """

    @abc.abstractmethod
    def bounds(
        self,
        time_profile: 'GenericProfile',
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        """Get a list of tuples of bounds for the parameters of this profile.

        Uses another time profile to constrain the bounds. This is usually
        needed to constrain the bounds of a signal time profile given a
        background time profile.

        Args:
            time_profile: Another time profile to constrain from.

        Returns:
            A list of tuples of bounds for fitting the  parameters of this time
            profile.
        """

    @abc.abstractmethod
    def cdf(self, times: np.ndarray) -> np.ndarray:
        """Docstring"""

    @abc.abstractmethod
    def inverse_transform_sample(
            self,
            start_times: np.ndarray,
            stop_times: np.ndarray) -> np.ndarray:
        """Docstring"""

    @property
    @abc.abstractmethod
    def params(self) -> dict:
        """Docstring"""

    @params.setter
    @abc.abstractmethod
    def params(self, params: Params) -> None:
        """Docstring"""

    @property
    @abc.abstractmethod
    def param_bounds(self) -> dict:
        """Docstring"""

    @property
    @abc.abstractmethod
    def exposure(self) -> float:
        """Docstring"""

    @property
    @abc.abstractmethod
    def range(self) -> Tuple[float, float]:
        """Gets the maximum and minimum values for the times in this profile.
        """

    @property
    @abc.abstractmethod
    def default_params(self) -> Dict[str, float]:
        """Returns the initial parameters formatted for ts calculation output.
        """

    @property
    @abc.abstractmethod
    def param_dtype(self) -> np.dtype:
        """Returns the parameter names and datatypes
           formatted for numpy dtypes.
        """


@dataclasses.dataclass
class GaussProfile(GenericProfile):
    """Time profile class for a gaussian distribution.

    Use this to produce gaussian-distributed times for your source.

    Attributes:
        mean (float): The center of the distribution.
        sigma (float): The spread of the distribution.
        scipy_dist(scipy.stats.rv_continuous):
        exposure (float):
        range (Tuple[Optional[float], Optional[float]]): The range of allowed
            times for for events injected using this time profile.
        default_params (Dict[str, float]): A dictionary of fitting parameters
            for this time profile.
        param_dtype (List[Tuple[str, str]]): The numpy dytpe for the fitting
            parameters.
    """
    scipy_dist: scipy.stats.distributions.rv_frozen = dataclasses.field(
        init=False)
    _mean: float = dataclasses.field(init=False, repr=False)
    _sigma: float = dataclasses.field(init=False, repr=False)
    _param_dtype: ClassVar[np.dtype] = np.dtype(
        [('mean', np.float32), ('sigma', np.float32)])

    def __post_init__(self) -> None:
        """Initializes the time profile."""
        self._mean = self.config['mean']
        self._sigma = self.config['sigma']
        self.scipy_dist = scipy.stats.norm(self._mean, self._sigma)

    def pdf(self, times: np.ndarray) -> np.ndarray:
        """Calculates the probability for each time.

        Args:
            times: A numpy list of times to evaluate.

        Returns:
            A numpy array of probability amplitudes at the given times.
        """
        return self.scipy_dist.pdf(times)

    def logpdf(self, times: np.ndarray) -> np.ndarray:
        """Calculates the log(probability) for each time.

        Args:
            times: A numpy list of times to evaluate.

        Returns:
            A numpy array of log(probability) at the given times.
        """
        return self.scipy_dist.logpdf(times)

    def random(self, size: int = 1) -> np.ndarray:
        """Returns random values following the gaussian distribution.

        Args:
            size: The number of random values to return.

        Returns:
            An array of times.
        """
        return self.scipy_dist.rvs(size=size)

    def x0(self, times: np.ndarray) -> tuple:
        """Returns good guesses for mean and sigma based on given times.

        Args:
            times: A numpy list of times to evaluate.

        Returns:
            A tuple of mean and sigma guesses.
        """
        x0_mean = np.average(times)
        x0_sigma = np.std(times)
        return x0_mean, x0_sigma

    def bounds(self, time_profile: GenericProfile) -> List[tuple]:
        """Returns good bounds for this time profile
           given another time profile.

        Limits the mean to be within the range of the other profile and limits
        the sigma to be >= 0 and <= the width of the other profile.

        Args:
            time_profile: Another time profile to use to define the parameter
                bounds of this time profile.

        Returns:
            A list of tuples of bounds for fitting the parameters in this time
            profile.
        """
        if np.nan in time_profile.range:
            return [time_profile.range, (0, np.nan)]

        diff = time_profile.range[1] - time_profile.range[0]
        return [time_profile.range, (0, diff)]

    def cdf(self, times: np.ndarray) -> np.ndarray:
        """Docstring"""
        return self.scipy_dist.cdf(times)

    def inverse_transform_sample(
            self,
            start_times: np.ndarray,
            stop_times: np.ndarray) -> np.ndarray:
        """Docstring"""
        start_cdfs = self.cdf(start_times)
        stop_cdfs = self.cdf(stop_times)
        cdfs = np.random.uniform(start_cdfs, stop_cdfs)
        return self.scipy_dist.ppf(cdfs)

    @property
    def params(self) -> dict:
        """Docstring"""
        return {'mean': self._mean, 'sigma': self._sigma}

    @params.setter
    def params(self, params: Params) -> None:
        """Docstring"""
        update = False

        if 'mean' in params:
            self._mean = params['mean']
            update = True
        if 'sigma' in params:
            self._sigma = params['sigma']
            update = True

        if update:
            self.scipy_dist = scipy.stats.norm(self._mean, self._sigma)

    @property
    def param_bounds(self) -> dict:
        return {'mean': self.range, 'sigma': (0, np.inf)}

    @property
    def exposure(self) -> float:
        return np.sqrt(2 * np.pi * self._sigma**2)

    @property
    def range(self) -> Tuple[float, float]:
        return -np.inf, np.inf

    @property
    def param_dtype(self) -> np.dtype:
        return self._param_dtype

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['mean'] = np.nan
        config['sigma'] = np.nan
        return config


@dataclasses.dataclass
class UniformProfile(GenericProfile):
    """Time profile class for a uniform distribution.

    Use this for background or if you want to assume a steady signal from
    your source.

    Attributes:
        exposure (float):
        range (Tuple[Optional[float], Optional[float]]): The range of allowed
            times for for events injected using this time profile.
        default_params (Dict[str, float]): A dictionary of fitting parameters
            for this time profile.
        param_dtype (List[Tuple[str, str]]): The numpy dytpe for the fitting
            parameters.
    """
    _range: Tuple[float, float] = dataclasses.field(init=False, repr=False)
    _param_dtype: ClassVar[np.dtype] = np.dtype(
        [('start', np.float32), ('length', np.float32)])

    def __post_init__(self) -> None:
        """Constructs the time profile."""
        self._range = (
            self.config['start'],
            self.config['start']
            + self.config['length'])

    def pdf(self, times: np.ndarray) -> np.ndarray:
        """Calculates the probability for each time.

        Args:
            times: A numpy list of times to evaluate.

        Returns:
            A numpy array of probability amplitudes at the given times.
        """
        output = np.zeros_like(times)
        output[
            (times >= self.range[0]) & (times < self.range[1])
        ] = 1 / (self.range[1] - self.range[0])
        return output

    def logpdf(self, times: np.ndarray) -> np.ndarray:
        """Calculates the log(probability) for each time.

        Args:
            times: A numpy list of times to evaluate.

        Returns:
            A numpy array of log(probability) at the given times.
        """
        return np.log(self.pdf(times))

    def random(self, size: int = 1) -> np.ndarray:
        """Returns random values following the uniform distribution.

        Args:
            size: The number of random values to return.

        Returns:
            An array of times.
        """
        return np.random.uniform(*self.range, size)

    def x0(self, times: np.ndarray) -> Tuple[float, float]:
        """Returns good guesses for start and stop based on given times.

        Args:
            times: A numpy list of times to evaluate.

        Returns:
            A tuple of start and stop guesses.
        """
        x0_start = np.min(times)
        x0_end = np.max(times)
        return x0_start, x0_end

    def bounds(self, time_profile: GenericProfile
               ) -> List[Tuple[Optional[float], Optional[float]]]:
        """Given some other profile, returns allowable ranges for parameters.

        Args:
            time_profile: Another time profile used to get the limits of start
                and length.

        Returns:
            A list of tuples of bounds for fitting.
        """
        diff = time_profile.range[1] - time_profile.range[0]
        return [time_profile.range, (0, diff)]

    def cdf(self, times: np.ndarray) -> np.ndarray:
        """Docstring"""
        return np.clip(
            (times - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)

    def inverse_transform_sample(
            self,
            start_times: np.ndarray,
            stop_times: np.ndarray) -> np.ndarray:
        """Docstring"""
        return np.random.uniform(
            np.maximum(start_times, self.range[0]),
            np.minimum(stop_times, self.range[1]),
        )

    @property
    def params(self) -> dict:
        """Docstring"""
        return {
            'start': self._range[0],
            'length': self._range[1] - self._range[0]}

    @params.setter
    def params(self, params: Params) -> None:
        """Docstring"""
        if 'start' in params:
            self._range = (
                params['start'],
                params['start']
                + self._range[1]
                - self._range[0])
        if 'length' in params:
            self._range = (self._range[0], self._range[0] + params['length'])

    @property
    def param_bounds(self) -> dict:
        return {'start': (-np.inf, np.inf), 'length': (0, np.inf)}

    @property
    def exposure(self) -> float:
        return self._range[1] - self._range[0]

    @property
    def range(self) -> Tuple[float, float]:
        return self._range

    @property
    def param_dtype(self) -> np.dtype:
        return self._param_dtype

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['start'] = np.nan
        config['length'] = np.nan
        return config


@dataclasses.dataclass
class CustomProfile(GenericProfile):
    """Time profile class for a custom binned distribution.

    This time profile uses a binned pdf defined between 0 and 1. Normalization
    is handled internally and not required beforehand.

    Attributes:
        pdf (Callable[[np.array, Tuple[float, float]], np.array]): The
            distribution function.
            This function needs to accept an array of bin
            centers and a time window as a tuple, and it needs to return an
            array of probability densities at the given bin centers.
        dist (scipy.stats.rv_histogram): The histogrammed version of the
            distribution function.
        exposure (float):
        range (Tuple[Optional[float], Optional[float]]): The range of allowed
            times for for events injected using this time profile.
        default_params (Dict[str, float]): A dictionary of fitting parameters
            for this time profile.
        param_dtype (List[Tuple[str, str]]): The numpy dytpe for the fitting
            parameters.
    """
    dist: Callable[[np.ndarray, Tuple[float, float]], np.ndarray]
    _dist: scipy.stats.rv_histogram = dataclasses.field(init=False, repr=False)
    _offset: float = dataclasses.field(init=False, repr=False)
    _exposure: float = dataclasses.field(init=False, repr=False)
    _param_dtype: ClassVar[np.dtype] = np.dtype([('offset', np.float32)])

    @property
    def dist(self) -> scipy.stats.rv_histogram:
        """Docstring"""
        return self._dist

    @dist.setter
    def dist(self, _) -> None:
        """Constructs the time profile.

        Args:
            dist:
        """
        self._offset = self.config['offset']

        if isinstance(self.config['bins'], int):
            bin_edges = np.linspace(*self.config['range'],
                                    self.config['bins'] + 1)
        else:
            bin_edges = np.array(self.config['bins'])
        bin_widths = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + bin_widths / 2

        hist, _ = np.histogram(bin_centers, bins=bin_edges)
        hist = hist.astype(float)
        area_under_hist = np.sum(hist * bin_widths)
        hist *= 1. / area_under_hist
        self._exposure = 1 / np.max(hist)
        hist *= bin_widths
        self._dist = scipy.stats.rv_histogram((hist, bin_edges))

    def pdf(self, times: np.ndarray) -> np.ndarray:
        """Calculates the probability density for each time.

        Args:
            times: An array of times to evaluate.

        Returns:
            An array of probability densities at the given times.
        """
        return self.dist.pdf(times + self.offset)

    def logpdf(self, times: np.ndarray) -> np.ndarray:
        """Calculates the log(probability) for each time.

        Args:
            times: An array of times to evaluate.

        Returns:
            An array of the log(probability density) for the given times.
        """
        return self.dist.logpdf(times + self.offset)

    def random(self, size: int = 1) -> np.ndarray:
        """Returns random values following the uniform distribution.

        Args:
            size: The number of random values to return.

        Returns:
            An array of random values sampled from the histogram distribution.
        """
        return self.dist.rvs(size=size) + self.offset

    def x0(self, times: np.ndarray) -> Tuple[float, float]:
        """Gives a guess of the parameters of this type of time profile.

        Args:
            times: An array of times to use to guess the parameters.

        Returns:
            The guessed start and end times of the distribution that generated
            the given times.
        """
        x0_start = np.min(times)
        x0_end = np.max(times)
        return x0_start, x0_end

    def bounds(self, time_profile: GenericProfile
               ) -> List[Tuple[Optional[float], Optional[float]]]:
        """Given some other profile, returns allowable ranges for parameters.

        Args:
            time_profile: Another time profile to use to get the bounds.

        Returns:
            The fitting bounds for the parameters of this time profile.
        """
        return [time_profile.range, time_profile.range]

    def cdf(self, times: np.ndarray) -> np.ndarray:
        """Docstring"""
        return self.dist.cdf(times)

    def inverse_transform_sample(
            self,
            start_times: np.ndarray,
            stop_times: np.ndarray) -> np.ndarray:
        """Docstring"""
        start_cdfs = self.cdf(start_times)
        stop_cdfs = self.cdf(stop_times)
        cdfs = np.random.uniform(start_cdfs, stop_cdfs)
        return self.dist.ppf(cdfs)

    @property
    def params(self) -> dict:
        """Docstring"""
        return {'offset': self.offset}

    @params.setter
    def params(self, params: Params) -> None:
        """Docstring"""
        if 'offset' in params:
            self.offset = params['offset']

    @property
    def param_bounds(self) -> dict:
        return {'offset': (-np.inf, np.inf)}

    @property
    def exposure(self) -> float:
        return self._exposure

    @property
    def offset(self) -> float:
        return self._offset

    @offset.setter
    def offset(self, offset: float) -> None:
        self._offset = offset

    @property
    def range(self) -> Tuple[Optional[float], Optional[float]]:
        return (
            self.config['range'][0]
            + self.offset,
            self.config['range'][1]
            + self.offset)

    @property
    def param_dtype(self) -> np.dtype:
        return self._param_dtype

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['range'] = (np.nan, np.nan)
        config['bins'] = 100
        config['offset'] = 0
        return config

@dataclasses.dataclass
class MultiFlareProfile(GenericProfile):
    """A fittable mixture of N box-shaped ("flare") time windows.

    Each flare i is a real UniformProfile box [start_i, start_i+length_i)
    with a relative weight_i. The signal time PDF is the weighted mixture:

        pdf(t) = sum_i  (weight_i / sum_j weight_j) * box_i.pdf(t)

    Unlike the fixed-shape version of this class, start_i, length_i, and
    weight_i for every flare are exposed as free parameters through
    `params`/`param_bounds`, so the analysis minimizer can search over
    where each flare sits, how wide it is, and how much it contributes,
    given a fixed number of flares `n_flares` chosen up front (n_flares
    itself is a hyperparameter, not something that gets fit).

    Because pdf(t) only depends on the *relative* weights, the absolute
    scale of the weight parameters is unconstrained (multiplying every
    weight_i by the same constant leaves pdf unchanged). This flat
    direction is harmless for a grid-search minimizer but can slow down
    gradient-based ones; if that's an issue, fix one flare's weight (e.g.
    via param_bounds) rather than leaving all of them free.

    Use this same class for both roles:
      - As the *fit* signal_time_profile passed to TimeTermFactory: its
        params get updated every minimizer iteration via the params
        setter.
      - As the *injection truth* passed to data_handler.signal_time_profile:
        just don't call the params setter on it after construction, and
        its box positions/weights stay fixed while `.random()`/
        `.inverse_transform_sample()` draw injected event times from it.

    Attributes:
        n_flares: Number of box flares in the mixture (fixed, not fit).
        search_range: (start, stop) -- the overall time window flares are
            allowed to move within. Used for parameter bounds and as this
            profile's own `.range`.
    """

    n_flares: int = dataclasses.field(default=None)
    search_range: Tuple[float, float] = dataclasses.field(default=None)
    _boxes: List['UniformProfile'] = dataclasses.field(
        init=False, repr=False)
    _weights: np.ndarray = dataclasses.field(init=False, repr=False)
    _param_dtype: np.dtype = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initializes flare boxes (as real UniformProfile instances) and
        each flare's neutrino-count weight."""
        if self.n_flares is None:
            self.n_flares = self.config['n_flares']
        if self.search_range is None:
            self.search_range = self.config['search_range']
        if self.n_flares < 1:
            raise ValueError('n_flares must be >= 1.')

        start, stop = self.search_range
        span = stop - start

        init_starts = self.config.get('init_starts')
        init_lengths = self.config.get('init_lengths')
        init_weights = self.config.get('init_weights')

        if init_starts is None:
            # Spread flares evenly across the search range as a starting
            # guess for the minimizer.
            centers = np.linspace(
                start + span / (2 * self.n_flares),
                stop - span / (2 * self.n_flares),
                self.n_flares,
            )
            default_length = span / (2 * self.n_flares)
            init_starts = centers - default_length / 2
        if init_lengths is None:
            init_lengths = np.full(
                self.n_flares, span / (2 * self.n_flares))
        if init_weights is None:
            init_weights = np.ones(self.n_flares) / self.n_flares

        if not (len(init_starts) == len(init_lengths)
                == len(init_weights) == self.n_flares):
            raise ValueError(
                'init_starts, init_lengths, init_weights (if given) must '
                'all have length n_flares.')

        self._boxes = [
            UniformProfile(config={
                **UniformProfile.generate_config(),
                'start': float(init_starts[i]),
                'length': float(init_lengths[i]),
            })
            for i in range(self.n_flares)
        ]
        self._weights = np.array(init_weights, dtype=float)

        self._param_dtype = np.dtype([
            (f'flare{i}_{name}', np.float32)
            for i in range(self.n_flares)
            for name in ('start', 'length', 'weight')
        ])

    # -- internal helper ---------------------------------------------------

    def _norm_weights(self) -> np.ndarray:
        total = self._weights.sum()
        if total <= 0:
            return np.ones(self.n_flares) / self.n_flares
        return self._weights / total

    # -- required GenericProfile interface --------------------------------

    def pdf(self, times: np.ndarray) -> np.ndarray:
        """Weighted sum of each flare's own UniformProfile.pdf."""
        times = np.asarray(times, dtype=float)
        weights = self._norm_weights()
        total = np.zeros_like(times, dtype=float)
        for i, weight in enumerate(weights):
            total += weight * self._boxes[i].pdf(times)
        return total

    def logpdf(self, times: np.ndarray) -> np.ndarray:
        """Log(probability), matching the convention used by
        UniformProfile.logpdf elsewhere in this file (plain log of pdf,
        -inf at zero-probability times). TimeTerm.sob uses .pdf() in
        linear space, not this -- exists to satisfy the interface."""
        return np.log(self.pdf(times))

    def random(self, size: int = 1) -> np.ndarray:
        """Draws random times: pick a flare per event (weighted), then
        draw from that flare's own UniformProfile.random."""
        weights = self._norm_weights()
        flare_idx = np.random.choice(self.n_flares, size=size, p=weights)
        out = np.empty(size, dtype=float)
        for i in range(self.n_flares):
            mask = flare_idx == i
            n = int(mask.sum())
            if n > 0:
                out[mask] = self._boxes[i].random(size=n)
        return out

    def x0(self, times: np.ndarray) -> Tuple:
        """Guesses flare parameters by spreading n_flares boxes evenly
        across the observed times, with equal weight."""
        times = np.asarray(times, dtype=float)
        if len(times) == 0:
            lo, hi = self.search_range
        else:
            lo, hi = np.min(times), np.max(times)
        span = hi - lo
        centers = np.linspace(
            lo + span / (2 * self.n_flares),
            hi - span / (2 * self.n_flares),
            self.n_flares,
        )
        length = span / (2 * self.n_flares) if self.n_flares > 0 else span
        starts = centers - length / 2
        weights = np.ones(self.n_flares) / self.n_flares
        params = []
        for i in range(self.n_flares):
            params.extend([starts[i], length, weights[i]])
        return tuple(params)

    def bounds(
        self,
        time_profile: 'GenericProfile',
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        """Bounds for (start, length, weight) per flare, constrained by
        another profile's (e.g. the background's) time range."""
        rng = time_profile.range
        span = rng[1] - rng[0]
        out = []
        for _ in range(self.n_flares):
            out.append(rng)              # start
            out.append((0, span))        # length
            out.append((0, 1))           # weight
        return out

    def cdf(self, times: np.ndarray) -> np.ndarray:
        """Weighted sum of each flare's own UniformProfile.cdf."""
        times = np.asarray(times, dtype=float)
        weights = self._norm_weights()
        total = np.zeros_like(times, dtype=float)
        for i, weight in enumerate(weights):
            total += weight * self._boxes[i].cdf(times)
        return total

    def inverse_transform_sample(
        self,
        start_times: np.ndarray,
        stop_times: np.ndarray,
    ) -> np.ndarray:
        """Per-event: pick a flare *conditioned on how much of that flare
        overlaps [start_time, stop_time]*, then draw from that flare's
        own UniformProfile.inverse_transform_sample.

        This is called by TimeDependentNuSourcesDataHandler._randomize_times
        with (run_start, run_stop) for a GRL run already chosen because
        the *mixture* has probability mass there. Picking the flare with
        the unconditional (global) weights instead of conditioning on
        overlap with that specific run would let the draw pick a flare
        that doesn't overlap the chosen run at all, collapsing to a
        degenerate zero-width sample at the run boundary. Conditioning on
        overlap avoids that.
        """
        start_times = np.asarray(start_times, dtype=float)
        stop_times = np.asarray(stop_times, dtype=float)
        n = len(start_times)
        weights = self._norm_weights()

        # Probability mass each flare contributes within each event's own
        # [start_time, stop_time] window.
        masses = np.stack([
            weights[i] * (
                self._boxes[i].cdf(stop_times)
                - self._boxes[i].cdf(start_times)
            )
            for i in range(self.n_flares)
        ], axis=0)  # shape (n_flares, n)
        totals = masses.sum(axis=0)
        totals_safe = np.where(totals > 0, totals, 1.0)
        probs = masses / totals_safe  # shape (n_flares, n), columns sum to 1

        cumulative = np.cumsum(probs, axis=0)
        draws = np.random.uniform(size=n)
        flare_idx = (draws[None, :] < cumulative).argmax(axis=0)

        out = np.empty(n, dtype=float)
        for i in range(self.n_flares):
            mask = flare_idx == i
            if not np.any(mask):
                continue
            out[mask] = self._boxes[i].inverse_transform_sample(
                start_times[mask], stop_times[mask])
        return out

    @property
    def params(self) -> dict:
        """Flat dict of every flare's start/length/weight."""
        out = {}
        for i, box in enumerate(self._boxes):
            box_params = box.params  # {'start': ..., 'length': ...}
            out[f'flare{i}_start'] = box_params['start']
            out[f'flare{i}_length'] = box_params['length']
            out[f'flare{i}_weight'] = self._weights[i]
        return out

    @params.setter
    def params(self, params) -> None:
        """Updates whichever flare params are present in `params`,
        forwarding start/length updates to each flare's own
        UniformProfile.params setter (which correctly handles start and
        length being updated together, regardless of dict key order)."""
        for i, box in enumerate(self._boxes):
            box_update = {}
            if f'flare{i}_start' in params:
                box_update['start'] = params[f'flare{i}_start']
            if f'flare{i}_length' in params:
                box_update['length'] = params[f'flare{i}_length']
            if box_update:
                box.params = box_update
            if f'flare{i}_weight' in params:
                self._weights[i] = params[f'flare{i}_weight']

    @property
    def param_bounds(self) -> dict:
        """Flat dict of bounds matching `params`' keys."""
        rng = self.search_range
        span = rng[1] - rng[0]
        min_len = self.config.get('min_box_length', 0)
        max_len = self.config.get('max_box_length', span)
        out = {}
        for i in range(self.n_flares):
            out[f'flare{i}_start'] = rng
            out[f'flare{i}_length'] = (min_len, max_len)
            out[f'flare{i}_weight'] = (0, 1)
        return out

    @property
    def exposure(self) -> float:
        """Weighted-average box length (approximate effective exposure;
        exact only for non-overlapping, equal-weight boxes). Reuses
        UniformProfile.exposure (= box length) directly."""
        weights = self._norm_weights()
        lengths = np.array([box.exposure for box in self._boxes])
        return float(np.sum(weights * lengths))

    @property
    def range(self) -> Tuple[float, float]:
        """The fixed overall search range (does not shrink/grow as the
        fit moves individual flares)."""
        return self.search_range

    @property
    def default_params(self) -> dict:
        """Initial parameter values, for ts-output/logging purposes."""
        return self.params

    @property
    def param_dtype(self) -> np.dtype:
        return self._param_dtype

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['n_flares'] = None
        config['search_range'] = None
        config['init_starts'] = None
        config['init_lengths'] = None
        config['init_weights'] = None
        config['min_box_length'] = 0
        config['max_box_length'] = None  # defaults to full search span
        return config