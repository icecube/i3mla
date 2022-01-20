"""
The classes in this file are example time profiles that can be used in the
analysis classes. There is also GenericProfile, an abstract parent class to
inherit from to create other time profiles.
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
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
            times: An array of event times to get the probability amplitude for.

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
        self, start_times: np.ndarray, stop_times: np.ndarray) -> np.ndarray:
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
        """Returns the parameter names and datatypes formatted for numpy dtypes.
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
    _param_dtype: ClassVar[np.dtype] = np.dtype(
        [('mean', np.float32), ('sigma', np.float32)])
    scipy_dist: scipy.stats.distributions.rv_frozen = dataclasses.field(init=False)
    _mean: float = dataclasses.field(init=False, repr=False)
    _sigma: float = dataclasses.field(init=False, repr=False)
    _param_dtype: np.dtype = dataclasses.field(init=False, repr=False)

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
        """Returns good bounds for this time profile given another time profile.

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
        self, start_times: np.ndarray, stop_times: np.ndarray) -> np.ndarray:
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
    _param_dtype: ClassVar[np.dtype] = np.dtype(
        [('start', np.float32), ('length', np.float32)])
    _range: Tuple[float, float] = dataclasses.field(init=False, repr=False)
    _param_dtype: np.dtype = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Constructs the time profile."""
        self._range = (self.config['start'], self.config['start'] + self.config['length'])

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
        return np.clip((times - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)

    def inverse_transform_sample(
            self, start_times: np.ndarray, stop_times: np.ndarray) -> np.ndarray:
        """Docstring"""
        return np.random.uniform(
            np.maximum(start_times, self.range[0]),
            np.minimum(stop_times, self.range[1]),
        )

    @property
    def params(self) -> dict:
        """Docstring"""
        return {'start': self._range[0], 'length': self._range[1] - self._range[0]}

    @params.setter
    def params(self, params: Params) -> None:
        """Docstring"""
        if 'start' in params:
            self._range = (
                params['start'], params['start'] + self._range[1] - self._range[0])
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
            distribution function. This function needs to accept an array of bin
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
    _param_dtype: ClassVar[np.dtype] = np.dtype([('offset', np.float32)])
    pdf_func: dataclasses.InitVar[Callable[[np.ndarray, Tuple[float, float]], np.ndarray]]
    dist: scipy.stats.rv_histogram = dataclasses.field(init=False, repr=False)
    _offset: float = dataclasses.field(init=False, repr=False)
    _exposure: float = dataclasses.field(init=False, repr=False)

    def __post_init__(
        self, pdf_func: Callable[[np.ndarray, Tuple[float, float]], np.ndarray]) -> None:
        """Constructs the time profile.

        Args:
            pdf_func:
        """
        self._offset = self.config['offset']

        if isinstance(self.config['bins'], int):
            bin_edges = np.linspace(*self.config['range'], self.config['bins'])
        else:
            span = self.config['range'][1] - self.config['range'][0]
            bin_edges = span * np.array(self.config['bins'])

        bin_widths = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + bin_widths
        hist = pdf_func(bin_centers, tuple(self.config['range']))

        area_under_hist = np.sum(hist * bin_widths)
        hist *= 1 / area_under_hist
        self._exposure = 1 / np.max(hist)
        hist *= bin_widths

        self.dist = scipy.stats.rv_histogram((hist, bin_edges))

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
        self, start_times: np.ndarray, stop_times: np.ndarray) -> np.ndarray:
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
            self.config['range'][0] + self.offset, self.config['range'][1] + self.offset)

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
