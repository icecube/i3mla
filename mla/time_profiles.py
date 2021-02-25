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

from typing import Callable, Dict, List, Optional, Tuple, Union

import abc
import numpy as np
import scipy.stats


class GenericProfile:
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
    def __init__(self) -> None:
        """Initializes the time profile."""

    @abc.abstractmethod
    def pdf(self, times: np.array) -> np.array:
        """Get the probability amplitude given a time for this time profile.

        Args:
            times: An array of event times to get the probability amplitude for.

        Returns:
            A numpy array of probability amplitudes at the given times.
        """

    @abc.abstractmethod
    def logpdf(self, times: np.array) -> np.array:
        """Get the log(probability) given a time for this time profile.

        Args:
            times: An array of times to get the log(probability) of.

        Returns:
            A numpy array of log(probability) at the given times.
        """

    @abc.abstractmethod
    def random(self, size: int) -> np.array:
        """Get random times sampled from the pdf of this time profile.

        Args:
            size: The number of times to return.

        Returns:
            An array of times.
        """

    @abc.abstractmethod
    def x0(self, times: np.array) -> Tuple:
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
    def bounds(self, time_profile: 'GenericProfile',
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
    def cdf(self, times: np.array) -> np.array:
        """Docstring"""

    @abc.abstractmethod
    def inverse_transform_sample(self, start_times: np.array,
                                 stop_times: np.array) -> np.array:
        """Docstring"""

    @abc.abstractmethod
    def update_params(self, params: np.ndarray) -> None:
        """Docstring"""

    @property
    @abc.abstractmethod
    def exposure(self) -> float:
        """Docstring"""

    @property
    @abc.abstractmethod
    def range(self) -> Tuple[Optional[float], Optional[float]]:
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

    def __init__(self, mean: float, sigma: float) -> None:
        """Initializes the time profile.
        """
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.scipy_dist = scipy.stats.norm(mean, sigma)
        self._range = None, None
        self._default_params = {'mean': mean, 'sigma': sigma}
        self._param_dtype = np.dtype(
            [('mean', np.float32), ('sigma', np.float32)])

    def pdf(self, times: np.array) -> np.array:
        """Calculates the probability for each time.

        Args:
            times: A numpy list of times to evaluate.

        Returns:
            A numpy array of probability amplitudes at the given times.
        """
        return self.scipy_dist.pdf(times)

    def logpdf(self, times: np.array) -> np.array:
        """Calculates the log(probability) for each time.

        Args:
            times: A numpy list of times to evaluate.

        Returns:
            A numpy array of log(probability) at the given times.
        """
        return self.scipy_dist.logpdf(times)

    def random(self, size: int = 1) -> np.array:
        """Returns random values following the gaussian distribution.

        Args:
            size: The number of random values to return.

        Returns:
            An array of times.
        """
        return self.scipy_dist.rvs(size=size)

    def x0(self, times: np.array) -> Tuple[float, float]:
        """Returns good guesses for mean and sigma based on given times.

        Args:
            times: A numpy list of times to evaluate.

        Returns:
            A tuple of mean and sigma guesses.
        """
        x0_mean = np.average(times)
        x0_sigma = np.std(times)
        return x0_mean, x0_sigma

    def bounds(self, time_profile: GenericProfile) -> List[List[float]]:
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

        if None in time_profile.range:
            return [time_profile.range, (0, None)]

        diff = time_profile.range[1] - time_profile.range[0]

        return [time_profile.range, (0, diff)]

    def cdf(self, times: np.array) -> np.array:
        return self.scipy_dist.cdf(times)

    def inverse_transform_sample(self, start_times: np.array,
                                 stop_times: np.array) -> np.array:
        start_cdfs = self.cdf(start_times)
        stop_cdfs = self.cdf(stop_times)
        cdfs = np.random.uniform(start_cdfs, stop_cdfs)
        return self.scipy_dist.ppf(cdfs)

    def update_params(self, params: np.ndarray) -> None:
        """Docstring"""
        self.mean = params['mean']
        self.sigma = params['sigma']

    @property
    def exposure(self) -> float:
        return np.sqrt(2 * np.pi * self.sigma**2)

    @property
    def range(self) -> Tuple[Optional[float], Optional[float]]:
        return self._range

    @property
    def default_params(self) -> Dict[str, float]:
        return self._default_params

    @property
    def param_dtype(self) -> np.dtype:
        return self._param_dtype


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

    def __init__(self, start: float, length: float) -> None:
        """Constructs the time profile.

        Args:
            start: (days) lower bound for the uniform distribution.
            length: (days) length of the uniform distribution.
        """
        super().__init__()
        self._range = (start, start + length)
        self._default_params = {'start': self._range[0], 'length': length}
        self._param_dtype = np.dtype(
            [('start', np.float32), ('length', np.float32)])

    def pdf(self, times: np.array) -> np.array:
        """Calculates the probability for each time.

        Args:
            times: A numpy list of times to evaluate.

        Returns:
            A numpy array of probability amplitudes at the given times.
        """
        output = np.zeros_like(times)
        output[
            (times >= self._range[0]) & (times < self._range[1])
        ] = 1 / (self._range[1] - self._range[0])
        return output

    def logpdf(self, times: np.array) -> np.array:
        """Calculates the log(probability) for each time.

        Args:
            times: A numpy list of times to evaluate.

        Returns:
            A numpy array of log(probability) at the given times.
        """
        return np.log(self.pdf(times))

    def random(self, size: int = 1) -> np.array:
        """Returns random values following the uniform distribution.

        Args:
            size: The number of random values to return.

        Returns:
            An array of times.
        """
        return np.random.uniform(*self._range, size)

    def x0(self, times: np.array) -> Tuple[float, float]:
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

    def cdf(self, times: np.array) -> np.array:
        return np.clip(
            (times - self.range[0]) / (self.range[1] - self.range[0]),
            0,
            1,
        )

    def inverse_transform_sample(self, start_times: np.array,
                                 stop_times: np.array) -> np.array:
        return np.random.uniform(
            np.maximum(start_times, self.range[0]),
            np.minimum(stop_times, self.range[1]),
        )

    def update_params(self, params: np.ndarray) -> None:
        """Docstring"""
        self.start = params['start']
        self.length = params['length']

    @property
    def exposure(self) -> float:
        return self._range[1] - self._range[0]

    @property
    def range(self) -> Tuple[Optional[float], Optional[float]]:
        return self._range

    @property
    def default_params(self) -> Dict[str, float]:
        return self._default_params

    @property
    def param_dtype(self) -> np.dtype:
        return self._param_dtype


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

    def __init__(self, pdf: Callable[[np.array, Tuple[float, float]], np.array],
                 time_range: Tuple[float],
                 bins: Union[List[float], int] = 100,
                 offset: Optional[float] = 0) -> None:
        """Constructs the time profile.

        Args:
            time_range: lower and upper bound for the distribution.
            bins: Either a list of specific bin edges to use (values should be
                between 0 and 1), or an integer giving the number of linear
                spaced bins to use.
            offset: the offset to the time pdf
        """
        super().__init__()
        self._range = (time_range[0] + offset, time_range[1] + offset)
        self._offset = offset
        self._default_params = {'offset': self._offset}
        self._param_dtype = np.dtype([('offset', np.float32)])
        self.dist = self._build_rv(pdf, bins)

    def _build_rv(self,
                  pdf: Callable[[np.array, Tuple[float, float]], np.array],
                  bins: Union[List[float], int]) -> scipy.stats.rv_histogram:
        """Builds a scipy.stats.rv_histogram object for this time profile.

        Args:
            pdf: The normalized distribution function (takes times and time
                window).
            bins: Either a list of specific bin edges to use (values should be
                between 0 and 1), or an integer giving the number of linear
                spaced bins to use.

        Returns:
            The scipy histogram distribution based on the bin edges and the
            distribution function.
        """
        if isinstance(bins, int):
            bin_edges = np.linspace(*self._range, bins)
        else:
            span = self._range[1] - self._range[0]
            bin_edges = span * np.array(bins)

        bin_widths = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + bin_widths
        hist = pdf(bin_centers, tuple(self._range))

        area_under_hist = np.sum(hist * bin_widths)
        hist *= 1 / area_under_hist
        self._exposure = 1 / np.max(hist)
        hist *= bin_widths

        return scipy.stats.rv_histogram((hist, bin_edges))

    def pdf(self, times: np.array) -> np.array:
        """Calculates the probability density for each time.

        Args:
            times: An array of times to evaluate.

        Returns:
            An array of probability densities at the given times.
        """
        return self.dist.pdf(times + self.offset)

    def logpdf(self, times: np.array) -> np.array:
        """Calculates the log(probability) for each time.

        Args:
            times: An array of times to evaluate.

        Returns:
            An array of the log(probability density) for the given times.
        """
        return self.dist.logpdf(times + self.offset)

    def random(self, size: int = 1) -> np.array:
        """Returns random values following the uniform distribution.

        Args:
            size: The number of random values to return.

        Returns:
            An array of random values sampled from the histogram distribution.
        """
        return self.dist.rvs(size=size) + self.offset

    def x0(self, times: np.array) -> Tuple[float, float]:
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

    def cdf(self, times: np.array) -> np.array:
        return self.dist.cdf(times)

    def inverse_transform_sample(self, start_times: np.array,
                                 stop_times: np.array) -> np.array:
        start_cdfs = self.cdf(start_times)
        stop_cdfs = self.cdf(stop_times)
        cdfs = np.random.uniform(start_cdfs, stop_cdfs)
        return self.dist.ppf(cdfs)

    def update_params(self, params: np.ndarray) -> None:
        """Docstring"""
        self.offset = params['offset']

    @property
    def exposure(self) -> float:
        return self._exposure

    @property
    def offset(self) -> float:
        return self._offset

    @offset.setter
    def offset(self, x):
        self._offset = x

    @property
    def range(self) -> Tuple[Optional[float], Optional[float]]:
        return self._range

    @property
    def default_params(self) -> Dict[str, float]:
        return self._default_params

    @property
    def param_dtype(self) -> np.dtype:
        return self._param_dtype
