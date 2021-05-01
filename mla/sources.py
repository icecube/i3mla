"""
Docstring
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'


import dataclasses

import numpy as np


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


@dataclasses.dataclass
class Source:
    """Stores a source object name and location"""
    name: str
    ra: float
    dec: float

    def sample_location(self, size: int):
        """Sample locations.

        Args:
            size: number of points to sample
        """
        return (np.ones(size) * self.ra, np.ones(size) * self.dec)

    def signal_spatial_pdf(self,
                           events: np.ndarray) -> np.array:
        """Calculates the signal probability of events.

        Gives a gaussian probability based on their angular distance from the
        source object.

        Args:
            events: An array of events including their positional data.

        Returns:
            The value for the signal spatial pdf for the given events angular
            distances.
        """
        sigma = events['angErr']
        dist = angular_distance(events['ra'], events['dec'], self.ra,
                                self.dec)
        norm = 1 / (2 * np.pi * sigma**2)
        return norm * np.exp(-dist**2 / (2 * sigma**2))


@dataclasses.dataclass
class GaussianExtendedSource(Source):
    """Gaussian Extended Source"""
    sigma: float

    def sample_location(self, size: int):
        """Sample locations.

        Args:
            size: number of points to sample
        """
        return (np.random.normal(self.ra, self.sigma, size),
                np.random.normal(self.dec, self.sigma, size))

    def signal_spatial_pdf(self,
                           events: np.ndarray) -> np.array:
        """signal probability of events for GaussianExtendedSource.

        Gives a gaussian probability based on their angular distance from the
        source object.

        Args:
            events: An array of events including their positional data.

        Returns:
            The value for the signal spatial pdf for the given events angular
            distances.
        """
        sigma = events['angErr'] + self.sigma
        dist = angular_distance(events['ra'], events['dec'], self.ra,
                                self.dec)
        norm = 1 / (2 * np.pi * sigma**2)
        return norm * np.exp(-dist**2 / (2 * sigma**2))


def ra_to_rad(hrs: float, mins: float, secs: float) -> float:
    """Converts right ascension to radians.

    Args:
        hrs: Hours.
        mins: Minutes.
        secs: Seconds.

    Returns:
        Radian representation of right ascension.
    """
    return (hrs * 15 + mins / 4 + secs / 240) * np.pi / 180


def dec_to_rad(sign: int, deg: float, mins: float, secs: float) -> float:
    """Converts declination to radians.

    Args:
        sign: A positive integer for a positive sign, a negative integer for a
            negative sign.
        deg: Degrees.
        mins: Minutes.
        secs: Seconds.

    Returns:
        Radian representation of declination.
    """
    return sign / np.abs(sign) * (deg + mins / 60 + secs / 3600) * np.pi / 180
