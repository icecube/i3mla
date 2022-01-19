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

from . import utility_functions as uf

import numpy as np

@dataclasses.dataclass
class PointSource:
    """Stores a source object name and location"""
    name: str
    ra: float
    dec: float

    def sample(self, size: int = 1) -> tuple:
        """Sample locations.

        Args:
            size: number of points to sample
        """
        return (np.ones(size) * self.ra, np.ones(size) * self.dec)

    def spatial_pdf(self, events: np.ndarray) -> np.ndarray:
        """calculates the signal probability of events.

        gives a gaussian probability based on their angular distance from the
        source object.

        args:
            source:
            events: an array of events including their positional data.

        returns:
            the value for the signal spatial pdf for the given events angular
            distances.
        """
        sigma2 = events['angErr']**2 + self.sigma**2
        dist = uf.angular_distance(events['ra'], events['dec'], *self.location)
        norm = 1 / (2 * np.pi * sigma2)
        return norm * np.exp(-dist**2 / (2 * sigma2))

    @property
    def location(self) -> tuple:
        """return location of the source"""
        return (self.ra, self.dec)

    @property
    def sigma(self) -> float:
        """return 0 for point source"""
        return 0


@dataclasses.dataclass
class GaussianExtendedSource(PointSource):
    """Gaussian Extended Source"""
    sigma: float
    _sigma: float = dataclasses.field(init=False, repr=False)

    def sample(self, size: int = 1) -> np.ndarray:
        """Sample locations.

        Args:
            size: number of points to sample
        """
        mean = self.location
        cov = self.sigma * np.identity(2)
        return np.random.multivariate_normal(mean, cov, size).T

    @property
    def sigma(self) -> float:
        """return sigma for GaussianExtendedSource"""
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        """Docstring"""
        self._sigma = sigma
