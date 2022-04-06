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

from . import configurable
from . import utility_functions as uf

import numpy as np


@dataclasses.dataclass
class PointSource(configurable.Configurable):
    """Stores a source object name and location"""
    def sample(self, size: int = 1) -> tuple:
        """Sample locations.

        Args:
            size: number of points to sample
        """
        return (np.ones(size) * self.config['ra'], np.ones(size) * self.config['dec'])

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
        return (self.config['ra'], self.config['dec'])

    @property
    def sigma(self) -> float:
        """return 0 for point source"""
        return 0

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['name'] = 'source_name'
        config['ra'] = np.nan
        config['dec'] = np.nan
        return config


@dataclasses.dataclass
class GaussianExtendedSource(PointSource):
    """Gaussian Extended Source"""
    def sample(self, size: int = 1) -> np.ndarray:
        """Sample locations.

        Args:
            size: number of points to sample
        """
        mean = self.location
        x = np.random.normal(mean[0], self.sigma, size=size)
        y = np.random.normal(mean[1], self.sigma, size=size)
        return np.array([x,y])

    @property
    def sigma(self) -> float:
        """return sigma for GaussianExtendedSource"""
        return self.config['sigma']

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['sigma'] = np.deg2rad(1)
        return config
