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
from typing import ClassVar, Tuple

from .configurable import Configurable
from . import utility_functions as uf

import numpy as np


@dataclasses.dataclass(kw_only=True)
class PointSource(Configurable):
    """Stores a source object name and location"""
    _config_map: ClassVar[dict] = {
        'name': ('Source Name', 'source_name'),
        '_ra': ('Right Ascension (rad)', np.nan),
        '_dec': ('Declination (rad)', np.nan),
    }

    name: str = 'source_name'
    _ra: float
    _dec: float

    @classmethod
    def from_config(cls, config: dict) -> 'PointSource':
        """Docstring"""
        return cls(**cls._map_kwargs(config))

    def sample(self, size: int = 1) -> tuple:
        """Sample locations.

        Args:
            size: number of points to sample
        """
        return (np.ones(size) * self._ra, np.ones(size) * self._dec)

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
    def location(self) -> Tuple[float, float]:
        """return location of the source"""
        return (self._ra, self._dec)

    @property
    def sigma(self) -> float:
        """return 0 for point source"""
        return 0


@dataclasses.dataclass(kw_only=True)
class GaussianExtendedSource(PointSource, Configurable):
    """Gaussian Extended Source"""
    _config_map: ClassVar[dict] = {
        **PointSource._config_map,
        '_sigma': ('Sigma (rad)', np.deg2rad(1)),
    }

    sigma: float = np.deg2rad(1)

    @classmethod
    def from_config(cls, config: dict) -> 'GaussianExtendedSource':
        """Docstring"""
        return cls(**cls._map_kwargs(config))

    def sample(self, size: int = 1) -> np.ndarray:
        """Sample locations.

        Args:
            size: number of points to sample
        """
        mean = self.location
        x = np.random.normal(mean[0], self.sigma, size=size)
        y = np.random.normal(mean[1], self.sigma, size=size)
        return np.array([x, y])
