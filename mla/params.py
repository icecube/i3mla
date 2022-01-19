"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List, Optional

import dataclasses
import numpy as np
import numpy.lib.recfunctions as rf


@dataclasses.dataclass
class Params:
    """Docstring"""
    values: np.ndarray
    key_idx_map: dict
    bounds: dict

    def __contains__(self, item: str) -> bool:
        """Docstring"""
        return item in self.key_idx_map

    def __getitem__(self, key: str):
        """Docstring"""
        return self.values[self.key_idx_map[key]]

    @property
    def names(self) -> List[str]:
        """Docstring"""
        return [*self.key_idx_map]

    @classmethod
    def from_dict(cls, value_dict: dict, bounds: Optional[dict] = None) -> 'Params':
        """Docstring"""
        values = np.array(value_dict.values)
        key_idx_map = {key: i for i, key in enumerate(value_dict)}
        return cls._build_params(values, key_idx_map, bounds)

    @classmethod
    def from_array(
        cls, value_array: np.ndarray, bounds: Optional[dict] = None) -> 'Params':
        """Docstring"""
        values = rf.structured_to_unstructured(value_array, copy=True)[0]
        key_idx_map = {name: i for i, name in enumerate(value_array.dtype.names)}
        return cls._build_params(values, key_idx_map, bounds)

    @classmethod
    def _build_params(
        cls, values: np.ndarray, key_idx_map: dict, bounds: Optional[dict]) -> 'Params':
        """Docstring"""
        if bounds is None:
            bounds = {key: (-np.inf, np.inf) for key in key_idx_map}
        return cls(values, key_idx_map, bounds)
