"""
Top-level analysis code, and functions that are generic enough to not belong
in any class.
"""

__author__ = 'John Evans and Jason Fan'
__copyright__ = 'Copyright 2024'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '1.4.1'
__maintainer__ = 'Jason Fan'
__email__ = 'klfan@terpmail.umd.edu'
__status__ = 'Development'

from typing import List, Optional, Tuple

import abc
import dataclasses

import numpy as np
import scipy.optimize

from . import configurable
from .test_statistics import LLHTestStatistic


@dataclasses.dataclass
class Minimizer(configurable.Configurable):
    """Docstring"""
    test_statistic: LLHTestStatistic

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(
            self, fitting_params: Optional[List[str]] = None) -> Tuple[float, np.ndarray]:
        """Docstring"""


@dataclasses.dataclass
class GridSearchMinimizer(Minimizer):
    """Docstring"""
    def __call__(
            self, fitting_params: Optional[List[str]] = None) -> Tuple[float, np.ndarray]:
        """Docstring"""
        if fitting_params is None:
            fitting_key_idx_map = self.test_statistic.params.key_idx_map
            fitting_bounds = self.test_statistic.params.bounds
        else:
            fitting_key_idx_map = {
                key: val for key, val in self.test_statistic.params.key_idx_map.items()
                if key in fitting_params
            }
            fitting_bounds = {
                key: val for key, val in self.test_statistic.params.bounds.items()
                if key in fitting_params
            }

        if self.test_statistic.n_kept == 0:
            return 0, np.array([(0,)], dtype=[('ns', np.float64)])

        grid = [
            np.linspace(lo, hi, self.config['gridsearch_points'])
            for lo, hi in fitting_bounds.values()
        ]

        points = np.array(np.meshgrid(*grid)).T

        grid_ts_values = np.array([
            self._eval_test_statistic(point, fitting_key_idx_map)
            for point in points
        ])

        return self._minimize(
            points[grid_ts_values.argmin()], fitting_key_idx_map, fitting_bounds)

    def _eval_test_statistic(self, point: np.ndarray, fitting_key_idx_map: dict) -> float:
        """Docstring"""
        return self.test_statistic(self._param_values(point, fitting_key_idx_map))

    def _param_values(self, point: np.ndarray, fitting_key_idx_map: dict) -> np.ndarray:
        """Docstring"""
        param_values = self.test_statistic.params.value_array.copy()

        for i, j in enumerate(fitting_key_idx_map.values()):
            param_values[j] = point[i]

        return param_values

    def _minimize(
        self,
        point: np.ndarray,
        fitting_key_idx_map: dict,
        fitting_bounds: dict,
    ) -> Tuple[float, np.ndarray]:
        """Docstring"""
        result = scipy.optimize.minimize(
            self._eval_test_statistic,
            x0=point,
            args=(fitting_key_idx_map,),
            bounds=fitting_bounds.values(),
            method=self.config['scipy_minimize_method'],
        )

        best_ts_value = -result.fun
        best_param_values = self._param_values(result.x, fitting_key_idx_map)

        if 'ns' not in fitting_key_idx_map:
            idx = self.test_statistic.params.key_idx_map['ns']
            best_param_values[idx] = self.test_statistic.best_ns

        return best_ts_value, best_param_values

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['gridsearch_points'] = 5
        config['scipy_minimize_method'] = 'L-BFGS-B'
        return config
