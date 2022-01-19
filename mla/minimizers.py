"""
Top-level analysis code, and functions that are generic enough to not belong
in any class.
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List, Optional, Tuple
from typing import TYPE_CHECKING

import dataclasses

import numpy as np
import scipy.optimize

if TYPE_CHECKING:
    from .test_statistics import LLHTestStatistic
else:
    LLHTestStatistic = object  # pylint: disable=invalid-name


@dataclasses.dataclass
class Minimizer:
    """Docstring"""
    config: dict
    test_statistic: LLHTestStatistic

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        return {}


@dataclasses.dataclass
class GridSearchMinimizer(Minimizer):
    """Docstring"""
    def __call__(
        self, fitting_params: Optional[List[str]] = None) -> Tuple[float, np.ndarray]:
        """Docstring"""
        if fitting_params is None:
            fitting_params = self.test_statistic.params.names

        if self.test_statistic.n_kept == 0:
            return 0, np.array([(0,)], dtype=[('ns', np.float64)])

        grid = [
            np.linspace(lo, hi, self.config['gridsearch_points'])
            for lo, hi in self.test_statistic.params.bounds[fitting_params]
        ]

        points = np.array(np.meshgrid(*grid)).T

        grid_ts_values = np.array([
            self._eval_test_statistic(point, fitting_params) for point in points
        ])

        return self._minimize(points[grid_ts_values.argmin()], fitting_params)

    def _eval_test_statistic(self, point: np.ndarray, fitting_params: List[str]) -> float:
        """Docstring"""
        return self.test_statistic(self._param_values(point, fitting_params))

    def _param_values(self, point: np.ndarray, fitting_params: List[str]) -> np.ndarray:
        """Docstring"""
        param_values = self.test_statistic.params.values.copy()

        for i, j in enumerate(self.test_statistic.params.key_idx_map[fitting_params]):
            param_values[j] = point[i]

        return param_values

    def _minimize(
        self, point: np.ndarray, fitting_params: List[str]) -> Tuple[float, np.ndarray]:
        """Docstring"""
        result = scipy.optimize.minimize(
            self._eval_test_statistic,
            x0=point,
            bounds=self.test_statistic.params.bounds[fitting_params],
            method=self.config['scipy_minimize_method'],
        )

        best_ts_value = -result.fun
        best_param_values = self._param_values(result.x, fitting_params)

        if 'ns' not in fitting_params:
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
