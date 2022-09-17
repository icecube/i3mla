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

from typing import ClassVar, List, Optional, Tuple

import abc
import dataclasses

import numpy as np
import scipy.optimize

from .configurable import Configurable
from .test_statistics import LLHTestStatistic, FlareStackLLHTestStatistic


@dataclasses.dataclass
class MinimizerFactory(metaclass=abc.ABCMeta):
    """Docstring"""

    @abc.abstractmethod
    def __call__(self, test_statistic: LLHTestStatistic) -> 'Minimizer':
        """Docstring"""


@dataclasses.dataclass(kw_only=True)
class Minimizer(metaclass=abc.ABCMeta):
    """Docstring"""
    test_statistic: LLHTestStatistic

    @abc.abstractmethod
    def __call__(
            self, fitting_params: Optional[List[str]] = None) -> Tuple[float, np.ndarray]:
        """Docstring"""


@dataclasses.dataclass(kw_only=True)
class GridSearchMinimizerFactory(MinimizerFactory, Configurable):
    """Docstring"""
    _config_map: ClassVar[dict] = {
        '_gs_pts': ('GridSearch Points', 5),
        '_min_method': ('Scipy Minimize Method', 'L-BFGS-B'),
    }

    _gs_pts: int = 5
    _min_method: str = 'L-BFGS-B'

    @classmethod
    def from_config(cls, config: dict) -> 'GridSearchMinimizerFactory':
        """Docstring"""
        return cls(**cls._map_kwargs(config))

    def __call__(self, test_statistic: LLHTestStatistic) -> 'GridSearchMinimizer':
        """Docstring"""
        return GridSearchMinimizer(
            _gs_pts=self._gs_pts,
            _min_method=self._min_method,
            test_statistic=test_statistic,
        )


@dataclasses.dataclass(kw_only=True)
class GridSearchMinimizer(Minimizer):
    """Docstring"""
    _gs_pts: int
    _min_method: str

    def __call__(
            self, fitting_params: Optional[List[str]] = None) -> dict:
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
            np.linspace(lo, hi, self._gs_pts)
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
            method=self._min_method,
        )

        best_ts_value = -result.fun
        best_param_values = self._param_values(result.x, fitting_key_idx_map)

        if 'ns' not in fitting_key_idx_map:
            idx = self.test_statistic.params.key_idx_map['ns']
            best_param_values[idx] = self.test_statistic.best_ns

        to_return = {
            'ts': best_ts_value,
            **{
                key: best_param_values[idx]
                for key, idx in self.test_statistic.params.key_idx_map.items()
            },
        }

        for key, val in self.test_statistic.best_time_params.items():
            to_return[key] = val

        return to_return

@dataclasses.dataclass(kw_only=True)
class FlareStackGridSearchMinimizer(GridSearchMinimizer):
    test_statistic: FlareStackLLHTestStatistic
    _return_perflare_ts: bool = False

    def _minimize(
        self,
        point: np.ndarray,
        fitting_key_idx_map: dict,
        fitting_bounds: dict,
    ) -> Tuple[float, np.ndarray]:
        best_dict = super()._minimize(point, fitting_key_idx_map, fitting_bounds)
        if self._return_perflare_ts:
            best_dict['perflare_ts'] = self.test_statistic.best_ts_dict
        return best_dict

@dataclasses.dataclass(kw_only=True)
class FlareStackGridSearchMinimizerFactory(GridSearchMinimizerFactory, Configurable):
    _config_map: ClassVar[dict] = {
        **GridSearchMinimizerFactory._config_map,
        '_return_perflare_ts': ('Return Per-Flare Test Statistic', False),
    }

    _return_perflare_ts: bool = False

    def __call__(
        self,
        test_statistic: FlareStackLLHTestStatistic,
    ) -> FlareStackGridSearchMinimizer:
        """Docstring"""
        return FlareStackGridSearchMinimizer(
            _gs_pts=self._gs_pts,
            _min_method=self._min_method,
            _return_perflare_ts=self._return_perflare_ts,
            test_statistic=test_statistic,
        )
